"""Message bus implementations.

``InMemoryEventBus`` is a deterministic FIFO bus used in both backtest and live
mode (live producers push through the thread-safe ``put_threadsafe``).
``RedisStreamBridge`` optionally mirrors every event onto Redis Streams for
observability / out-of-process consumers.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
from collections import defaultdict, deque
from dataclasses import asdict, is_dataclass
from typing import Callable, DefaultDict, Deque, List, Optional

from quantbot.core.events import Event, EventType

log = logging.getLogger(__name__)
Handler = Callable[[Event], None]


class InMemoryEventBus:
    def __init__(self, max_events_per_drain: int = 1_000_000):
        self._handlers: DefaultDict[EventType, List[Handler]] = defaultdict(list)
        self._wildcard: List[Handler] = []
        self._queue: Deque[Event] = deque()
        self._external: "queue.Queue[Event]" = queue.Queue()
        self._max = max_events_per_drain
        self._lock = threading.Lock()
        self.dispatched = 0

    def subscribe(self, etype: Optional[EventType], handler: Handler) -> None:
        """Subscribe ``handler``; registration order defines dispatch order.
        ``etype=None`` subscribes to every event (taps, loggers, bridges)."""
        if etype is None:
            self._wildcard.append(handler)
        else:
            self._handlers[etype].append(handler)

    def publish(self, event: Event) -> None:
        self._queue.append(event)

    def put_threadsafe(self, event: Event) -> None:
        """Called from broker/feed threads in live mode."""
        self._external.put(event)

    def _pull_external(self) -> None:
        while True:
            try:
                self._queue.append(self._external.get_nowait())
            except queue.Empty:
                return

    def drain(self) -> int:
        """Dispatch until the queue is empty. Returns number of events dispatched."""
        n = 0
        with self._lock:
            self._pull_external()
            while self._queue:
                ev = self._queue.popleft()
                for h in self._wildcard:
                    h(ev)
                for h in self._handlers.get(ev.type, ()):
                    h(ev)
                n += 1
                if n > self._max:
                    raise RuntimeError("Event storm: too many events in one drain (handler feedback loop?)")
                if not self._queue:
                    self._pull_external()
        self.dispatched += n
        return n

    def __len__(self) -> int:
        return len(self._queue)


def _jsonable(ev: Event) -> str:
    def default(o):
        if hasattr(o, "isoformat"):
            return o.isoformat()
        if hasattr(o, "value"):
            return o.value
        if is_dataclass(o):
            return asdict(o)
        return str(o)

    payload = asdict(ev) if is_dataclass(ev) else {"repr": repr(ev)}
    payload["event_type"] = ev.type.value
    return json.dumps(payload, default=default)


class RedisStreamBridge:
    """Mirror events to Redis Streams (``XADD``). Requires the ``redis`` package."""

    def __init__(self, url: str = "redis://localhost:6379/0", stream: str = "quantbot.events",
                 maxlen: int = 100_000, types: Optional[List[EventType]] = None):
        import redis  # optional dependency

        self._r = redis.Redis.from_url(url)
        self._stream = stream
        self._maxlen = maxlen
        self._types = set(types) if types else None

    def __call__(self, ev: Event) -> None:
        if self._types and ev.type not in self._types:
            return
        try:
            self._r.xadd(self._stream, {"data": _jsonable(ev)}, maxlen=self._maxlen, approximate=True)
        except Exception as exc:  # telemetry must never break trading
            log.warning("redis bridge error: %s", exc)
