"""Live/paper market-data feeds producing ``MarketEvent`` objects on demand.

``PollingBarFeed`` polls any callable returning the latest completed bars and
only emits bars it has not emitted before (monotone ``ts_event``), protecting
the engine from duplicated or revised vendor data.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, Iterable, Optional

import pandas as pd

from quantbot.core.events import MarketEvent
from quantbot.data_engineering.normalizers.schemas import Bar

log = logging.getLogger(__name__)


class PollingBarFeed:
    def __init__(self, symbols: Iterable[str], fetch_latest: Callable[[str], Optional[pd.DataFrame]]):
        self.symbols = list(symbols)
        self._fetch = fetch_latest
        self._last_ts: Dict[str, pd.Timestamp] = {}
        self.last_data_wallclock: Optional[pd.Timestamp] = None

    def warmup_frames(self) -> Dict[str, pd.DataFrame]:
        out = {}
        for s in self.symbols:
            df = self._fetch(s)
            if df is not None and len(df):
                # drop the possibly-incomplete current bar
                out[s] = df.iloc[:-1] if df.index[-1] > pd.Timestamp.now(tz="UTC") else df
        return out

    def poll(self) -> Optional[MarketEvent]:
        bars: Dict[str, Bar] = {}
        now = pd.Timestamp.now(tz="UTC")
        for s in self.symbols:
            try:
                df = self._fetch(s)
            except Exception as exc:
                log.warning("feed error for %s: %s", s, exc)
                continue
            if df is None or not len(df):
                continue
            df = df[df.index <= now]  # only completed bars
            if not len(df):
                continue
            ts = df.index[-1]
            if s in self._last_ts and ts <= self._last_ts[s]:
                continue
            r = df.iloc[-1]
            bars[s] = Bar(s, ts, float(r.open), float(r.high), float(r.low), float(r.close), float(r.volume),
                          ts_recv=now)
            self._last_ts[s] = ts
        if not bars:
            return None
        self.last_data_wallclock = now
        return MarketEvent(ts=max(b.ts_event for b in bars.values()), bars=bars)
