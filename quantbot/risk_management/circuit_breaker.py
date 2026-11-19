"""Circuit breaker / kill switch.

On a HALT-level risk event: cancel all working orders, flatten risky positions
with aggressive (market) orders, and lock the pre-trade gate until a manual
reset. Also handles data-feed staleness and broker disconnects.
"""
from __future__ import annotations

import enum
import logging
from typing import Callable, List, Optional

import pandas as pd

from quantbot.core.events import RiskEvent, RiskLevel
from quantbot.risk_management.limits import RiskLimits
from quantbot.risk_management.pre_trade import PreTradeRisk

log = logging.getLogger(__name__)


class BreakerState(enum.Enum):
    NORMAL = "NORMAL"
    HALTED = "HALTED"  # no new risk; positions flattened
    PAUSED = "PAUSED"  # no new risk (e.g. stale data / disconnect); positions kept


class CircuitBreaker:
    def __init__(self, limits: RiskLimits, pre_trade: PreTradeRisk, cancel_all: Callable[[], None],
                 flatten_all: Callable[[str], None], is_backtest: bool = False):
        self.lim = limits
        self.pre_trade = pre_trade
        self._cancel_all = cancel_all
        self._flatten_all = flatten_all
        self.state = BreakerState.NORMAL
        self.is_backtest = is_backtest
        self.trips: List[dict] = []
        self._bars_halted = 0
        self.on_reset: Optional[Callable[[], None]] = None

    @property
    def trading_allowed(self) -> bool:
        return self.state is BreakerState.NORMAL

    def on_risk_event(self, ev: RiskEvent) -> None:
        if ev.level is not RiskLevel.HALT or self.state is BreakerState.HALTED:
            return
        flatten = (ev.kind == "daily_loss" and self.lim.flatten_on_daily_loss) or \
                  (ev.kind == "max_drawdown" and self.lim.flatten_on_max_drawdown) or ev.kind == "manual"
        self.trip(ev.ts, f"{ev.kind}: {ev.message}", flatten=flatten)

    def trip(self, ts: pd.Timestamp, reason: str, flatten: bool = True) -> None:
        log.warning("CIRCUIT BREAKER TRIPPED @ %s: %s (flatten=%s)", ts, reason, flatten)
        self.state = BreakerState.HALTED
        self.pre_trade.locked = True
        self._bars_halted = 0
        self.trips.append({"ts": ts, "reason": reason, "flatten": flatten})
        self._cancel_all()
        if flatten:
            self._flatten_all(reason)

    def pause(self, ts: pd.Timestamp, reason: str) -> None:
        if self.state is BreakerState.NORMAL:
            log.warning("trading PAUSED @ %s: %s", ts, reason)
            self.state = BreakerState.PAUSED
            self.pre_trade.locked = True
            self.trips.append({"ts": ts, "reason": f"pause: {reason}", "flatten": False})

    def resume_from_pause(self) -> None:
        if self.state is BreakerState.PAUSED:
            self.state = BreakerState.NORMAL
            self.pre_trade.locked = False

    def reset(self, operator: str = "manual") -> None:
        log.warning("circuit breaker reset by %s", operator)
        self.state = BreakerState.NORMAL
        self.pre_trade.locked = False
        if self.on_reset:
            self.on_reset()

    def on_bar(self) -> None:
        """Backtests can't wait for a human: optionally auto-reset after N bars."""
        if self.state is BreakerState.HALTED and self.is_backtest and self.lim.backtest_auto_reset_bars > 0:
            self._bars_halted += 1
            if self._bars_halted >= self.lim.backtest_auto_reset_bars:
                self.reset("backtest-auto")

    def check_staleness(self, now: pd.Timestamp, last_data: Optional[pd.Timestamp]) -> None:
        if last_data is None:
            return
        age = (now - last_data).total_seconds()
        if age > self.lim.stale_data_seconds:
            self.pause(now, f"stale market data ({age:.0f}s)")
        elif self.state is BreakerState.PAUSED:
            self.resume_from_pause()

    def on_disconnect(self, ts: pd.Timestamp, venue: str) -> None:
        self.pause(ts, f"gateway disconnected: {venue}")
