"""Clocks: simulated (backtest, advanced by the data feed) vs wall clock (live)."""
from __future__ import annotations

import time

import pandas as pd


class Clock:
    def now(self) -> pd.Timestamp:  # pragma: no cover - interface
        raise NotImplementedError


class SimulatedClock(Clock):
    def __init__(self, start: pd.Timestamp | None = None):
        self._now = pd.Timestamp(start) if start is not None else pd.Timestamp(0, tz="UTC")

    def now(self) -> pd.Timestamp:
        return self._now

    def advance_to(self, ts: pd.Timestamp) -> None:
        ts = pd.Timestamp(ts)
        if ts < self._now:
            raise ValueError(f"Clock cannot go backwards: {ts} < {self._now}")
        self._now = ts


class WallClock(Clock):
    def now(self) -> pd.Timestamp:
        return pd.Timestamp.now(tz="UTC")

    @staticmethod
    def sleep(seconds: float) -> None:
        time.sleep(max(0.0, seconds))
