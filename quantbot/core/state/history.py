"""Hot-tier rolling history buffer.

The buffer only ever contains bars that have already been delivered by the feed,
so any model reading from it is lookahead-free by construction.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from quantbot.data_engineering.normalizers.schemas import Bar

FIELDS = ("open", "high", "low", "close", "volume")


class _Ring:
    __slots__ = ("cap", "n", "data", "ts", "aux")

    def __init__(self, cap: int):
        self.cap = cap
        self.n = 0
        self.data = np.full((len(FIELDS), 2 * cap), np.nan)
        self.ts: List[pd.Timestamp] = []
        self.aux: Dict[str, np.ndarray] = {}

    def append(self, bar: Bar) -> None:
        # double-length array trick: write in [n], compact when full -> O(1) amortized, contiguous views
        if self.n == 2 * self.cap:
            self.data[:, : self.cap] = self.data[:, self.cap:]
            for k in self.aux:
                self.aux[k][: self.cap] = self.aux[k][self.cap:]
                self.aux[k][self.cap:] = np.nan
            self.data[:, self.cap:] = np.nan
            self.ts = self.ts[self.cap:]
            self.n = self.cap
        self.data[:, self.n] = (bar.open, bar.high, bar.low, bar.close, bar.volume)
        for k, v in bar.aux.items():
            if k not in self.aux:
                self.aux[k] = np.full(2 * self.cap, np.nan)
            self.aux[k][self.n] = v
        self.ts.append(bar.ts_event)
        self.n += 1


class HistoryBuffer:
    def __init__(self, capacity: int = 1500):
        self.capacity = capacity
        self._rings: Dict[str, _Ring] = {}
        self.last_ts: Optional[pd.Timestamp] = None

    def update(self, bars: Dict[str, Bar]) -> None:
        for sym, bar in bars.items():
            ring = self._rings.get(sym)
            if ring is None:
                ring = self._rings[sym] = _Ring(self.capacity)
            if ring.ts and bar.ts_event <= ring.ts[-1]:
                raise ValueError(f"Out-of-order bar for {sym}: {bar.ts_event} <= {ring.ts[-1]}")
            ring.append(bar)
            if self.last_ts is None or bar.ts_event > self.last_ts:
                self.last_ts = bar.ts_event

    def symbols(self) -> List[str]:
        return list(self._rings)

    def length(self, symbol: str) -> int:
        r = self._rings.get(symbol)
        return 0 if r is None else min(r.n, self.capacity)

    def get(self, symbol: str, field: str = "close", n: Optional[int] = None) -> np.ndarray:
        r = self._rings.get(symbol)
        if r is None:
            return np.empty(0)
        avail = min(r.n, self.capacity)
        n = avail if n is None else min(n, avail)
        if field in FIELDS:
            row = FIELDS.index(field)
            return r.data[row, r.n - n: r.n].copy()
        arr = r.aux.get(field)
        if arr is None:
            return np.full(n, np.nan)
        return arr[r.n - n: r.n].copy()

    def last(self, symbol: str, field: str = "close") -> float:
        a = self.get(symbol, field, 1)
        return float(a[0]) if len(a) else np.nan

    def timestamps(self, symbol: str, n: Optional[int] = None) -> List[pd.Timestamp]:
        r = self._rings.get(symbol)
        if r is None:
            return []
        avail = min(r.n, self.capacity)
        n = avail if n is None else min(n, avail)
        return r.ts[len(r.ts) - n:]

    def frame(self, symbol: str, n: Optional[int] = None) -> pd.DataFrame:
        cols = {f: self.get(symbol, f, n) for f in FIELDS}
        r = self._rings.get(symbol)
        if r is not None:
            for k in r.aux:
                cols[k] = self.get(symbol, k, n)
        return pd.DataFrame(cols, index=pd.DatetimeIndex(self.timestamps(symbol, n)))

    def close_matrix(self, symbols: List[str], n: int) -> pd.DataFrame:
        """Aligned close-price matrix (by timestamp) of the last ``n`` bars."""
        series = {}
        for s in symbols:
            k = self.length(s)
            if k == 0:
                continue
            m = min(n, k)
            series[s] = pd.Series(self.get(s, "close", m), index=pd.DatetimeIndex(self.timestamps(s, m)))
        if not series:
            return pd.DataFrame()
        return pd.DataFrame(series).sort_index().iloc[-n:]
