"""Commodity term-structure (carry / roll-yield) and seasonality model.

Roll yield: RY = (F1 - F2) / F1 * 365 / (T2 - T1)   (positive = backwardation).
Signal = blend of time-series carry tanh(RY/scale) and the cross-sectional
carry rank, plus a seasonal component estimated strictly from prior years'
same-calendar-month returns.
"""
from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, List, Optional

import numpy as np

from quantbot.core.state.history import HistoryBuffer
from quantbot.data_engineering.normalizers.schemas import Bar
from quantbot.quant_models.alpha.base import AlphaModel


def roll_yield(f1: float, f2: float, t1: float, t2: float) -> float:
    if not all(np.isfinite(v) for v in (f1, f2, t1, t2)) or f1 <= 0 or t2 <= t1:
        return np.nan
    return (f1 - f2) / f1 * 365.0 / (t2 - t1)


class CommodityTermStructure(AlphaModel):
    name = "carry"

    def __init__(self, symbols: Iterable[str], ry_scale: float = 0.10, smooth_halflife: float = 5.0,
                 xs_weight: float = 0.5, seasonal_weight: float = 0.25, min_seasonal_years: int = 3,
                 name: Optional[str] = None):
        super().__init__(symbols, name)
        self.ry_scale = ry_scale
        self.alpha = 1 - 0.5 ** (1 / smooth_halflife)
        self.xs_weight = xs_weight
        self.seasonal_weight = seasonal_weight
        self.min_years = min_seasonal_years
        self._ry: Dict[str, float] = {}
        self._month_ret: Dict[str, DefaultDict[tuple, float]] = {}
        self._cur_month: Dict[str, tuple] = {}

    @property
    def warmup_bars(self) -> int:
        return 20

    def _update_season(self, sym: str, bar: Bar) -> None:
        key = (bar.ts_event.year, bar.ts_event.month)
        r = bar.aux.get("ret", 0.0)
        store = self._month_ret.setdefault(sym, defaultdict(float))
        store[key] += np.log1p(r) if np.isfinite(r) else 0.0
        self._cur_month[sym] = key

    def _seasonal(self, sym: str, ts) -> float:
        """Average log return of the *upcoming* calendar month across completed past years."""
        store = self._month_ret.get(sym)
        if not store:
            return 0.0
        nxt_month = ts.month % 12 + 1
        vals = [v for (y, m), v in store.items() if m == nxt_month and (y, m) != self._cur_month.get(sym)
                and (y < ts.year or (y == ts.year and m < ts.month))]
        if len(vals) < self.min_years:
            return 0.0
        all_m = np.array([v for k, v in store.items() if k != self._cur_month.get(sym)])
        sd = np.std(all_m) if len(all_m) > 5 else np.nan
        if not np.isfinite(sd) or sd == 0:
            return 0.0
        t = np.mean(vals) / (sd / np.sqrt(len(vals)))
        return float(np.tanh(t / 2.0))

    def on_bar(self, ts, bars: Dict[str, Bar], history: HistoryBuffer) -> Optional[Dict[str, float]]:
        for sym in self.symbols:
            bar = bars.get(sym)
            if bar is None:
                continue
            self._update_season(sym, bar)
            ry = roll_yield(bar.aux.get("f1", np.nan), bar.aux.get("f2", np.nan),
                            bar.aux.get("t1", np.nan), bar.aux.get("t2", np.nan))
            if np.isfinite(ry):
                prev = self._ry.get(sym)
                self._ry[sym] = ry if prev is None else (1 - self.alpha) * prev + self.alpha * ry
        active = [s for s in self.symbols if s in self._ry and history.length(s) >= self.warmup_bars]
        if not active:
            return None
        rys = np.array([self._ry[s] for s in active])
        if len(active) > 1:
            ranks = rys.argsort().argsort().astype(float)
            xs = 2 * ranks / (len(active) - 1) - 1
        else:
            xs = np.zeros(1)
        out = {}
        for i, s in enumerate(active):
            ts_carry = np.tanh(rys[i] / self.ry_scale)
            carry = (1 - self.xs_weight) * ts_carry + self.xs_weight * xs[i]
            sig = (1 - self.seasonal_weight) * carry + self.seasonal_weight * self._seasonal(s, ts)
            out[s] = self.clip(sig)
        return out

    def current_roll_yields(self) -> Dict[str, float]:
        return dict(self._ry)
