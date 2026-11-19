"""Multi-horizon CTA trend following with a Donchian breakout filter.

Signal construction follows Baz, Granger, Harvey, Le Roux & Rattray (2015):
for each (short, long) EWMA pair, x = (EMA_s - EMA_l) / (P * sigma_ret),
y = x / std(x), response u = y * exp(-y^2/4) / 0.89; average across horizons.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from quantbot.core.state.history import HistoryBuffer
from quantbot.data_engineering.normalizers.schemas import Bar
from quantbot.quant_models.alpha.base import AlphaModel


class _SymState:
    def __init__(self, n_h: int, norm_window: int, breakout_window: int):
        self.index = 1.0
        self.prev_close: Optional[float] = None
        self.ema_s = [None] * n_h
        self.ema_l = [None] * n_h
        self.var = None
        self.xs: List[Deque[float]] = [deque(maxlen=norm_window) for _ in range(n_h)]
        self.levels: Deque[float] = deque(maxlen=breakout_window)
        self.n = 0


class TrendFollowing(AlphaModel):
    name = "trend"

    def __init__(self, symbols: Iterable[str], horizons: Sequence[Tuple[int, int]] = ((8, 24), (16, 48), (32, 96)),
                 vol_halflife: int = 30, norm_window: int = 252, breakout_window: int = 100,
                 breakout_weight: float = 0.25, name: Optional[str] = None):
        super().__init__(symbols, name)
        self.horizons = list(horizons)
        self.vol_alpha = 1 - 0.5 ** (1 / vol_halflife)
        self.norm_window = norm_window
        self.breakout_window = breakout_window
        self.breakout_weight = breakout_weight
        self._st: Dict[str, _SymState] = {}

    @property
    def warmup_bars(self) -> int:
        return max(l for _, l in self.horizons) + 60

    def on_bar(self, ts, bars: Dict[str, Bar], history: HistoryBuffer) -> Optional[Dict[str, float]]:
        out = {}
        for sym in self.symbols:
            bar = bars.get(sym)
            if bar is None:
                continue
            st = self._st.setdefault(sym, _SymState(len(self.horizons), self.norm_window, self.breakout_window))
            r = bar.aux.get("ret")
            if r is None:
                r = 0.0 if st.prev_close is None else bar.close / st.prev_close - 1.0
            st.prev_close = bar.close
            if not np.isfinite(r):
                r = 0.0
            st.index *= 1.0 + r
            st.n += 1
            st.var = r * r if st.var is None else (1 - self.vol_alpha) * st.var + self.vol_alpha * r * r
            st.levels.append(st.index)
            p = st.index
            sig_ret = np.sqrt(st.var) if st.var and st.var > 0 else np.nan
            us = []
            for k, (hs, hl) in enumerate(self.horizons):
                a_s, a_l = 2 / (hs + 1), 2 / (hl + 1)
                st.ema_s[k] = p if st.ema_s[k] is None else (1 - a_s) * st.ema_s[k] + a_s * p
                st.ema_l[k] = p if st.ema_l[k] is None else (1 - a_l) * st.ema_l[k] + a_l * p
                if not np.isfinite(sig_ret) or sig_ret <= 0:
                    continue
                x = (st.ema_s[k] - st.ema_l[k]) / (p * sig_ret)
                st.xs[k].append(x)
                if len(st.xs[k]) >= 60:
                    sd = np.std(st.xs[k])
                    if sd > 0:
                        y = x / sd
                        us.append(y * np.exp(-y * y / 4) / 0.89)
            if st.n < self.warmup_bars or not us:
                continue
            trend = float(np.mean(us))
            lv = np.asarray(st.levels)
            hi, lo = lv.max(), lv.min()
            breakout = 0.0 if hi <= lo else (p - (hi + lo) / 2) / ((hi - lo) / 2)
            sig = (1 - self.breakout_weight) * trend + self.breakout_weight * breakout
            out[sym] = self.clip(sig)
        return out or None
