"""Position sizing: volatility targeting, fractional-Kelly leverage cap and
gross/per-asset limits."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


class PositionSizer:
    def __init__(self, target_vol: float = 0.12, max_gross_leverage: float = 3.0, max_weight: float = 0.35,
                 kelly_fraction: Optional[float] = 0.3, ic: float = 0.05, periods_per_year: int = 252,
                 max_scale: float = 10.0):
        self.target_vol = target_vol
        self.max_gross = max_gross_leverage
        self.max_weight = max_weight
        self.kelly_fraction = kelly_fraction
        self.ic = ic
        self.ppy = periods_per_year
        self.max_scale = max_scale
        self.last_info: Dict[str, float] = {}

    def size(self, raw: Dict[str, float], cov: Optional[pd.DataFrame],
             forecasts: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        if not raw:
            return {}
        w = pd.Series(raw, dtype=float)
        if cov is None or w.abs().sum() == 0:
            return {s: 0.0 for s in raw}
        syms = [s for s in w.index if s in cov.index]
        w = w[syms]
        C = cov.loc[syms, syms].to_numpy()
        port_var = float(w.to_numpy() @ C @ w.to_numpy())
        ann_vol = np.sqrt(max(port_var, 0.0) * self.ppy)
        scale = self.target_vol / ann_vol if ann_vol > 0 else 0.0
        scale = min(scale, self.max_scale)
        info = {"pre_scale_vol": ann_vol, "vol_scale": scale}

        # fractional Kelly cap: the Kelly-optimal leverage along direction w is
        # L* = (w'mu)/(w'Sigma w); never exceed fraction * L*.
        if self.kelly_fraction and forecasts:
            sig = np.sqrt(np.diag(C))
            mu = self.ic * sig * np.array([forecasts.get(s, 0.0) for s in syms])
            edge = float(w.to_numpy() @ mu)
            if edge > 0 and port_var > 0:
                kelly_scale = self.kelly_fraction * edge / port_var
                info["kelly_scale"] = kelly_scale
                scale = min(scale, kelly_scale)
            elif edge <= 0:
                info["kelly_scale"] = 0.0
        w = w * scale
        w = w.clip(-self.max_weight, self.max_weight)
        gross = w.abs().sum()
        if gross > self.max_gross:
            w *= self.max_gross / gross
        info["gross"] = float(w.abs().sum())
        info["net"] = float(w.sum())
        info["ex_ante_vol"] = float(np.sqrt(max(w.to_numpy() @ C @ w.to_numpy(), 0) * self.ppy))
        self.last_info = info
        return {s: float(v) for s, v in w.items()}
