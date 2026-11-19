"""Real-time portfolio risk: rolling VaR / CVaR (parametric & historical),
intraday (daily) loss and peak-to-trough drawdown tracking."""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from quantbot.core.events import RiskEvent, RiskLevel
from quantbot.risk_management.limits import RiskLimits


def historical_var(returns: np.ndarray, conf: float = 0.99) -> float:
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    return float(-np.quantile(r, 1 - conf)) if len(r) >= 20 else np.nan


def historical_cvar(returns: np.ndarray, conf: float = 0.99) -> float:
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    if len(r) < 20:
        return np.nan
    q = np.quantile(r, 1 - conf)
    tail = r[r <= q]
    return float(-tail.mean()) if len(tail) else float(-q)


def parametric_var(weights: np.ndarray, cov: np.ndarray, conf: float = 0.99) -> float:
    sd = float(np.sqrt(max(weights @ cov @ weights, 0.0)))
    return float(norm.ppf(conf) * sd)


def parametric_cvar(weights: np.ndarray, cov: np.ndarray, conf: float = 0.99) -> float:
    sd = float(np.sqrt(max(weights @ cov @ weights, 0.0)))
    z = norm.ppf(conf)
    return float(sd * norm.pdf(z) / (1 - conf))


class RealTimeRiskMonitor:
    def __init__(self, limits: RiskLimits):
        self.lim = limits
        self.equity_hist: Deque[float] = deque(maxlen=limits.var_lookback + 1)
        self.peak = None
        self.day_start_equity = None
        self.current_day = None
        self.snapshot: Dict[str, float] = {}

    def update(self, ts: pd.Timestamp, equity: float, weights: Optional[Dict[str, float]] = None,
               cov: Optional[pd.DataFrame] = None) -> List[RiskEvent]:
        events: List[RiskEvent] = []
        day = ts.normalize()
        prev_eq = self.equity_hist[-1] if self.equity_hist else None
        if self.current_day is None or day != self.current_day:
            self.current_day = day
            self.day_start_equity = prev_eq if prev_eq is not None else equity
        self.equity_hist.append(equity)
        self.peak = equity if self.peak is None else max(self.peak, equity)

        daily_pnl = equity / self.day_start_equity - 1.0 if self.day_start_equity else 0.0
        dd = equity / self.peak - 1.0 if self.peak else 0.0
        eq = np.asarray(self.equity_hist)
        rets = eq[1:] / eq[:-1] - 1.0 if len(eq) > 1 else np.empty(0)
        hvar = historical_var(rets, self.lim.var_confidence)
        hcvar = historical_cvar(rets, self.lim.var_confidence)
        pvar = pcvar = np.nan
        if weights and cov is not None:
            syms = [s for s in weights if s in cov.index]
            if syms:
                w = np.array([weights[s] for s in syms])
                C = cov.loc[syms, syms].to_numpy()
                pvar = parametric_var(w, C, self.lim.var_confidence)
                pcvar = parametric_cvar(w, C, self.lim.var_confidence)
        self.snapshot = {"equity": equity, "daily_pnl_pct": daily_pnl, "drawdown": dd, "hist_var": hvar,
                         "hist_cvar": hcvar, "param_var": pvar, "param_cvar": pcvar}

        if daily_pnl <= -self.lim.daily_loss_limit_pct:
            events.append(RiskEvent(ts, RiskLevel.HALT, "daily_loss", f"daily loss {daily_pnl:.2%}",
                                    daily_pnl, -self.lim.daily_loss_limit_pct))
        if dd <= -self.lim.max_drawdown_pct:
            events.append(RiskEvent(ts, RiskLevel.HALT, "max_drawdown", f"drawdown {dd:.2%}", dd,
                                    -self.lim.max_drawdown_pct))
        var_now = pvar if np.isfinite(pvar) else hvar
        if np.isfinite(var_now) and var_now > self.lim.var_limit_pct:
            events.append(RiskEvent(ts, RiskLevel.BREACH, "var", f"1d VaR {var_now:.2%} > limit", var_now,
                                    self.lim.var_limit_pct))
        return events

    def reset_peak(self, equity: float) -> None:
        self.peak = equity
