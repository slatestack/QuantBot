"""Performance analytics, including the Probabilistic and Deflated Sharpe
ratios (Bailey & Lopez de Prado, 2012/2014) to account for non-normality and
multiple testing."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def probabilistic_sharpe(returns: np.ndarray, sr_benchmark: float = 0.0) -> float:
    """P(true per-period SR > benchmark) given sample length, skew and kurtosis."""
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 10 or r.std(ddof=1) == 0:
        return np.nan
    sr = r.mean() / r.std(ddof=1)
    g3 = skew(r)
    g4 = kurtosis(r, fisher=False)
    denom = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    return float(norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / denom))


def deflated_sharpe(returns: np.ndarray, n_trials: int = 1, sr_trials_var: Optional[float] = None) -> float:
    """Deflated Sharpe ratio: PSR against the expected maximum SR of ``n_trials``
    independent strategy trials (False Strategy Theorem)."""
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    if n_trials <= 1:
        return probabilistic_sharpe(r, 0.0)
    if sr_trials_var is None:
        sr_trials_var = 1.0 / max(len(r) - 1, 1)  # variance of SR estimator under the null
    emc = 0.5772156649
    sr0 = np.sqrt(sr_trials_var) * ((1 - emc) * norm.ppf(1 - 1 / n_trials) + emc * norm.ppf(1 - 1 / (n_trials * np.e)))
    return probabilistic_sharpe(r, sr0)


def performance_summary(equity: pd.Series, trades: Optional[pd.DataFrame] = None, periods_per_year: int = 252,
                        n_trials: int = 1, gross_exposure: Optional[pd.Series] = None) -> Dict[str, float]:
    eq = equity.dropna()
    if len(eq) < 2:
        return {}
    r = eq.pct_change().dropna()
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    total = eq.iloc[-1] / eq.iloc[0] - 1
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if eq.iloc[-1] > 0 else -1.0
    vol = r.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year) if r.std() > 0 else np.nan
    downside = r[r < 0].std(ddof=1) * np.sqrt(periods_per_year)
    sortino = r.mean() * periods_per_year / downside if downside > 0 else np.nan
    dd = drawdown_series(eq)
    mdd = dd.min()
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan
    # longest drawdown duration (days)
    underwater = dd < 0
    longest = cur = 0
    start = None
    for ts, u in underwater.items():
        if u:
            start = start or ts
            cur = (ts - start).days
            longest = max(longest, cur)
        else:
            start = None
    out = {
        "start": str(eq.index[0].date()), "end": str(eq.index[-1].date()), "years": years,
        "total_return": total, "cagr": cagr, "ann_vol": vol, "sharpe": sharpe, "sortino": sortino,
        "max_drawdown": mdd, "calmar": calmar, "longest_dd_days": longest,
        "hit_rate_daily": float((r > 0).mean()), "skew": float(skew(r)), "kurtosis": float(kurtosis(r)),
        "var_99_hist": float(-np.quantile(r, 0.01)), "psr_vs_0": probabilistic_sharpe(r.to_numpy()),
        "deflated_sharpe": deflated_sharpe(r.to_numpy(), n_trials), "final_equity": float(eq.iloc[-1]),
    }
    if trades is not None and len(trades):
        traded = trades[trades["quantity"] != 0]
        notional = (traded["quantity"].abs() * traded["price"]).sum()
        out["n_trades"] = int(len(traded))
        out["total_commission"] = float(trades["commission"].sum())
        out["total_slippage"] = float(trades["slippage_cost"].sum())
        out["cost_drag_ann"] = float((out["total_commission"] + out["total_slippage"]) / eq.mean() / years)
        out["traded_notional"] = float(notional)
    if gross_exposure is not None and len(gross_exposure):
        out["avg_gross_leverage"] = float((gross_exposure / eq.reindex(gross_exposure.index)).mean())
    return out
