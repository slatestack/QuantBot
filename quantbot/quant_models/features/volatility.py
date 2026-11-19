"""Range- and close-based volatility estimators (daily, non-annualized unless noted)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def close_to_close(close: np.ndarray, window: int) -> float:
    r = np.diff(np.log(close[-(window + 1):]))
    return float(np.std(r, ddof=1)) if len(r) > 1 else np.nan


def ewma_vol(returns: np.ndarray, halflife: float = 20.0) -> float:
    r = returns[np.isfinite(returns)]
    if len(r) < 2:
        return np.nan
    lam = 0.5 ** (1.0 / halflife)
    w = lam ** np.arange(len(r))[::-1]
    return float(np.sqrt(np.sum(w * r ** 2) / np.sum(w)))


def parkinson(high: np.ndarray, low: np.ndarray) -> float:
    hl = np.log(high / low) ** 2
    return float(np.sqrt(np.mean(hl) / (4 * np.log(2))))


def garman_klass(o, h, l, c) -> float:
    v = 0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2
    return float(np.sqrt(max(np.mean(v), 0.0)))


def rogers_satchell(o, h, l, c) -> float:
    v = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
    return float(np.sqrt(max(np.mean(v), 0.0)))


def yang_zhang(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> float:
    """Yang-Zhang (2000) estimator: overnight + k*open-to-close + (1-k)*Rogers-Satchell.
    Handles opening gaps and drift. Inputs are aligned arrays of length n+1 (first
    element provides the previous close)."""
    o, h, l, c = (np.asarray(x, float) for x in (o, h, l, c))
    n = len(c) - 1
    if n < 2:
        return np.nan
    overnight = np.log(o[1:] / c[:-1])
    open_close = np.log(c[1:] / o[1:])
    rs = (np.log(h[1:] / c[1:]) * np.log(h[1:] / o[1:]) + np.log(l[1:] / c[1:]) * np.log(l[1:] / o[1:]))
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    var = np.var(overnight, ddof=1) + k * np.var(open_close, ddof=1) + (1 - k) * np.mean(rs)
    return float(np.sqrt(max(var, 0.0)))


def yang_zhang_series(df: pd.DataFrame, window: int = 20) -> pd.Series:
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    out = np.full(len(df), np.nan)
    for i in range(window, len(df)):
        out[i] = yang_zhang(o[i - window: i + 1], h[i - window: i + 1], l[i - window: i + 1], c[i - window: i + 1])
    return pd.Series(out, index=df.index, name=f"yz_{window}")


def annualize(daily_vol: float, periods: int = 252) -> float:
    return daily_vol * np.sqrt(periods)
