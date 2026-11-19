"""Fixed-width-window fractional differentiation (Lopez de Prado, AFML ch. 5)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ffd_weights(d: float, threshold: float = 1e-4, max_len: int = 10_000) -> np.ndarray:
    w = [1.0]
    k = 1
    while k < max_len:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        w.append(w_k)
        k += 1
    return np.array(w[::-1])  # oldest weight first


def frac_diff_ffd(series: pd.Series, d: float, threshold: float = 1e-4) -> pd.Series:
    """(1 - B)^d X_t with a fixed window; only uses past values (causal)."""
    w = ffd_weights(d, threshold)
    width = len(w)
    x = series.to_numpy(float)
    out = np.full(len(x), np.nan)
    for i in range(width - 1, len(x)):
        window = x[i - width + 1: i + 1]
        if np.all(np.isfinite(window)):
            out[i] = float(np.dot(w, window))
    return pd.Series(out, index=series.index, name=f"ffd_{d:.2f}")


def min_ffd_order(series: pd.Series, p_value: float = 0.05, grid=None, threshold: float = 1e-3) -> float:
    """Smallest d in ``grid`` whose FFD series passes the ADF test - preserves
    maximum memory while achieving stationarity."""
    from quantbot.research.statistical_tests import adf_test

    grid = np.round(np.arange(0.0, 1.01, 0.1), 2) if grid is None else grid
    logp = np.log(series)
    for d in grid:
        fd = frac_diff_ffd(logp, d, threshold).dropna()
        if len(fd) < 50:
            continue
        if adf_test(fd.to_numpy())["p_value"] < p_value:
            return float(d)
    return 1.0
