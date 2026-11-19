"""Stationarity, cointegration and mean-reversion diagnostics."""
from __future__ import annotations

from typing import Dict

import numpy as np


def adf_test(x: np.ndarray, regression: str = "c", autolag: str = "AIC") -> Dict[str, float]:
    import warnings

    from statsmodels.tsa.stattools import adfuller

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        stat, p, lags, nobs, crit, _ = adfuller(np.asarray(x, float), regression=regression, autolag=autolag)
    return {"stat": float(stat), "p_value": float(p), "lags": int(lags), "nobs": int(nobs),
            "crit_1": crit["1%"], "crit_5": crit["5%"], "crit_10": crit["10%"]}


def engle_granger(y: np.ndarray, x: np.ndarray) -> Dict[str, float]:
    """Engle-Granger two-step cointegration test with MacKinnon p-values."""
    from statsmodels.tsa.stattools import coint

    y, x = np.asarray(y, float), np.asarray(x, float)
    stat, p, crit = coint(y, x)
    beta, alpha = np.polyfit(x, y, 1)
    return {"stat": float(stat), "p_value": float(p), "beta": float(beta), "alpha": float(alpha),
            "crit_5": float(crit[1])}


def ou_fit(spread: np.ndarray, dt: float = 1.0) -> Dict[str, float]:
    """Fit dX = theta(mu - X)dt + sigma dW via exact AR(1) discretisation."""
    s = np.asarray(spread, float)
    s = s[np.isfinite(s)]
    if len(s) < 10:
        return {"theta": np.nan, "mu": np.nan, "sigma": np.nan, "half_life": np.nan, "sigma_eq": np.nan}
    x, y = s[:-1], s[1:]
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    if not (0 < b < 1):
        return {"theta": 0.0, "mu": float(np.mean(s)), "sigma": float(np.std(resid)), "half_life": np.inf,
                "sigma_eq": np.inf}
    theta = -np.log(b) / dt
    mu = a / (1 - b)
    sigma = np.std(resid, ddof=2) * np.sqrt(2 * theta / (1 - b ** 2))
    return {"theta": float(theta), "mu": float(mu), "sigma": float(sigma),
            "half_life": float(np.log(2) / theta), "sigma_eq": float(sigma / np.sqrt(2 * theta))}


def half_life(spread: np.ndarray) -> float:
    return ou_fit(spread)["half_life"]


def hurst_exponent(x: np.ndarray, max_lag: int = 100) -> float:
    """Hurst exponent from the scaling of lagged differences' std: ~0.5 random
    walk, <0.5 mean-reverting, >0.5 trending."""
    x = np.asarray(x, float)
    lags = np.unique(np.logspace(0.3, np.log10(min(max_lag, len(x) // 4)), 20).astype(int))
    lags = lags[lags >= 2]
    tau = np.array([np.std(x[l:] - x[:-l]) for l in lags])
    ok = tau > 0
    if ok.sum() < 3:
        return np.nan
    return float(np.polyfit(np.log(lags[ok]), np.log(tau[ok]), 1)[0])


def variance_ratio(returns: np.ndarray, q: int = 5) -> Dict[str, float]:
    """Lo-MacKinlay variance ratio with heteroskedasticity-robust z-stat."""
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    n = len(r)
    mu = r.mean()
    var1 = np.sum((r - mu) ** 2) / (n - 1)
    rq = np.convolve(r, np.ones(q), "valid")
    m = q * (n - q + 1) * (1 - q / n)
    varq = np.sum((rq - q * mu) ** 2) / m
    vr = varq / var1
    dev2 = (r - mu) ** 2
    theta = 0.0
    for j in range(1, q):
        delta = np.sum(dev2[j:] * dev2[:-j]) / (np.sum(dev2) ** 2)
        theta += (2 * (q - j) / q) ** 2 * delta
    z = (vr - 1) / np.sqrt(theta) if theta > 0 else np.nan
    return {"vr": float(vr), "z": float(z)}
