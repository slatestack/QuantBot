"""Covariance estimation and portfolio optimisation.

* Ledoit-Wolf (2004) constant-correlation-free shrinkage to scaled identity.
* Marcenko-Pastur random-matrix denoising (constant residual eigenvalue method).
* Hierarchical Risk Parity (Lopez de Prado, 2016).
* Fractional-Kelly mean-variance: w = gamma * Sigma^-1 mu.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


# ----------------------------------------------------------------- covariance
def ledoit_wolf(returns: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf shrinkage of the sample covariance toward mu*I."""
    X = np.asarray(returns, float)
    n, p = X.shape
    X = X - X.mean(0)
    S = X.T @ X / n
    mu = np.trace(S) / p
    F = mu * np.eye(p)
    d2 = np.sum((S - F) ** 2) / p
    # sum_k ||x_k x_k' - S||_F^2 = sum_k ||x_k||^4 - n ||S||_F^2
    b2_bar = (np.sum(np.sum(X ** 2, axis=1) ** 2) - n * np.sum(S ** 2)) / (n * n * p)
    b2 = min(b2_bar, d2)
    shrink = b2 / d2 if d2 > 0 else 1.0
    return shrink * F + (1 - shrink) * S


def cov_to_corr(cov: np.ndarray):
    sd = np.sqrt(np.diag(cov))
    sd[sd == 0] = 1e-12
    corr = cov / np.outer(sd, sd)
    return np.clip(corr, -1, 1), sd


def marcenko_pastur_denoise(cov: np.ndarray, n_obs: int) -> np.ndarray:
    """Replace eigenvalues below the Marcenko-Pastur upper edge by their average
    (preserving the trace) - removes noise eigen-directions."""
    corr, sd = cov_to_corr(cov)
    p = corr.shape[0]
    q = n_obs / p
    if q <= 1:
        return cov
    evals, evecs = np.linalg.eigh(corr)
    lam_max = (1 + 1 / np.sqrt(q)) ** 2  # sigma^2 = 1 for a correlation matrix (approximation)
    noise = evals < lam_max
    if noise.sum() <= 1:
        return cov
    evals = evals.copy()
    evals[noise] = evals[noise].mean()
    c = evecs @ np.diag(evals) @ evecs.T
    d = np.sqrt(np.diag(c))
    c = c / np.outer(d, d)
    return c * np.outer(sd, sd)


def estimate_covariance(returns: pd.DataFrame, method: str = "ledoit_wolf", halflife: Optional[float] = None) -> pd.DataFrame:
    R = returns.dropna(how="all").fillna(0.0)
    if halflife:
        w = 0.5 ** (np.arange(len(R))[::-1] / halflife)
        w = w / w.sum()
        Rc = R - (R * w[:, None]).sum()
        R = Rc * np.sqrt(w[:, None] * len(R))
    X = R.to_numpy()
    if method == "sample":
        C = np.cov(X, rowvar=False)
    elif method == "ledoit_wolf":
        C = ledoit_wolf(X)
    elif method == "mp_denoise":
        C = marcenko_pastur_denoise(np.cov(X, rowvar=False), len(X))
    elif method == "lw_mp":
        C = marcenko_pastur_denoise(ledoit_wolf(X), len(X))
    else:
        raise ValueError(method)
    C = np.atleast_2d(C)
    return pd.DataFrame(C, index=R.columns, columns=R.columns)


# ------------------------------------------------------------------------ HRP
def _cluster_var(cov: np.ndarray, items: List[int]) -> float:
    c = cov[np.ix_(items, items)]
    ivp = 1.0 / np.diag(c)
    ivp /= ivp.sum()
    return float(ivp @ c @ ivp)


def hrp_weights(cov: pd.DataFrame, linkage_method: str = "single") -> pd.Series:
    """Hierarchical Risk Parity: tree clustering on correlation distance,
    quasi-diagonalisation, recursive bisection with inverse-variance splits."""
    names = list(cov.index)
    C = cov.to_numpy()
    if len(names) == 1:
        return pd.Series([1.0], index=names)
    corr, _ = cov_to_corr(C)
    dist = np.sqrt(np.clip((1 - corr) / 2.0, 0, 1))
    np.fill_diagonal(dist, 0.0)
    Z = linkage(squareform(dist, checks=False), method=linkage_method)
    order = list(leaves_list(Z))
    w = np.ones(len(names))
    clusters = [order]
    while clusters:
        nxt = []
        for cl in clusters:
            if len(cl) <= 1:
                continue
            half = len(cl) // 2
            left, right = cl[:half], cl[half:]
            vl, vr = _cluster_var(C, left), _cluster_var(C, right)
            alpha = 1 - vl / (vl + vr)
            w[left] *= alpha
            w[right] *= 1 - alpha
            nxt += [left, right]
        clusters = nxt
    return pd.Series(w / w.sum(), index=names)


def inverse_vol_weights(cov: pd.DataFrame) -> pd.Series:
    iv = 1 / np.sqrt(np.diag(cov.to_numpy()))
    return pd.Series(iv / iv.sum(), index=cov.index)


def kelly_mvo(mu: pd.Series, cov: pd.DataFrame, fraction: float = 0.3, ridge: float = 1e-6) -> pd.Series:
    """Fractional Kelly: w* = gamma Sigma^-1 mu (per-period mu & Sigma)."""
    C = cov.loc[mu.index, mu.index].to_numpy()
    C = C + ridge * np.trace(C) / len(C) * np.eye(len(C))
    w = fraction * np.linalg.solve(C, mu.to_numpy())
    return pd.Series(w, index=mu.index)


class PortfolioOptimizer:
    """Maps combined forecasts f in [-1,1]^N to raw (pre-sizing) weights.

    method='hrp'   : w_i = f_i * HRP_i * N  (forecast-scaled risk-parity budget)
    method='kelly' : mu_i = IC * sigma_i * f_i (Grinold), w = gamma Sigma^-1 mu
    method='ivol'  : w_i = f_i * invvol_i * N
    """

    def __init__(self, method: str = "hrp", cov_method: str = "lw_mp", lookback: int = 252,
                 cov_halflife: Optional[float] = 90, kelly_fraction: float = 0.3, ic: float = 0.05):
        self.method = method
        self.cov_method = cov_method
        self.lookback = lookback
        self.cov_halflife = cov_halflife
        self.kelly_fraction = kelly_fraction
        self.ic = ic
        self.last_cov: Optional[pd.DataFrame] = None

    def covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        R = returns.iloc[-self.lookback:]
        R = R.loc[:, R.notna().sum() >= max(30, int(0.5 * len(R)))]
        cov = estimate_covariance(R, self.cov_method, self.cov_halflife)
        self.last_cov = cov
        return cov

    def optimize(self, forecasts: Dict[str, float], returns: pd.DataFrame) -> Dict[str, float]:
        syms = [s for s in forecasts if s in returns.columns]
        if not syms:
            return {}
        cov = self.covariance(returns[syms])
        syms = [s for s in syms if s in cov.index]
        if not syms:
            return {}
        cov = cov.loc[syms, syms]
        f = pd.Series({s: forecasts[s] for s in syms})
        if self.method == "kelly":
            sig = np.sqrt(np.diag(cov.to_numpy()))
            mu = self.ic * sig * f
            w = kelly_mvo(mu, cov, self.kelly_fraction)
        else:
            base = hrp_weights(cov) if self.method == "hrp" else inverse_vol_weights(cov)
            w = f * base * len(syms)
        return {s: float(v) for s, v in w.items() if np.isfinite(v)}
