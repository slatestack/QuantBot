"""Synthetic multi-asset market generator for pipeline validation and tests.

Produces a reproducible universe with *realistic* stylized facts — fat tails,
volatility clustering, cross-sectional correlation, slow-moving drift regimes,
a cointegrated equity pair and commodity futures curves with a mean-reverting
convenience yield and seasonality.

IMPORTANT: synthetic data is for verifying that the machinery works (and for
lookahead / risk tests). Backtest results on it are NOT evidence of real-world
profitability.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from quantbot.core.instruments import FutureContract, contract_symbol
from quantbot.data_engineering.futures_roller import FuturesRoller


def _ohlc_from_close(rng: np.random.Generator, close: np.ndarray, daily_vol: np.ndarray,
                     gap_frac: float = 0.3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    prev = np.r_[close[0], close[:-1]]
    open_ = prev * np.exp(rng.normal(0, daily_vol * gap_frac))
    hi_ext = np.abs(rng.normal(0, daily_vol * 0.6))
    lo_ext = np.abs(rng.normal(0, daily_vol * 0.6))
    high = np.maximum(open_, close) * np.exp(hi_ext)
    low = np.minimum(open_, close) * np.exp(-lo_ext)
    return open_, high, low


def _garch_returns(rng, n, ann_vol, persistence=0.97, alpha=0.06, df=5):
    """GARCH(1,1) with Student-t shocks."""
    target_var = (ann_vol / np.sqrt(252)) ** 2
    beta = persistence - alpha
    omega = target_var * (1 - persistence)
    var = np.empty(n)
    r = np.empty(n)
    v = target_var
    t_scale = np.sqrt((df - 2) / df)
    for i in range(n):
        var[i] = v
        z = rng.standard_t(df) * t_scale
        r[i] = np.sqrt(v) * z
        v = omega + alpha * r[i] ** 2 + beta * v
    return r, np.sqrt(var)


@dataclass
class SyntheticUniverse:
    bars: Dict[str, pd.DataFrame]
    contracts: Dict[str, List[FutureContract]]
    contract_bars: Dict[str, Dict[str, pd.DataFrame]]


def generate_universe(start: str = "2014-01-01", end: str = "2024-12-31", n_equities: int = 8,
                      commodities: Tuple[str, ...] = ("CL", "GC", "SI", "HG", "ZC", "NG"),
                      seed: int = 7, roll_method: str = "panama") -> SyntheticUniverse:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end, tz="UTC") + pd.Timedelta(hours=21)  # 21:00 UTC close
    n = len(dates)
    bars: Dict[str, pd.DataFrame] = {}

    # --- equities: market factor + idio + persistent drift regime (time-series momentum)
    mkt, _ = _garch_returns(rng, n, 0.16)
    drift_regime = np.zeros(n)
    for i in range(1, n):
        drift_regime[i] = 0.995 * drift_regime[i - 1] + rng.normal(0, 0.00012)
    for k in range(n_equities):
        beta = rng.uniform(0.6, 1.4)
        idio, ivol = _garch_returns(rng, n, rng.uniform(0.15, 0.35))
        own_drift = np.zeros(n)
        for i in range(1, n):
            own_drift[i] = 0.996 * own_drift[i - 1] + rng.normal(0, 0.00010)
        r = 0.0002 + beta * mkt + idio + drift_regime + own_drift
        close = 50 * np.exp(np.cumsum(r)) * rng.uniform(0.5, 4)
        dv = np.sqrt((beta * 0.16 / np.sqrt(252)) ** 2 + ivol ** 2)
        o, h, l = _ohlc_from_close(rng, close, dv)
        vol = rng.lognormal(np.log(2e6), 0.4, n) * (1 + 20 * np.abs(r))
        bars[f"SYN_EQ{k + 1}"] = pd.DataFrame({"open": o, "high": h, "low": l, "close": close, "volume": vol},
                                              index=dates)

    # --- cointegrated pair: log Y = a + b log X + OU spread
    x_r, x_vol = _garch_returns(rng, n, 0.25)
    log_x = np.log(40) + np.cumsum(x_r + 0.5 * mkt)
    spread = np.zeros(n)
    theta, sig = 0.06, 0.012
    for i in range(1, n):
        spread[i] = spread[i - 1] - theta * spread[i - 1] + sig * rng.normal()
    hedge = 1.0 + 0.2 * np.sin(np.linspace(0, 3 * np.pi, n))  # slowly time-varying beta -> Kalman helps
    log_y = 0.8 + hedge * log_x + spread
    for sym, lp in (("SYN_PAIR_X", log_x), ("SYN_PAIR_Y", log_y)):
        close = np.exp(lp)
        dv = np.full(n, 0.016)
        o, h, l = _ohlc_from_close(rng, close, dv)
        bars[sym] = pd.DataFrame({"open": o, "high": h, "low": l, "close": close,
                                  "volume": rng.lognormal(np.log(3e6), 0.3, n)}, index=dates)

    # --- commodity futures curves
    roller = FuturesRoller(roll_days_before=5, method=roll_method)
    contracts: Dict[str, List[FutureContract]] = {}
    contract_bars: Dict[str, Dict[str, pd.DataFrame]] = {}
    spot0 = {"CL": 60.0, "GC": 1300.0, "SI": 17.0, "HG": 3.0, "ZC": 400.0, "NG": 3.0}
    vols = {"CL": 0.32, "GC": 0.15, "SI": 0.27, "HG": 0.22, "ZC": 0.24, "NG": 0.50}
    r_f = 0.02
    for root in commodities:
        s_r, s_vol = _garch_returns(rng, n, vols.get(root, 0.25))
        y = np.zeros(n)  # convenience yield (annual), OU
        ybar, kappa, ysig = rng.uniform(-0.04, 0.06), 0.01, 0.012
        season_amp = 0.06 if root in ("NG", "ZC") else 0.015
        doy = dates.dayofyear.values
        for i in range(1, n):
            y[i] = y[i - 1] + kappa * (ybar - y[i - 1]) + ysig * rng.normal()
        y_eff = y + season_amp * np.sin(2 * np.pi * doy / 365.25)
        # carry premium: spot drifts up more when curve is backwardated (Koijen et al. 2018)
        tsmom = np.zeros(n)
        for i in range(1, n):
            tsmom[i] = 0.995 * tsmom[i - 1] + rng.normal(0, 0.00012)
        spot_ret = s_r + tsmom + 0.25 * (y_eff - r_f) / 252
        spot = spot0.get(root, 50.0) * np.exp(np.cumsum(spot_ret))
        # list monthly contracts covering the window (+ 2 years ahead)
        cs: List[FutureContract] = []
        frames: Dict[str, pd.DataFrame] = {}
        months = pd.date_range(dates[0].tz_localize(None) - pd.DateOffset(months=1),
                               dates[-1].tz_localize(None) + pd.DateOffset(months=4), freq="MS")
        for m in months:
            expiry = (m - pd.DateOffset(months=1)) + pd.DateOffset(days=19)
            sym = contract_symbol(root, m.year, m.month)
            c = FutureContract(sym, root, pd.Timestamp(expiry), first_notice=pd.Timestamp(expiry))
            live = (dates.tz_localize(None) <= expiry) & (dates.tz_localize(None) >= expiry - pd.DateOffset(months=14))
            if not live.any():
                continue
            idx = dates[live]
            tau = np.maximum((expiry - idx.tz_localize(None)).days.values / 365.0, 1 / 365)
            sp = spot[live]
            fut = sp * np.exp((r_f - y_eff[live]) * tau) * np.exp(rng.normal(0, 0.0015, live.sum()))
            dv = s_vol[live]
            o, h, l = _ohlc_from_close(rng, fut, dv)
            vol = rng.lognormal(np.log(2e5), 0.3, live.sum()) * np.exp(-tau * 4)
            frames[sym] = pd.DataFrame({"open": o, "high": h, "low": l, "close": fut, "volume": vol}, index=idx)
            cs.append(c)
        contracts[root] = cs
        contract_bars[root] = frames
        cont = roller.stitch(cs, frames)
        bars[root] = cont.drop(columns=["contract"])
    return SyntheticUniverse(bars=bars, contracts=contracts, contract_bars=contract_bars)
