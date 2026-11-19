"""Market-microstructure features: order-book imbalance, order-flow imbalance,
VPIN, Kyle's lambda and low-frequency spread estimators."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.stats import norm

from quantbot.data_engineering.normalizers.schemas import OrderBookSnapshot


def order_book_imbalance(book: OrderBookSnapshot, levels: int = 1) -> float:
    """OBI = (V_bid - V_ask) / (V_bid + V_ask) over the top ``levels``."""
    vb = sum(s for _, s in book.bids[:levels])
    va = sum(s for _, s in book.asks[:levels])
    return float((vb - va) / (vb + va)) if vb + va > 0 else 0.0


def order_flow_imbalance(prev: OrderBookSnapshot, cur: OrderBookSnapshot) -> float:
    """Cont, Kukanov & Stoikov (2014) best-level order-flow imbalance e_n."""
    pb0, qb0 = prev.bids[0]
    pb1, qb1 = cur.bids[0]
    pa0, qa0 = prev.asks[0]
    pa1, qa1 = cur.asks[0]
    e_bid = (qb1 if pb1 >= pb0 else 0.0) - (qb0 if pb1 <= pb0 else 0.0)
    e_ask = (qa1 if pa1 <= pa0 else 0.0) - (qa0 if pa1 >= pa0 else 0.0)
    return float(e_bid - e_ask)


def bulk_volume_classification(price_changes: np.ndarray, volumes: np.ndarray, sigma: Optional[float] = None):
    """Easley, Lopez de Prado & O'Hara bulk volume classification -> (buy_vol, sell_vol)."""
    dp = np.asarray(price_changes, float)
    v = np.asarray(volumes, float)
    s = sigma if sigma is not None else (np.std(dp, ddof=1) if len(dp) > 1 else 1.0)
    s = s if s > 0 else 1.0
    buy_frac = norm.cdf(dp / s)
    return v * buy_frac, v * (1 - buy_frac)


def vpin(price_changes: np.ndarray, volumes: np.ndarray, bucket_volume: float, n_buckets: int = 50) -> float:
    """Volume-Synchronized Probability of Informed Trading over the last ``n_buckets``
    equal-volume buckets (bulk-volume classified)."""
    vol = np.asarray(volumes, float)
    buy, _ = bulk_volume_classification(price_changes, vol)
    buy_frac = np.divide(buy, vol, out=np.full_like(vol, 0.5), where=vol > 0)
    imbalances = []
    acc_b = acc_v = 0.0
    for f, v in zip(buy_frac, vol):
        while v > 1e-12:
            take = min(v, bucket_volume - acc_v)
            acc_b += f * take
            acc_v += take
            v -= take
            if acc_v >= bucket_volume - 1e-9:
                imbalances.append(abs(2 * acc_b - acc_v))  # |V_buy - V_sell|
                acc_b = acc_v = 0.0
    if not imbalances:
        return np.nan
    imb = np.array(imbalances[-n_buckets:])
    return float(imb.sum() / (len(imb) * bucket_volume))


def kyle_lambda(price_changes: np.ndarray, signed_volume: np.ndarray) -> float:
    """OLS slope of price change on signed volume (price impact per unit volume)."""
    x = np.asarray(signed_volume, float)
    y = np.asarray(price_changes, float)
    if len(x) < 3 or np.var(x) == 0:
        return np.nan
    return float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))


def roll_spread(close: Sequence[float]) -> float:
    """Roll (1984) effective spread from negative serial covariance of price changes."""
    dp = np.diff(np.asarray(close, float))
    if len(dp) < 3:
        return np.nan
    c = np.cov(dp[1:], dp[:-1])[0, 1]
    return float(2 * np.sqrt(-c)) if c < 0 else 0.0


def corwin_schultz_spread(high: np.ndarray, low: np.ndarray) -> float:
    """Corwin & Schultz (2012) high-low spread estimator (proportional), averaged."""
    h, l = np.asarray(high, float), np.asarray(low, float)
    if len(h) < 2:
        return np.nan
    beta = np.log(h[1:] / l[1:]) ** 2 + np.log(h[:-1] / l[:-1]) ** 2
    gamma = np.log(np.maximum(h[1:], h[:-1]) / np.minimum(l[1:], l[:-1])) ** 2
    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return float(np.mean(np.maximum(s, 0.0)))
