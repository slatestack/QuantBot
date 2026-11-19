"""Unified, normalized market-data schemas.

All timestamps are timezone-aware UTC ``pd.Timestamp`` objects with nanosecond
resolution. ``ts_event`` is the exchange/publication timestamp; ``ts_recv`` is
the local ingestion timestamp. Alpha logic must only ever key off ``ts_event``
of data whose ``ts_recv`` is <= the current clock (enforced by the feeds).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def to_utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


@dataclass(frozen=True)
class Bar:
    symbol: str
    ts_event: pd.Timestamp  # bar close time
    open: float
    high: float
    low: float
    close: float
    volume: float
    ts_recv: Optional[pd.Timestamp] = None
    # auxiliary fields, e.g. futures term structure: f1, f2, t1, t2 (days to expiry), roll flag
    aux: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if not (self.high >= max(self.open, self.close) - 1e-9 and self.low <= min(self.open, self.close) + 1e-9):
            raise ValueError(f"Inconsistent OHLC for {self.symbol} @ {self.ts_event}: "
                             f"O={self.open} H={self.high} L={self.low} C={self.close}")
        if self.volume < 0:
            raise ValueError("negative volume")


@dataclass(frozen=True)
class Tick:
    symbol: str
    ts_event: pd.Timestamp
    price: float
    size: float
    side: int = 0  # +1 buyer-initiated, -1 seller-initiated, 0 unknown
    bid: float = np.nan
    ask: float = np.nan
    ts_recv: Optional[pd.Timestamp] = None


@dataclass(frozen=True)
class OrderBookSnapshot:
    symbol: str
    ts_event: pd.Timestamp
    bids: Tuple[Tuple[float, float], ...]  # (price, size) best first
    asks: Tuple[Tuple[float, float], ...]
    ts_recv: Optional[pd.Timestamp] = None

    @property
    def best_bid(self) -> float:
        return self.bids[0][0] if self.bids else np.nan

    @property
    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else np.nan

    @property
    def mid(self) -> float:
        return 0.5 * (self.best_bid + self.best_ask)

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid


BAR_COLUMNS: List[str] = ["open", "high", "low", "close", "volume"]


def normalize_bar_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize an OHLCV frame: lowercase columns, UTC DatetimeIndex, sorted,
    de-duplicated, positive prices, repaired OHLC envelope."""
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    if "adj_close" in out.columns and "close" in out.columns:
        # back-adjust OHLC for splits/dividends using adj_close / close ratio
        ratio = (out["adj_close"] / out["close"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        for c in ("open", "high", "low", "close"):
            out[c] = out[c] * ratio
        out = out.drop(columns=["adj_close"])
    if not isinstance(out.index, pd.DatetimeIndex):
        for cand in ("ts_event", "timestamp", "date", "datetime", "time"):
            if cand in out.columns:
                out = out.set_index(cand)
                break
        out.index = pd.to_datetime(out.index)
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    out.index.name = "ts_event"
    out = out[~out.index.duplicated(keep="last")].sort_index()
    if "volume" not in out.columns:
        out["volume"] = 0.0
    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out[(out[["open", "high", "low", "close"]] > 0).all(axis=1)]
    out["high"] = out[["open", "high", "low", "close"]].max(axis=1)
    out["low"] = out[["open", "high", "low", "close"]].min(axis=1)
    out["volume"] = out["volume"].fillna(0.0).clip(lower=0.0)
    return out
