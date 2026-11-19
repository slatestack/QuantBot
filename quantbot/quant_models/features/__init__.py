"""Feature pipeline helpers shared by alpha models."""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantbot.core.state.history import HistoryBuffer


def returns(history: HistoryBuffer, symbol: str, n: int) -> np.ndarray:
    """Last ``n`` one-period returns. Uses the roll-aware ``ret`` aux field for
    continuous futures (true held-contract return), else close-to-close."""
    r = history.get(symbol, "ret", n)
    if len(r) and np.isfinite(r).sum() >= max(1, int(0.9 * len(r))):
        return np.nan_to_num(r)
    c = history.get(symbol, "close", n + 1)
    if len(c) < 2:
        return np.empty(0)
    return np.diff(c) / c[:-1]


def price_index(history: HistoryBuffer, symbol: str, n: int) -> np.ndarray:
    """Total-return index built from ``returns`` (immune to back-adjustment artefacts)."""
    r = returns(history, symbol, n)
    return np.cumprod(1.0 + r)


def returns_frame(history: HistoryBuffer, symbols, n: int) -> pd.DataFrame:
    """Aligned (by timestamp) matrix of the last ``n`` returns for ``symbols``."""
    cols = {}
    for s in symbols:
        k = min(n, history.length(s) - 1)
        if k < 2:
            continue
        r = returns(history, s, k)
        ts = history.timestamps(s, k)
        cols[s] = pd.Series(r, index=pd.DatetimeIndex(ts))
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols).sort_index().iloc[-n:]
