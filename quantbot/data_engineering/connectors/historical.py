"""Historical replay feed: merges per-symbol bar frames into time-ordered
cross-sectional ``MarketEvent`` snapshots."""
from __future__ import annotations

from typing import Dict, Iterator, Optional

import numpy as np
import pandas as pd

from quantbot.core.events import MarketEvent
from quantbot.data_engineering.normalizers.schemas import BAR_COLUMNS, Bar, normalize_bar_frame


class HistoricalDataFeed:
    def __init__(self, frames: Dict[str, pd.DataFrame], start=None, end=None, latency: pd.Timedelta = pd.Timedelta(0)):
        self.frames: Dict[str, pd.DataFrame] = {}
        for sym, df in frames.items():
            aux_cols = [c for c in df.columns if c not in BAR_COLUMNS and pd.api.types.is_numeric_dtype(df[c])]
            nf = normalize_bar_frame(df[BAR_COLUMNS + aux_cols] if "volume" in df.columns else df)
            if start is not None:
                nf = nf[nf.index >= pd.Timestamp(start, tz="UTC")]
            if end is not None:
                nf = nf[nf.index <= pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)]
            if len(nf):
                self.frames[sym] = nf
        self.latency = latency
        self._index = pd.DatetimeIndex(sorted(set().union(*[f.index for f in self.frames.values()]))) \
            if self.frames else pd.DatetimeIndex([])

    @property
    def timestamps(self) -> pd.DatetimeIndex:
        return self._index

    def __len__(self) -> int:
        return len(self._index)

    def __iter__(self) -> Iterator[MarketEvent]:
        # pre-extract numpy arrays for speed
        cols = {}
        for sym, f in self.frames.items():
            aux = [c for c in f.columns if c not in BAR_COLUMNS]
            pos = pd.Series(np.arange(len(f)), index=f.index)
            cols[sym] = (pos, f[BAR_COLUMNS].to_numpy(float), {a: f[a].to_numpy(float) for a in aux})
        for ts in self._index:
            bars = {}
            for sym, (pos, arr, aux) in cols.items():
                i = pos.get(ts)
                if i is None:
                    continue
                o, h, l, c, v = arr[i]
                bars[sym] = Bar(sym, ts, o, h, l, c, v, ts_recv=ts + self.latency,
                                aux={k: float(a[i]) for k, a in aux.items() if np.isfinite(a[i])})
            yield MarketEvent(ts=ts, bars=bars)
