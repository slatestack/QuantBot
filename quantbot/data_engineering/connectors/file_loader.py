"""CSV / Parquet bar loaders for the historical data lake."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

from quantbot.data_engineering.normalizers.schemas import normalize_bar_frame


def load_bars(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)
    return normalize_bar_frame(df)


def load_directory(directory: str | Path, symbols: Optional[Iterable[str]] = None) -> Dict[str, pd.DataFrame]:
    d = Path(directory)
    out: Dict[str, pd.DataFrame] = {}
    for p in sorted(list(d.glob("*.csv")) + list(d.glob("*.parquet"))):
        sym = p.stem
        if symbols is not None and sym not in set(symbols):
            continue
        out[sym] = load_bars(p)
    return out
