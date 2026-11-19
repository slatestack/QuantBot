"""Resolve the configured data source into per-symbol bar frames."""
from __future__ import annotations

import logging
from typing import Any, Dict

import pandas as pd

from quantbot.config import universe_symbols

log = logging.getLogger(__name__)


def load_frames(cfg: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    dc = cfg["data"]
    src = dc.get("source", "synthetic")
    syms = universe_symbols(cfg)
    if src == "synthetic":
        from quantbot.data_engineering.connectors.synthetic import generate_universe

        u = generate_universe(dc.get("start") or "2014-01-01", dc.get("end") or "2024-12-31",
                              seed=cfg["system"].get("seed", 7))
        return {s: f for s, f in u.bars.items() if s in syms}
    if src == "yahoo":
        from quantbot.data_engineering.connectors.yahoo import load_universe

        start = dc.get("start") or "2005-01-01"
        # fetch one extra year of history before the start for model warm-up
        fetch_start = str((pd.Timestamp(start) - pd.DateOffset(years=0)).date())
        return load_universe(syms, fetch_start, dc.get("end"), dc.get("cache_dir", "data/cache"))
    if src == "parquet_lake":
        from quantbot.data_engineering.storage import ParquetStore

        store = ParquetStore(dc.get("path", "data/lake"))
        return {s: store.read_bars(s, dc.get("start"), dc.get("end")) for s in syms}
    if src == "files":
        from quantbot.data_engineering.connectors.file_loader import load_directory

        return load_directory(dc["path"], syms)
    raise ValueError(f"unknown data source {src!r}")
