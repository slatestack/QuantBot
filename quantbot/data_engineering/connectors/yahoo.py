"""Yahoo Finance public chart-API connector (daily bars, split/dividend adjusted).

Intended for research and paper trading only - check the data provider's terms
of use before relying on it. Results are cached as Parquet in ``data/cache``.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from quantbot.data_engineering.normalizers.schemas import normalize_bar_frame

log = logging.getLogger(__name__)
_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}?{qs}"


def fetch_daily(symbol: str, start: str = "2005-01-01", end: Optional[str] = None, retries: int = 3,
                timeout: float = 20.0) -> pd.DataFrame:
    p1 = int(pd.Timestamp(start, tz="UTC").timestamp())
    p2 = int((pd.Timestamp(end, tz="UTC") if end else pd.Timestamp.now(tz="UTC")).timestamp())
    qs = urllib.parse.urlencode({"period1": p1, "period2": p2, "interval": "1d", "events": "div,split",
                                 "includeAdjustedClose": "true"})
    url = _URL.format(sym=urllib.parse.quote(symbol), qs=qs)
    last_exc = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (quantbot research client)"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                doc = json.loads(resp.read().decode())
            res = doc["chart"]["result"][0]
            q = res["indicators"]["quote"][0]
            adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose", q["close"])
            # Yahoo daily timestamps are session open; stamp bars at the US close (21:00 UTC) so a bar
            # is never "known" before the session actually ended.
            idx = pd.to_datetime(res["timestamp"], unit="s", utc=True).normalize() + pd.Timedelta(hours=21)
            df = pd.DataFrame({"open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"],
                               "volume": q["volume"], "adj_close": adj}, index=idx).astype(float)
            df = df[df.index <= pd.Timestamp.now(tz="UTC")]  # drop the still-forming session bar
            return normalize_bar_frame(df)
        except Exception as exc:  # network / parse errors
            last_exc = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Yahoo fetch failed for {symbol}: {last_exc}")


def load_universe(symbols: Iterable[str], start: str = "2005-01-01", end: Optional[str] = None,
                  cache_dir: str | Path = "data/cache", refresh: bool = False) -> Dict[str, pd.DataFrame]:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    out: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        f = cache / f"yahoo_{sym.replace('=', '_').replace('^', '_')}.parquet"
        if f.exists() and not refresh:
            df = pd.read_parquet(f)
        else:
            log.info("downloading %s", sym)
            df = fetch_daily(sym, start, end)
            df.to_parquet(f)
        df = df[(df.index >= pd.Timestamp(start, tz="UTC")) & (df.index <= pd.Timestamp.now(tz="UTC"))]
        if end:
            df = df[df.index <= pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)]
        out[sym] = df
    return out
