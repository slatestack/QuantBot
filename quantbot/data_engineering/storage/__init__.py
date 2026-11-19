"""Cold-tier storage: Parquet data lake (partitioned by symbol/year) and a
QuestDB InfluxDB-Line-Protocol (ILP) writer for real-time bars/ticks."""
from __future__ import annotations

import logging
import socket
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

from quantbot.data_engineering.normalizers.schemas import Bar, normalize_bar_frame

log = logging.getLogger(__name__)


class ParquetStore:
    def __init__(self, root: str | Path = "data/lake"):
        self.root = Path(root)

    def _dir(self, dataset: str, symbol: str) -> Path:
        return self.root / dataset / f"symbol={symbol}"

    def write_bars(self, symbol: str, df: pd.DataFrame, dataset: str = "bars_1d") -> None:
        df = normalize_bar_frame(df) if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None else df
        for year, part in df.groupby(df.index.year):
            d = self._dir(dataset, symbol) / f"year={year}"
            d.mkdir(parents=True, exist_ok=True)
            f = d / "part.parquet"
            if f.exists():
                old = pd.read_parquet(f)
                part = pd.concat([old, part])
                part = part[~part.index.duplicated(keep="last")].sort_index()
            part.to_parquet(f)

    def read_bars(self, symbol: str, start=None, end=None, dataset: str = "bars_1d") -> pd.DataFrame:
        d = self._dir(dataset, symbol)
        files = sorted(d.glob("year=*/part.parquet"))
        if not files:
            return pd.DataFrame()
        df = pd.concat([pd.read_parquet(f) for f in files]).sort_index()
        if start is not None:
            df = df[df.index >= pd.Timestamp(start, tz="UTC")]
        if end is not None:
            df = df[df.index <= pd.Timestamp(end, tz="UTC")]
        return df

    def symbols(self, dataset: str = "bars_1d") -> list:
        d = self.root / dataset
        return sorted(p.name.split("=", 1)[1] for p in d.glob("symbol=*")) if d.exists() else []


class QuestDBWriter:
    """Minimal ILP-over-TCP writer (QuestDB default port 9009). Never raises on
    the hot path; failures are logged and the connection is retried lazily."""

    def __init__(self, host: str = "localhost", port: int = 9009, table: str = "bars"):
        self.host, self.port, self.table = host, port, table
        self._sock: Optional[socket.socket] = None

    def _connect(self) -> None:
        self._sock = socket.create_connection((self.host, self.port), timeout=2.0)

    @staticmethod
    def _escape(s: str) -> str:
        return s.replace(" ", "\\ ").replace(",", "\\,").replace("=", "\\=")

    def line(self, bar: Bar) -> str:
        ts_ns = int(bar.ts_event.value)
        return (f"{self.table},symbol={self._escape(bar.symbol)} open={bar.open},high={bar.high},"
                f"low={bar.low},close={bar.close},volume={bar.volume} {ts_ns}\n")

    def write(self, bars: Iterable[Bar]) -> None:
        payload = "".join(self.line(b) for b in bars).encode()
        if not payload:
            return
        try:
            if self._sock is None:
                self._connect()
            self._sock.sendall(payload)  # type: ignore[union-attr]
        except OSError as exc:
            log.warning("QuestDB write failed: %s", exc)
            self._sock = None
