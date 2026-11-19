"""Alpha-model interface and the signal aggregator / decay engine."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from quantbot.core.state.history import HistoryBuffer
from quantbot.data_engineering.normalizers.schemas import Bar


class AlphaModel(ABC):
    """An alpha model consumes bars (incrementally) and emits dimensionless
    forecasts in [-1, 1] per symbol. It may only use ``history`` (data already
    delivered) - never anything from the future."""

    name: str = "alpha"

    def __init__(self, symbols: Iterable[str], name: Optional[str] = None):
        self.symbols: List[str] = list(symbols)
        if name:
            self.name = name

    @property
    def warmup_bars(self) -> int:
        return 0

    @abstractmethod
    def on_bar(self, ts: pd.Timestamp, bars: Dict[str, Bar], history: HistoryBuffer) -> Optional[Dict[str, float]]:
        """Update state with the latest bars; return current forecasts or None."""

    @staticmethod
    def clip(x: float) -> float:
        return float(np.clip(x, -1.0, 1.0)) if np.isfinite(x) else 0.0


class SignalAggregator:
    """Combines forecasts from several models with static weights and
    exponential decay of stale signals (half-life measured in bars)."""

    def __init__(self, weights: Dict[str, float], decay_halflife_bars: float = 5.0,
                 diversification_multiplier: float = 1.0):
        self.weights = dict(weights)
        self.halflife = decay_halflife_bars
        self.fdm = diversification_multiplier
        self._latest: Dict[str, Dict[str, float]] = {}
        self._age: Dict[str, int] = {}

    def tick(self) -> None:
        """Advance one bar (ages every stored signal)."""
        for m in self._age:
            self._age[m] += 1

    def update(self, model: str, forecasts: Dict[str, float]) -> None:
        self._latest[model] = {s: float(np.clip(v, -1, 1)) for s, v in forecasts.items() if np.isfinite(v)}
        self._age[model] = 0

    def combined(self) -> Dict[str, float]:
        num: Dict[str, float] = {}
        den: Dict[str, float] = {}
        for m, fc in self._latest.items():
            w = self.weights.get(m, 0.0)
            if w <= 0:
                continue
            decay = 0.5 ** (self._age.get(m, 0) / self.halflife) if self.halflife > 0 else 1.0
            for s, v in fc.items():
                num[s] = num.get(s, 0.0) + w * v * decay
                den[s] = den.get(s, 0.0) + w
        return {s: float(np.clip(self.fdm * num[s] / den[s], -1, 1)) for s in num if den[s] > 0}

    def per_model(self) -> Dict[str, Dict[str, float]]:
        return {m: dict(v) for m, v in self._latest.items()}
