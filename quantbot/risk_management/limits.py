"""Risk limit configuration."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import List

import yaml


@dataclass
class RiskLimits:
    # pre-trade
    max_order_notional: float = 2_000_000.0
    max_order_qty: float = 100_000.0
    max_order_pct_equity: float = 0.40
    price_collar_pct: float = 0.05
    max_position_pct_equity: float = 0.40
    max_gross_leverage: float = 3.0
    max_net_leverage: float = 2.0
    margin_buffer_pct: float = 0.10  # free equity must exceed initial margin by this fraction of equity
    max_participation: float = 0.10  # of bar volume
    restricted_symbols: List[str] = field(default_factory=list)
    # real-time
    daily_loss_limit_pct: float = 0.025
    max_drawdown_pct: float = 0.20
    var_confidence: float = 0.99
    var_limit_pct: float = 0.04  # 1-day 99% VaR as fraction of equity
    var_lookback: int = 250
    stale_data_seconds: float = 300.0
    # circuit breaker behaviour
    flatten_on_daily_loss: bool = True
    flatten_on_max_drawdown: bool = True
    backtest_auto_reset_bars: int = 5  # simulated "manual" re-enable after N bars in backtests (0 = never)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RiskLimits":
        p = Path(path)
        if not p.exists():
            return cls()
        doc = yaml.safe_load(p.read_text()) or {}
        names = {f.name for f in fields(cls)}
        flat = {}
        for section in doc.values() if all(isinstance(v, dict) for v in doc.values()) else [doc]:
            flat.update(section)
        return cls(**{k: v for k, v in flat.items() if k in names})
