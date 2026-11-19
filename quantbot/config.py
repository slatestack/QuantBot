"""Configuration loading (YAML) with defaults and deep-merge."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List

import yaml

DEFAULTS: Dict[str, Any] = {
    "system": {
        "name": "quantbot",
        "initial_capital": 1_000_000.0,
        "rebalance_every": 1,
        "rebalance_threshold": 0.01,  # min |delta weight| to trade (no-trade buffer)
        "min_trade_notional": 500.0,
        "history_capacity": 1500,
        "cash_rate": 0.0,
        "warmup_bars": 260,
        "seed": 7,
    },
    "universe": {"equities": [], "futures": []},
    "data": {"source": "synthetic", "start": "2014-01-01", "end": "2024-12-31", "path": "data/lake",
             "cache_dir": "data/cache"},
    "models": {},
    "aggregator": {"decay_halflife_bars": 5, "diversification_multiplier": 1.2},
    "portfolio": {"method": "hrp", "cov_method": "lw_mp", "lookback": 252, "cov_halflife": 90, "ic": 0.05,
                  "kelly_fraction": 0.3},
    "sizing": {"target_vol": 0.12, "max_gross_leverage": 3.0, "max_weight": 0.35, "kelly_fraction": 0.3,
               "ic": 0.05},
    "execution": {"algo": "immediate", "algo_params": {}, "max_participation": 0.10,
                  "slippage": {"sqrt_impact_y": 0.7, "spread": True, "fixed_bps": 0.0}},
    "risk_limits_file": "config/risk_limits.yaml",
    "instrument_files": ["config/instruments/equities.yaml", "config/instruments/commodities_futures.yaml"],
    "venues_file": "config/venues.yaml",
    "monitoring": {"prometheus_port": 9108, "enabled": False, "redis_url": None},
    "live": {"poll_seconds": 60, "bar_interval": "1d", "flatten_on_shutdown": False,
             "reconcile_every_polls": 10},
}


def deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        doc = yaml.safe_load(Path(path).read_text()) or {}
        cfg = deep_merge(cfg, doc)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def universe_symbols(cfg: Dict[str, Any], group: Any = "all") -> List[str]:
    eq = list(cfg["universe"].get("equities") or [])
    fu = list(cfg["universe"].get("futures") or [])
    if group in (None, "all"):
        return eq + fu
    if group == "equities":
        return eq
    if group == "futures":
        return fu
    if isinstance(group, (list, tuple)):
        return list(group)
    raise ValueError(f"unknown symbol group {group!r}")
