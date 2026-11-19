"""Asset-class abstraction: equities and commodity futures behind one ``Instrument``."""
from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import yaml


class AssetClass(enum.Enum):
    EQUITY = "EQUITY"
    FUTURE = "FUTURE"


@dataclass(frozen=True)
class Instrument:
    symbol: str
    asset_class: AssetClass = AssetClass.EQUITY
    currency: str = "USD"
    exchange: str = ""
    tick_size: float = 0.01
    multiplier: float = 1.0
    lot_size: float = 1.0
    # margin: equities use rates on notional; futures use $/contract
    initial_margin_rate: float = 0.5
    maintenance_margin_rate: float = 0.25
    initial_margin: float = 0.0
    maintenance_margin: float = 0.0
    # frictions
    commission_per_unit: float = 0.0
    commission_bps: float = 0.0
    min_commission: float = 0.0
    half_spread_bps: float = 1.0
    short_borrow_rate: float = 0.0025  # annual, equities only
    shortable: bool = True
    # futures specifics
    root: str = ""
    roll_days_before_fnd: int = 5
    contract_months: str = ""
    sector: str = ""
    trading_hours: str = ""
    session_break: str = ""

    @property
    def is_future(self) -> bool:
        return self.asset_class is AssetClass.FUTURE

    def notional(self, qty: float, price: float) -> float:
        return qty * price * self.multiplier

    def round_price(self, price: float) -> float:
        if self.tick_size <= 0:
            return price
        return round(round(price / self.tick_size) * self.tick_size, 10)

    def round_qty(self, qty: float) -> float:
        """Round toward zero to a whole number of lots (never increase exposure)."""
        lots = qty / self.lot_size
        lots = math.floor(lots) if lots > 0 else math.ceil(lots)
        return float(lots * self.lot_size)

    def commission(self, qty: float, price: float) -> float:
        c = abs(qty) * self.commission_per_unit + abs(self.notional(qty, price)) * self.commission_bps * 1e-4
        return max(c, self.min_commission) if qty != 0 else 0.0

    def initial_margin_req(self, qty: float, price: float) -> float:
        if self.is_future:
            return abs(qty) * self.initial_margin
        return abs(self.notional(qty, price)) * self.initial_margin_rate

    def maintenance_margin_req(self, qty: float, price: float) -> float:
        if self.is_future:
            return abs(qty) * self.maintenance_margin
        return abs(self.notional(qty, price)) * self.maintenance_margin_rate


@dataclass(frozen=True)
class FutureContract:
    """A single dated futures contract (e.g. CLZ26)."""

    symbol: str
    root: str
    expiry: pd.Timestamp
    first_notice: Optional[pd.Timestamp] = None

    @property
    def last_safe_day(self) -> pd.Timestamp:
        return self.first_notice if self.first_notice is not None else self.expiry


MONTH_CODES = "FGHJKMNQUVXZ"


def contract_symbol(root: str, year: int, month: int) -> str:
    return f"{root}{MONTH_CODES[month - 1]}{year % 100:02d}"


class InstrumentRegistry:
    def __init__(self, instruments: Iterable[Instrument] = ()):
        self._by_symbol: Dict[str, Instrument] = {}
        for ins in instruments:
            self.add(ins)

    def add(self, ins: Instrument) -> None:
        self._by_symbol[ins.symbol] = ins

    def get(self, symbol: str) -> Instrument:
        if symbol not in self._by_symbol:
            # sensible default: a liquid US equity
            self._by_symbol[symbol] = Instrument(symbol=symbol)
        return self._by_symbol[symbol]

    def __contains__(self, symbol: str) -> bool:
        return symbol in self._by_symbol

    def __iter__(self):
        return iter(self._by_symbol.values())

    def symbols(self) -> List[str]:
        return list(self._by_symbol)

    @classmethod
    def from_yaml(cls, *paths: str | Path) -> "InstrumentRegistry":
        reg = cls()
        for p in paths:
            p = Path(p)
            if not p.exists():
                continue
            doc = yaml.safe_load(p.read_text()) or {}
            defaults = doc.get("defaults", {})
            for sym, spec in (doc.get("instruments") or {}).items():
                spec = {**defaults, **(spec or {})}
                ac = AssetClass(spec.pop("asset_class", "EQUITY").upper())
                known = {f for f in Instrument.__dataclass_fields__}
                kwargs = {k: v for k, v in spec.items() if k in known}
                reg.add(Instrument(symbol=sym, asset_class=ac, **kwargs))
        return reg

    def with_overrides(self, symbol: str, **kw) -> None:
        self._by_symbol[symbol] = replace(self.get(symbol), **kw)
