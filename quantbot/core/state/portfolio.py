"""Real-time portfolio, positions and PnL ledger.

Accounting conventions
----------------------
* Equities are cash instruments: buying debits ``qty * price`` from cash.
* Futures exchange no notional; open PnL is settled into cash at every mark
  (daily variation margin), tracked via ``last_settle``.
* Equity (NAV) = cash + equity market value + unsettled futures PnL.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from quantbot.core.events import FillEvent
from quantbot.core.instruments import InstrumentRegistry


@dataclass
class Position:
    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    realized_pnl: float = 0.0
    last_price: float = np.nan
    last_settle: float = np.nan  # futures only

    @property
    def is_flat(self) -> bool:
        return abs(self.quantity) < 1e-12


@dataclass
class TradeRecord:
    ts: pd.Timestamp
    symbol: str
    quantity: float
    price: float
    commission: float
    slippage_cost: float
    order_id: str
    is_roll: bool = False


class Portfolio:
    def __init__(self, initial_cash: float, registry: InstrumentRegistry, cash_rate: float = 0.0):
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.registry = registry
        self.cash_rate = cash_rate  # annual interest on cash balance
        self.positions: Dict[str, Position] = {}
        self.trades: List[TradeRecord] = []
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.total_borrow_cost = 0.0
        self.total_roll_cost = 0.0
        self._history: List[dict] = []
        self._last_accrual_ts: Optional[pd.Timestamp] = None
        self.high_water_mark = self.initial_cash
        # price used for notional/exposure (front-contract price for back-adjusted futures)
        self.notional_px: Dict[str, float] = {}

    # ------------------------------------------------------------------ fills
    def position(self, symbol: str) -> Position:
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol)
        return self.positions[symbol]

    def on_fill(self, fill: FillEvent) -> None:
        ins = self.registry.get(fill.symbol)
        pos = self.position(fill.symbol)
        q, px, m = fill.quantity, fill.price, ins.multiplier
        if ins.is_future:
            # settle existing open qty to the fill price, then re-base
            if not pos.is_flat and not np.isnan(pos.last_settle):
                self.cash += pos.quantity * (px - pos.last_settle) * m
            pos.last_settle = px
        else:
            self.cash -= q * px * m

        # realized PnL bookkeeping on average-cost basis (reporting only)
        if pos.is_flat or np.sign(pos.quantity) == np.sign(q):
            new_qty = pos.quantity + q
            pos.avg_price = (pos.avg_price * pos.quantity + px * q) / new_qty if new_qty else 0.0
            pos.quantity = new_qty
        else:
            closing = min(abs(q), abs(pos.quantity)) * np.sign(q)
            pos.realized_pnl += -closing * (px - pos.avg_price) * m
            new_qty = pos.quantity + q
            if abs(new_qty) < 1e-12:
                pos.quantity, pos.avg_price = 0.0, 0.0
            elif np.sign(new_qty) != np.sign(pos.quantity):
                pos.quantity, pos.avg_price = new_qty, px  # flipped
            else:
                pos.quantity = new_qty
        pos.last_price = px

        self.cash -= fill.commission
        self.total_commission += fill.commission
        self.total_slippage += fill.slippage_cost
        if fill.is_roll:
            self.total_roll_cost += fill.commission + fill.slippage_cost
        self.trades.append(TradeRecord(fill.ts, fill.symbol, q, px, fill.commission, fill.slippage_cost,
                                       fill.order_id, fill.is_roll))

    # ---------------------------------------------------------------- marking
    def mark(self, prices: Dict[str, float], ts: pd.Timestamp, settle: bool = True,
             notional_prices: Optional[Dict[str, float]] = None) -> None:
        """Mark positions at ``prices``; settle futures variation margin and
        accrue financing (borrow on shorts, interest on cash)."""
        if notional_prices:
            self.notional_px.update(notional_prices)
        for sym, px in prices.items():
            if sym in self.positions and np.isfinite(px):
                pos = self.positions[sym]
                pos.last_price = px
                ins = self.registry.get(sym)
                if ins.is_future and settle and not pos.is_flat:
                    if np.isnan(pos.last_settle):
                        pos.last_settle = px
                    self.cash += pos.quantity * (px - pos.last_settle) * ins.multiplier
                    pos.last_settle = px
                elif ins.is_future and pos.is_flat:
                    pos.last_settle = px
        self._accrue(ts)
        eq = self.equity
        self.high_water_mark = max(self.high_water_mark, eq)
        self._history.append({
            "ts": ts, "equity": eq, "cash": self.cash, "gross_exposure": self.gross_exposure,
            "net_exposure": self.net_exposure, "n_positions": sum(not p.is_flat for p in self.positions.values()),
        })

    def _accrue(self, ts: pd.Timestamp) -> None:
        if self._last_accrual_ts is None:
            self._last_accrual_ts = ts
            return
        days = (ts - self._last_accrual_ts).total_seconds() / 86400.0
        if days <= 0:
            return
        yf = days / 365.0
        borrow = 0.0
        for sym, pos in self.positions.items():
            ins = self.registry.get(sym)
            if not ins.is_future and pos.quantity < 0 and np.isfinite(pos.last_price):  # equities: last == notional
                borrow += abs(pos.quantity * pos.last_price * ins.multiplier) * ins.short_borrow_rate * yf
        self.cash -= borrow
        self.total_borrow_cost += borrow
        if self.cash_rate and self.cash > 0:
            self.cash += self.cash * self.cash_rate * yf
        self._last_accrual_ts = ts

    # -------------------------------------------------------------- analytics
    def market_value(self, symbol: str) -> float:
        pos = self.positions.get(symbol)
        if pos is None or pos.is_flat or not np.isfinite(pos.last_price):
            return 0.0
        return self.registry.get(symbol).notional(pos.quantity, self.notional_px.get(symbol, pos.last_price))

    @property
    def equity(self) -> float:
        eq = self.cash
        for sym, pos in self.positions.items():
            if pos.is_flat or not np.isfinite(pos.last_price):
                continue
            ins = self.registry.get(sym)
            if ins.is_future:
                ref = pos.last_settle if np.isfinite(pos.last_settle) else pos.last_price
                eq += pos.quantity * (pos.last_price - ref) * ins.multiplier
            else:
                eq += pos.quantity * pos.last_price * ins.multiplier
        return eq

    @property
    def gross_exposure(self) -> float:
        return sum(abs(self.market_value(s)) for s in self.positions)

    @property
    def net_exposure(self) -> float:
        return sum(self.market_value(s) for s in self.positions)

    def weights(self) -> Dict[str, float]:
        eq = self.equity
        if eq <= 0:
            return {s: 0.0 for s in self.positions}
        return {s: self.market_value(s) / eq for s, p in self.positions.items() if not p.is_flat}

    def quantities(self) -> Dict[str, float]:
        return {s: p.quantity for s, p in self.positions.items() if not p.is_flat}

    def initial_margin_used(self) -> float:
        tot = 0.0
        for sym, pos in self.positions.items():
            if not pos.is_flat and np.isfinite(pos.last_price):
                tot += self.registry.get(sym).initial_margin_req(pos.quantity, self.notional_px.get(sym, pos.last_price))
        return tot

    def maintenance_margin_used(self) -> float:
        tot = 0.0
        for sym, pos in self.positions.items():
            if not pos.is_flat and np.isfinite(pos.last_price):
                tot += self.registry.get(sym).maintenance_margin_req(pos.quantity, self.notional_px.get(sym, pos.last_price))
        return tot

    @property
    def drawdown(self) -> float:
        return self.equity / self.high_water_mark - 1.0 if self.high_water_mark > 0 else 0.0

    def history_frame(self) -> pd.DataFrame:
        if not self._history:
            return pd.DataFrame(columns=["equity"])
        df = pd.DataFrame(self._history).set_index("ts")
        return df[~df.index.duplicated(keep="last")]

    def trades_frame(self) -> pd.DataFrame:
        return pd.DataFrame([t.__dict__ for t in self.trades])
