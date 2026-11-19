"""Simulated exchange / paper broker.

* Orders submitted at time t are only eligible to fill on bars with
  ``ts_event > t`` (strict no-lookahead).
* Market orders fill at the next bar's open +/- spread & square-root impact.
* Limit orders fill only if the bar trades *through* the limit by at least one
  tick (conservative queue-position model).
* Fill size is capped at ``max_participation`` x bar volume; remainders stay
  working (partial fills) until filled, cancelled or the order expires.
* Continuous-futures roll days charge the cost of rolling both legs.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Callable, Dict, List, Optional

import numpy as np

from quantbot.core.events import (FillEvent, MarketEvent, Order, OrderStatus, OrderType, OrderUpdateEvent,
                                  TimeInForce)
from quantbot.core.instruments import InstrumentRegistry
from quantbot.simulation.slippage_models import FillContext, SlippageModel, default_slippage

log = logging.getLogger(__name__)


class SimulatedGateway:
    def __init__(self, bus, registry: InstrumentRegistry, slippage: Optional[SlippageModel] = None,
                 max_participation: float = 0.10, name: str = "SIM",
                 position_lookup: Optional[Callable[[str], float]] = None, vol_halflife: float = 20.0,
                 adv_halflife: float = 20.0, max_order_age_bars: int = 3):
        self.bus = bus
        self.registry = registry
        self.slippage = slippage or default_slippage()
        self.max_participation = max_participation
        self.name = name
        self.connected = True
        self.position_lookup = position_lookup
        self.working: Dict[str, Order] = {}
        self._age: Dict[str, int] = {}
        self._vol: Dict[str, float] = {}
        self._adv: Dict[str, float] = {}
        self._prev_close: Dict[str, float] = {}
        self._a_vol = 1 - 0.5 ** (1 / vol_halflife)
        self._a_adv = 1 - 0.5 ** (1 / adv_halflife)
        self.max_order_age = max_order_age_bars

    # ------------------------------------------------------------ gateway API
    def connect(self) -> None:
        self.connected = True

    def send(self, order: Order) -> None:
        self.working[order.order_id] = order
        self._age[order.order_id] = 0
        self.bus.publish(OrderUpdateEvent(order.created_ts, order.order_id, OrderStatus.ACKNOWLEDGED,
                                          broker_order_id=f"SIM-{order.order_id}"))

    def cancel(self, order: Order) -> None:
        if self.working.pop(order.order_id, None) is not None:
            self._age.pop(order.order_id, None)
            self.bus.publish(OrderUpdateEvent(order.created_ts, order.order_id, OrderStatus.CANCELED))

    # ------------------------------------------------------------- matching
    def _update_stats(self, ev: MarketEvent) -> None:
        for sym, b in ev.bars.items():
            pc = self._prev_close.get(sym)
            r = b.aux.get("ret", (b.close / pc - 1.0) if pc else np.nan)
            if np.isfinite(r):
                v = self._vol.get(sym)
                self._vol[sym] = r * r if v is None else (1 - self._a_vol) * v + self._a_vol * r * r
            if b.volume > 0:
                a = self._adv.get(sym)
                self._adv[sym] = b.volume if a is None else (1 - self._a_adv) * a + self._a_adv * b.volume
            self._prev_close[sym] = b.close

    def on_market(self, ev: MarketEvent) -> None:
        """Match working orders against the new bar *before* stats update with it."""
        for oid, order in list(self.working.items()):
            bar = ev.bars.get(order.symbol)
            if bar is None or (order.created_ts is not None and bar.ts_event <= order.created_ts):
                continue
            self._match(order, bar, ev.ts)
            if oid in self.working:
                self._age[oid] += 1
                expired = order.tif is TimeInForce.IOC or (order.tif is TimeInForce.DAY and
                                                           self._age[oid] >= self.max_order_age)
                if expired:
                    self.working.pop(oid)
                    self.bus.publish(OrderUpdateEvent(ev.ts, oid, OrderStatus.CANCELED, reason="expired"))
        self._charge_rolls(ev)
        self._update_stats(ev)

    def _match(self, order: Order, bar, ts) -> None:
        ins = self.registry.get(order.symbol)
        remaining = order.remaining
        if abs(remaining) < 1e-12:
            self.working.pop(order.order_id, None)
            return
        cap = self.max_participation * bar.volume if bar.volume > 0 else abs(remaining)
        qty = np.sign(remaining) * min(abs(remaining), cap)
        if ins.lot_size > 0:
            qty = ins.round_qty(qty)
        if qty == 0:
            return
        side = np.sign(qty)
        if order.order_type is OrderType.MARKET:
            ref = bar.open
            dv = np.sqrt(self._vol.get(order.symbol, (0.02) ** 2))
            ctx = FillContext(ref, qty, dv, self._adv.get(order.symbol, bar.volume), bar.volume)
            imp = self.slippage.impact(ins, ctx)
            px = ref * (1 + side * imp)
            px = min(max(px, bar.low), bar.high) if imp < 0.02 else px  # keep inside the bar unless huge impact
            slip_cost = abs(qty) * abs(px - ref) * ins.multiplier
        else:
            lim = order.limit_price
            through = ins.tick_size
            if side > 0 and bar.low <= lim - through:
                px = min(bar.open, lim)
            elif side < 0 and bar.high >= lim + through:
                px = max(bar.open, lim)
            else:
                return
            slip_cost = 0.0
        comm = ins.commission(qty, px)
        self.bus.publish(FillEvent(ts, order.order_id, order.symbol, float(qty), float(px), comm, slip_cost, self.name))
        if abs(order.remaining - qty) < 1e-9:
            self.working.pop(order.order_id, None)
        # NOTE: the OMS updates order.filled_quantity when it processes the FillEvent

    def _charge_rolls(self, ev: MarketEvent) -> None:
        if self.position_lookup is None:
            return
        for sym, bar in ev.bars.items():
            if bar.aux.get("roll", 0.0) != 1.0:
                continue
            ins = self.registry.get(sym)
            q = self.position_lookup(sym)
            if not ins.is_future or q == 0:
                continue
            px = bar.aux.get("f1", bar.close)
            spread_cost = 2 * abs(q) * px * ins.multiplier * ins.half_spread_bps * 1e-4
            comm = 2 * ins.commission(q, px)
            self.bus.publish(FillEvent(ev.ts, f"ROLL-{sym}", sym, 0.0, bar.close, comm + spread_cost, 0.0, self.name,
                                       is_roll=True))

    # --------------------------------------------------------------- queries
    def positions(self) -> Dict[str, float]:
        return {}
