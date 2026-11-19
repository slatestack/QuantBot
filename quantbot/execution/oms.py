"""Order Management System: per-order finite state machine, pre-trade risk
gating, routing, fill accounting and broker reconciliation."""
from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from quantbot.core.events import (FillEvent, Order, OrderStatus, OrderUpdateEvent)
from quantbot.core.state.portfolio import Portfolio
from quantbot.risk_management.pre_trade import PreTradeRisk

log = logging.getLogger(__name__)

S = OrderStatus
VALID_TRANSITIONS = {
    S.PENDING_NEW: {S.ACKNOWLEDGED, S.PARTIALLY_FILLED, S.FILLED, S.REJECTED, S.CANCELED, S.PENDING_CANCEL},
    S.ACKNOWLEDGED: {S.PARTIALLY_FILLED, S.FILLED, S.PENDING_CANCEL, S.CANCELED, S.REJECTED},
    S.PARTIALLY_FILLED: {S.PARTIALLY_FILLED, S.FILLED, S.PENDING_CANCEL, S.CANCELED},
    S.PENDING_CANCEL: {S.CANCELED, S.PARTIALLY_FILLED, S.FILLED},
    S.FILLED: set(), S.CANCELED: set(), S.REJECTED: set(),
}


class InvalidTransition(RuntimeError):
    pass


class OrderManager:
    def __init__(self, router, pre_trade: PreTradeRisk, portfolio: Portfolio):
        self.router = router
        self.pre_trade = pre_trade
        self.portfolio = portfolio
        self.orders: Dict[str, Order] = {}
        self.by_broker_id: Dict[str, str] = {}
        self.rejected: List[Order] = []
        self.listeners: List[Callable[[Order], None]] = []

    # ------------------------------------------------------------- lifecycle
    def _transition(self, order: Order, new: OrderStatus) -> None:
        if new == order.status and new is S.PARTIALLY_FILLED:
            return
        if new not in VALID_TRANSITIONS[order.status]:
            raise InvalidTransition(f"{order.order_id}: {order.status.value} -> {new.value}")
        order.status = new
        for cb in self.listeners:
            cb(order)

    def submit(self, order: Order, ts: pd.Timestamp, ref_price: float, bar_volume: Optional[float] = None,
               bypass_lock_for_reduce: bool = True) -> bool:
        order.created_ts = ts
        if order.decision_price is None:
            order.decision_price = ref_price
        res = self.pre_trade.check(order, self.portfolio, ref_price, self.pending_qty(include_cancelling=False), bar_volume,
                                   risk_reducing_allowed=bypass_lock_for_reduce)
        self.orders[order.order_id] = order
        if not res.approved:
            order.reject_reason = res.reason
            self._transition(order, S.REJECTED)
            self.rejected.append(order)
            log.debug("order rejected %s %s %.4f: %s", order.order_id, order.symbol, order.quantity, res.reason)
            return False
        gw = self.router.route(order)
        order.venue = gw.name
        try:
            gw.send(order)
        except Exception as exc:
            order.reject_reason = f"gateway_error: {exc}"
            self._transition(order, S.REJECTED)
            self.rejected.append(order)
            log.error("gateway send failed: %s", exc)
            return False
        return True

    def cancel(self, order_id: str) -> None:
        o = self.orders.get(order_id)
        if o is None or not o.is_open:
            return
        self._transition(o, S.PENDING_CANCEL)
        gw = self.router.gateway_by_name(o.venue)
        gw.cancel(o)

    def cancel_all(self, symbol: Optional[str] = None) -> None:
        for o in list(self.open_orders()):
            if symbol is None or o.symbol == symbol:
                self.cancel(o.order_id)

    # ---------------------------------------------------------------- events
    def on_order_update(self, ev: OrderUpdateEvent) -> None:
        o = self.orders.get(ev.order_id)
        if o is None:
            return
        if ev.broker_order_id:
            o.broker_order_id = ev.broker_order_id
            self.by_broker_id[ev.broker_order_id] = o.order_id
        if ev.status is S.REJECTED:
            o.reject_reason = ev.reason
        if ev.status is S.ACKNOWLEDGED and o.status is not S.PENDING_NEW:
            return  # late ack (e.g. after a cancel request or fill) carries no new state
        if ev.status != o.status:
            try:
                self._transition(o, ev.status)
            except InvalidTransition as exc:
                log.warning("ignored update: %s", exc)

    def on_fill(self, ev: FillEvent) -> None:
        o = self.orders.get(ev.order_id)
        if o is None:
            return  # e.g. roll cost or externally initiated fill
        new_filled = o.filled_quantity + ev.quantity
        if abs(new_filled) > abs(o.quantity) + 1e-9:
            log.error("overfill on %s: %s > %s", o.order_id, new_filled, o.quantity)
        if new_filled != 0:
            o.avg_fill_price = (o.avg_fill_price * o.filled_quantity + ev.price * ev.quantity) / new_filled
        o.filled_quantity = new_filled
        done = abs(o.remaining) < 1e-9
        target = S.FILLED if done else S.PARTIALLY_FILLED
        if o.status is S.CANCELED:
            return  # late fill after cancel ack: quantities updated, status stays terminal
        self._transition(o, target)

    # ------------------------------------------------------------- queries
    def open_orders(self) -> List[Order]:
        return [o for o in self.orders.values() if o.is_open]

    def pending_qty(self, include_cancelling: bool = True) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for o in self.open_orders():
            if not include_cancelling and o.status is S.PENDING_CANCEL:
                continue
            out[o.symbol] = out.get(o.symbol, 0.0) + o.remaining
        return out

    def implementation_shortfall(self) -> pd.DataFrame:
        rows = []
        for o in self.orders.values():
            if o.filled_quantity and o.decision_price:
                is_bps = np.sign(o.filled_quantity) * (o.avg_fill_price / o.decision_price - 1) * 1e4
                rows.append({"order_id": o.order_id, "symbol": o.symbol, "qty": o.filled_quantity,
                             "decision": o.decision_price, "avg_fill": o.avg_fill_price, "is_bps": is_bps})
        return pd.DataFrame(rows)

    def reconcile(self, broker_positions: Dict[str, float], tol: float = 1e-6) -> Dict[str, tuple]:
        """Compare local ledger with broker drop-copy positions. Returns
        {symbol: (local, broker)} for every mismatch."""
        local = self.portfolio.quantities()
        diffs = {}
        for s in set(local) | set(broker_positions):
            a, b = local.get(s, 0.0), broker_positions.get(s, 0.0)
            if abs(a - b) > tol:
                diffs[s] = (a, b)
        if diffs:
            log.warning("position breaks vs broker: %s", diffs)
        return diffs
