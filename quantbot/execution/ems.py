"""Execution Management System: holds parent orders and releases child slices
to the OMS each bar according to the chosen execution algorithm."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import pandas as pd

from quantbot.core.events import Order, TimeInForce, next_order_id
from quantbot.execution.algos import ExecutionAlgo, Immediate
from quantbot.execution.oms import OrderManager

log = logging.getLogger(__name__)


@dataclass
class ParentOrder:
    parent_id: str
    symbol: str
    quantity: float
    slices: List[float]
    tag: str = ""
    released: int = 0
    decision_price: Optional[float] = None
    tif: TimeInForce = TimeInForce.DAY
    child_ids: List[str] = field(default_factory=list)

    @property
    def done(self) -> bool:
        return self.released >= len(self.slices)


class ExecutionManager:
    def __init__(self, oms: OrderManager, algo: Optional[ExecutionAlgo] = None):
        self.oms = oms
        self.algo = algo or Immediate()
        self.parents: Dict[str, ParentOrder] = {}

    def submit_parent(self, symbol: str, quantity: float, ts: pd.Timestamp, ref_price: float, lot: float = 1.0,
                      tag: str = "", algo: Optional[ExecutionAlgo] = None, bar_volume: Optional[float] = None,
                      decision_price: Optional[float] = None, tif: TimeInForce = TimeInForce.DAY,
                      **ctx) -> ParentOrder:
        """``ref_price`` is used for risk/notional checks; ``decision_price`` (the
        tradable price at decision time) for implementation-shortfall accounting."""
        a = algo or self.algo
        slices = a.schedule(quantity, lot=lot, price=ref_price, **ctx)
        p = ParentOrder(next_order_id("P"), symbol, quantity, slices, tag,
                        decision_price=decision_price if decision_price is not None else ref_price, tif=tif)
        self.parents[p.parent_id] = p
        self._release(p, ts, ref_price, bar_volume)  # first slice immediately
        return p

    def _release(self, p: ParentOrder, ts, ref_price: float, bar_volume: Optional[float]) -> None:
        if p.done:
            return
        q = p.slices[p.released]
        p.released += 1
        child = Order(symbol=p.symbol, quantity=q, parent_id=p.parent_id, tag=p.tag, decision_price=p.decision_price,
                      tif=p.tif)
        p.child_ids.append(child.order_id)
        self.oms.submit(child, ts, ref_price, bar_volume)

    def on_bar(self, ts: pd.Timestamp, prices: Dict[str, float], volumes: Dict[str, float]) -> None:
        """``prices`` are risk/notional reference prices."""
        for p in list(self.parents.values()):
            if p.done:
                del self.parents[p.parent_id]
                continue
            if p.symbol in prices:
                self._release(p, ts, prices[p.symbol], volumes.get(p.symbol))

    def cancel_all(self) -> None:
        self.parents.clear()
        self.oms.cancel_all()

    def outstanding_qty(self) -> Dict[str, float]:
        """Unreleased parent quantity per symbol (not yet sent to the OMS)."""
        out: Dict[str, float] = {}
        for p in self.parents.values():
            out[p.symbol] = out.get(p.symbol, 0.0) + sum(p.slices[p.released:])
        return out
