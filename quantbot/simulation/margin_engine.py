"""Futures/equity margin engine: initial & maintenance requirements, margin
calls and forced-liquidation detection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from quantbot.core.state.portfolio import Portfolio


@dataclass
class MarginStatus:
    equity: float
    initial_required: float
    maintenance_required: float
    excess_liquidity: float
    margin_call: bool
    liquidation: bool

    @property
    def utilization(self) -> float:
        return self.initial_required / self.equity if self.equity > 0 else float("inf")


class MarginEngine:
    def __init__(self, liquidation_buffer: float = 0.0):
        self.liquidation_buffer = liquidation_buffer
        self.calls = 0

    def status(self, portfolio: Portfolio) -> MarginStatus:
        eq = portfolio.equity
        im = portfolio.initial_margin_used()
        mm = portfolio.maintenance_margin_used()
        call = eq < im
        liq = eq < mm * (1 + self.liquidation_buffer)
        if call:
            self.calls += 1
        return MarginStatus(eq, im, mm, eq - mm, call, liq)
