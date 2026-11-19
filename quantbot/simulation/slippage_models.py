"""Transaction-cost / market-impact models for simulation.

Every model returns the *fractional* adverse price move for a fill of
``qty`` units (always >= 0); the simulator applies it against the trader.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from quantbot.core.instruments import Instrument


@dataclass
class FillContext:
    price: float
    qty: float
    daily_vol: float  # daily return volatility
    adv: float  # average daily volume (units)
    bar_volume: float


class SlippageModel:
    def impact(self, ins: Instrument, ctx: FillContext) -> float:  # pragma: no cover - interface
        raise NotImplementedError


class FixedBps(SlippageModel):
    def __init__(self, bps: float = 2.0):
        self.bps = bps

    def impact(self, ins, ctx) -> float:
        return self.bps * 1e-4


class SpreadSlippage(SlippageModel):
    """Cross half the bid-ask spread (instrument-level ``half_spread_bps``),
    floored at one tick."""

    def impact(self, ins, ctx) -> float:
        tick_frac = ins.tick_size / ctx.price if ctx.price > 0 else 0.0
        return max(ins.half_spread_bps * 1e-4, 0.5 * tick_frac)


class SquareRootImpact(SlippageModel):
    """Square-root law: dP/P ~= Y * sigma_daily * sqrt(Q / V)."""

    def __init__(self, y: float = 0.7, max_impact: float = 0.05):
        self.y = y
        self.max_impact = max_impact

    def impact(self, ins, ctx) -> float:
        if ctx.adv <= 0 or not np.isfinite(ctx.daily_vol):
            return 0.0
        return float(min(self.y * ctx.daily_vol * np.sqrt(abs(ctx.qty) / ctx.adv), self.max_impact))


class CompositeSlippage(SlippageModel):
    def __init__(self, models: List[SlippageModel]):
        self.models = models

    def impact(self, ins, ctx) -> float:
        return sum(m.impact(ins, ctx) for m in self.models)


def default_slippage() -> SlippageModel:
    return CompositeSlippage([SpreadSlippage(), SquareRootImpact(0.7)])
