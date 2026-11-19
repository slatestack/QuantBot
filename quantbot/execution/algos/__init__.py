"""Execution algorithms: slice a parent order into a child schedule.

Schedules are expressed in *slices* (one slice released per bar of the
driving feed - minutes intraday, days for daily strategies).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

import numpy as np


class ExecutionAlgo(ABC):
    name = "algo"

    @abstractmethod
    def schedule(self, quantity: float, **ctx) -> List[float]:
        """Return signed child quantities summing to ``quantity``."""

    @staticmethod
    def _round(sizes: np.ndarray, quantity: float, lot: float) -> List[float]:
        if lot <= 0:
            return [float(x) for x in sizes]
        lots = np.floor(np.abs(sizes) / lot)
        short = int(round(abs(quantity) / lot - lots.sum()))
        # distribute remaining lots to the largest fractional parts
        frac = np.abs(sizes) / lot - lots
        for i in np.argsort(-frac)[:max(short, 0)]:
            lots[i] += 1
        out = np.sign(quantity) * lots * lot
        return [float(x) for x in out if x != 0]


class Immediate(ExecutionAlgo):
    name = "immediate"

    def schedule(self, quantity: float, lot: float = 1.0, **ctx) -> List[float]:
        return [quantity]


class TWAP(ExecutionAlgo):
    name = "twap"

    def __init__(self, n_slices: int = 5):
        self.n = max(1, n_slices)

    def schedule(self, quantity: float, lot: float = 1.0, **ctx) -> List[float]:
        return self._round(np.full(self.n, quantity / self.n), quantity, lot)


class VWAP(ExecutionAlgo):
    """Volume-profile weighted slicing. ``profile`` = expected volume per slice
    (e.g. historical intraday curve); defaults to a U-shaped intraday curve."""

    name = "vwap"

    def __init__(self, n_slices: int = 13, profile: Optional[Sequence[float]] = None):
        self.n = n_slices
        self.profile = profile

    def schedule(self, quantity: float, lot: float = 1.0, profile: Optional[Sequence[float]] = None, **ctx):
        p = np.asarray(profile if profile is not None else self.profile if self.profile is not None
                       else 1.0 + 1.5 * np.linspace(-1, 1, self.n) ** 2, float)
        p = p / p.sum()
        return self._round(quantity * p, quantity, lot)


class AlmgrenChriss(ExecutionAlgo):
    """Almgren & Chriss (2000) optimal liquidation trajectory.

    Holdings x_j = X sinh(kappa (T - t_j)) / sinh(kappa T), where
    cosh(kappa tau) = 1 + tau^2 * lambda * sigma^2 / (2 eta_tilde). Higher risk
    aversion ``lam`` front-loads execution; lam -> 0 converges to TWAP.
    """

    name = "almgren_chriss"

    def __init__(self, n_slices: int = 10, risk_aversion: float = 1e-6, eta: float = 2.5e-6, gamma: float = 2.5e-7):
        self.n = max(1, n_slices)
        self.lam = risk_aversion
        self.eta = eta
        self.gamma = gamma

    def trajectory(self, quantity: float, sigma: float, tau: float = 1.0) -> np.ndarray:
        X = abs(quantity)
        eta_t = self.eta - 0.5 * self.gamma * tau
        if eta_t <= 0 or self.lam <= 0 or sigma <= 0:
            return X * (1 - np.arange(self.n + 1) / self.n)
        kt2 = self.lam * sigma ** 2 / eta_t
        kappa = np.arccosh(1 + 0.5 * kt2 * tau ** 2) / tau
        T = self.n * tau
        t = np.arange(self.n + 1) * tau
        return X * np.sinh(kappa * (T - t)) / np.sinh(kappa * T)

    def schedule(self, quantity: float, lot: float = 1.0, sigma: float = 0.02, price: float = 100.0, **ctx):
        x = self.trajectory(quantity, sigma * price)
        trades = -np.diff(x) * np.sign(quantity)
        return self._round(trades, quantity, lot)


def make_algo(name: str, **kw) -> ExecutionAlgo:
    return {"immediate": Immediate, "twap": TWAP, "vwap": VWAP, "almgren_chriss": AlmgrenChriss}[name](**kw)
