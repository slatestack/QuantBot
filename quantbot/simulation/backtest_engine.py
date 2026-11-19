"""Event-driven backtest engine.

Wires the production ``TradingSystem`` to a historical replay feed, a
simulated clock and the simulated exchange - nothing in the decision path is
backtest-specific.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from quantbot.core.bus import InMemoryEventBus
from quantbot.core.clock import SimulatedClock
from quantbot.core.engine import TradingSystem
from quantbot.core.events import EventType
from quantbot.core.instruments import InstrumentRegistry
from quantbot.data_engineering.connectors.historical import HistoricalDataFeed
from quantbot.execution.gateways.simulated import SimulatedGateway
from quantbot.risk_management.limits import RiskLimits
from quantbot.simulation.performance import performance_summary
from quantbot.simulation.slippage_models import CompositeSlippage, FixedBps, SpreadSlippage, SquareRootImpact

log = logging.getLogger(__name__)


def build_slippage(cfg: Dict[str, Any]):
    sc = cfg["execution"].get("slippage", {})
    models = []
    if sc.get("spread", True):
        models.append(SpreadSlippage())
    if sc.get("sqrt_impact_y", 0):
        models.append(SquareRootImpact(sc["sqrt_impact_y"]))
    if sc.get("fixed_bps", 0):
        models.append(FixedBps(sc["fixed_bps"]))
    return CompositeSlippage(models)


@dataclass
class BacktestResult:
    equity: pd.DataFrame
    trades: pd.DataFrame
    metrics: Dict[str, Any]
    targets: pd.DataFrame
    risk_events: pd.DataFrame
    breaker_trips: List[dict]
    shortfall: pd.DataFrame
    model_diagnostics: Dict[str, Any] = field(default_factory=dict)
    rejections: Dict[str, int] = field(default_factory=dict)

    def save(self, out_dir: str | Path, plot: bool = True) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.equity.to_csv(out / "equity_curve.csv")
        self.trades.to_csv(out / "trades.csv", index=False)
        self.targets.to_csv(out / "target_weights.csv")
        self.risk_events.to_csv(out / "risk_events.csv", index=False)
        self.shortfall.to_csv(out / "implementation_shortfall.csv", index=False)

        def clean(v):
            if isinstance(v, (np.floating, float)):
                return None if not np.isfinite(v) else float(v)
            if isinstance(v, (np.integer,)):
                return int(v)
            return v

        payload = {"metrics": {k: clean(v) for k, v in self.metrics.items()},
                   "breaker_trips": [{**t, "ts": str(t["ts"])} for t in self.breaker_trips],
                   "rejections": self.rejections, "model_diagnostics": self.model_diagnostics}
        (out / "metrics.json").write_text(json.dumps(payload, indent=2, default=str))
        if plot:
            self._plot(out / "tearsheet.png")
        return out

    def _plot(self, path: Path) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return
        eq = self.equity["equity"]
        fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1.2, 1.2]})
        axes[0].plot(eq.index, eq.values, color="#1f5fbf", lw=1.3)
        axes[0].set_title(f"Equity  |  CAGR {self.metrics.get('cagr', 0):.1%}  Sharpe {self.metrics.get('sharpe', 0):.2f}"
                          f"  MaxDD {self.metrics.get('max_drawdown', 0):.1%}")
        axes[0].grid(alpha=0.3)
        dd = eq / eq.cummax() - 1
        axes[1].fill_between(dd.index, dd.values, 0, color="#c0392b", alpha=0.5)
        axes[1].set_ylabel("drawdown")
        axes[1].grid(alpha=0.3)
        lev = self.equity["gross_exposure"] / self.equity["equity"]
        axes[2].plot(lev.index, lev.values, color="#555", lw=1)
        axes[2].set_ylabel("gross lev.")
        axes[2].grid(alpha=0.3)
        for t in self.breaker_trips:
            for ax in axes:
                ax.axvline(t["ts"], color="orange", alpha=0.6, lw=0.8)
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)


class BacktestEngine:
    def __init__(self, cfg: Dict[str, Any], frames: Dict[str, pd.DataFrame], registry: InstrumentRegistry,
                 limits: Optional[RiskLimits] = None, n_trials: int = 1, metrics_exporter=None):
        self.cfg = cfg
        self.registry = registry
        self.limits = limits or RiskLimits()
        self.n_trials = n_trials
        universe = set(cfg["universe"].get("equities") or []) | set(cfg["universe"].get("futures") or [])
        missing = universe - set(frames)
        if missing:
            log.warning("no data for %s - they will be ignored", sorted(missing))
        self.feed = HistoricalDataFeed({s: f for s, f in frames.items() if s in universe},
                                       cfg["data"].get("start"), cfg["data"].get("end"))
        self.clock = SimulatedClock()
        self.bus = InMemoryEventBus()
        # gateway subscribes FIRST: it matches orders against the new bar before the strategy sees it
        self.gateway = SimulatedGateway(self.bus, registry, build_slippage(cfg),
                                        cfg["execution"].get("max_participation", 0.1))
        self.bus.subscribe(EventType.MARKET, self.gateway.on_market)
        self.system = TradingSystem(cfg, self.bus, registry, [self.gateway], self.limits, is_backtest=True,
                                    metrics=metrics_exporter)
        pf = self.system.portfolio
        self.gateway.position_lookup = lambda s: pf.positions[s].quantity if s in pf.positions else 0.0

    def run(self, progress: bool = False) -> BacktestResult:
        t0 = time.time()
        n = len(self.feed)
        for i, ev in enumerate(self.feed):
            self.clock.advance_to(ev.ts)
            self.bus.publish(ev)
            self.bus.drain()
            if progress and (i % 250 == 0 or i == n - 1):
                eq = self.system.portfolio.equity
                log.info("[%5d/%d] %s equity=%.0f", i + 1, n, ev.ts.date(), eq)
        elapsed = time.time() - t0
        return self._result(elapsed)

    def _result(self, elapsed: float) -> BacktestResult:
        sysm = self.system
        pf = sysm.portfolio
        eq = pf.history_frame()
        trades = pf.trades_frame()
        if len(trades) == 0:
            trades = pd.DataFrame(columns=["ts", "symbol", "quantity", "price", "commission", "slippage_cost",
                                           "order_id", "is_roll"])
        metrics = performance_summary(eq["equity"], trades, n_trials=self.n_trials,
                                      gross_exposure=eq.get("gross_exposure"))
        metrics.update({"elapsed_sec": elapsed, "bars": len(self.feed), "events": self.bus.dispatched,
                        "borrow_cost": pf.total_borrow_cost, "roll_cost": pf.total_roll_cost,
                        "breaker_trips": len(sysm.breaker.trips), "orders": len(sysm.oms.orders),
                        "rejected_orders": len(sysm.oms.rejected)})
        targets = pd.DataFrame(sysm.targets_log).set_index("ts") if sysm.targets_log else pd.DataFrame()
        diags: Dict[str, Any] = {}
        for m in sysm.models:
            if hasattr(m, "diagnostics"):
                diags[m.name] = m.diagnostics()
            if hasattr(m, "train_log") and m.train_log:
                diags[m.name] = {"retrains": len(m.train_log),
                                 "last_in_sample_ic": m.train_log[-1]["in_sample_ic"]}
        return BacktestResult(eq, trades, metrics, targets, pd.DataFrame(sysm.risk_log), list(sysm.breaker.trips),
                              sysm.oms.implementation_shortfall(), diags, dict(sysm.pre_trade.rejections))
