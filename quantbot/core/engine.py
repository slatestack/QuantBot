"""The production trading engine shared verbatim by backtest, paper and live modes.

Per bar, the event flow is::

    MarketEvent ─► [SimulatedGateway matches working orders → Fill/OrderUpdate events]
                ─► TradingSystem.on_market: history ← bars, alpha models update, DecisionEvent queued
    FillEvents  ─► OMS state machine ─► Portfolio ledger
    DecisionEvent ─► mark-to-market ─► margin / real-time risk ─► circuit breaker
                  ─► EMS releases child slices
                  ─► (rebalance) aggregator → optimizer → sizer → risk gate → EMS parent orders

Because the decision step only runs after all fills for the bar have been
applied and orders it creates can only fill on a *later* bar, there is no
lookahead in simulation and the identical code path drives live trading.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from quantbot.config import universe_symbols
from quantbot.core.bus import InMemoryEventBus
from quantbot.core.events import (DecisionEvent, EventType, FillEvent, MarketEvent, OrderUpdateEvent, RiskEvent,
                                  RiskLevel, SignalEvent, TargetEvent, TimeInForce)
from quantbot.risk_management.circuit_breaker import BreakerState
from quantbot.core.instruments import InstrumentRegistry
from quantbot.core.state.history import HistoryBuffer
from quantbot.core.state.portfolio import Portfolio
from quantbot.execution.algos import make_algo
from quantbot.execution.ems import ExecutionManager
from quantbot.execution.oms import OrderManager
from quantbot.execution.router import SmartOrderRouter
from quantbot.quant_models.alpha import (AlphaModel, CommodityTermStructure, KalmanPairsStatArb, MLForecaster,
                                         SignalAggregator, TrendFollowing)
from quantbot.quant_models.features import returns_frame
from quantbot.quant_models.portfolio import PortfolioOptimizer, PositionSizer
from quantbot.risk_management import CircuitBreaker, PreTradeRisk, RealTimeRiskMonitor, RiskLimits
from quantbot.simulation.margin_engine import MarginEngine

log = logging.getLogger(__name__)


def build_models(cfg: Dict[str, Any]) -> List[AlphaModel]:
    models: List[AlphaModel] = []
    mcfg = cfg.get("models", {})
    for key, spec in mcfg.items():
        if not spec or not spec.get("enabled", True):
            continue
        params = dict(spec.get("params", {}))
        kind = spec.get("type", key)
        if kind == "trend":
            if "horizons" in params:
                params["horizons"] = [tuple(h) for h in params["horizons"]]
            models.append(TrendFollowing(universe_symbols(cfg, spec.get("symbols", "all")), name=key, **params))
        elif kind == "stat_arb":
            pairs = [tuple(p) for p in spec.get("pairs", [])]
            if pairs:
                models.append(KalmanPairsStatArb(pairs, name=key, **params))
        elif kind == "carry":
            syms = universe_symbols(cfg, spec.get("symbols", "futures"))
            if syms:
                models.append(CommodityTermStructure(syms, name=key, **params))
        elif kind == "ml":
            models.append(MLForecaster(universe_symbols(cfg, spec.get("symbols", "all")), name=key, **params))
        else:
            raise ValueError(f"unknown model type {kind!r}")
    return models


class TradingSystem:
    def __init__(self, cfg: Dict[str, Any], bus: InMemoryEventBus, registry: InstrumentRegistry, gateways: List,
                 limits: Optional[RiskLimits] = None, routes: Optional[Dict[str, List[str]]] = None,
                 is_backtest: bool = True, metrics=None):
        self.cfg = cfg
        self.bus = bus
        self.registry = registry
        self.is_backtest = is_backtest
        self.metrics = metrics
        sc = cfg["system"]
        self.universe = universe_symbols(cfg)
        self.history = HistoryBuffer(sc["history_capacity"])
        self.portfolio = Portfolio(sc["initial_capital"], registry, sc.get("cash_rate", 0.0))
        self.models = build_models(cfg)
        agg = cfg["aggregator"]
        weights = {m.name: float(cfg["models"][m.name].get("weight", 1.0)) for m in self.models}
        self.aggregator = SignalAggregator(weights, agg["decay_halflife_bars"], agg["diversification_multiplier"])
        pc = cfg["portfolio"]
        self.optimizer = PortfolioOptimizer(pc["method"], pc["cov_method"], pc["lookback"], pc.get("cov_halflife"),
                                            pc.get("kelly_fraction", 0.3), pc.get("ic", 0.05))
        self.sizer = PositionSizer(**cfg["sizing"])
        self.limits = limits or RiskLimits()
        self.pre_trade = PreTradeRisk(self.limits, registry)
        self.monitor = RealTimeRiskMonitor(self.limits)
        self.margin = MarginEngine()
        self.router = SmartOrderRouter(registry, gateways, routes)
        self.oms = OrderManager(self.router, self.pre_trade, self.portfolio)
        ec = cfg["execution"]
        self.ems = ExecutionManager(self.oms, make_algo(ec["algo"], **ec.get("algo_params", {})))
        self.breaker = CircuitBreaker(self.limits, self.pre_trade, self.ems.cancel_all, self.flatten_all, is_backtest)
        self.breaker.on_reset = lambda: self.monitor.reset_peak(self.portfolio.equity)
        self.rebalance_every = int(sc["rebalance_every"])
        self.rebalance_threshold = float(sc["rebalance_threshold"])
        self.min_trade_notional = float(sc["min_trade_notional"])
        self.warmup_bars = int(sc.get("warmup_bars", 0))
        self.trading_enabled = True
        self.bar_count = 0
        self.last_bars: Dict[str, Any] = {}
        self.targets_log: List[dict] = []
        self.signal_log: List[dict] = []
        self.risk_log: List[dict] = []
        self.last_data_ts: Optional[pd.Timestamp] = None
        self._flatten_reason: Optional[str] = None

        bus.subscribe(EventType.MARKET, self.on_market)
        bus.subscribe(EventType.ORDER_UPDATE, self.oms.on_order_update)
        bus.subscribe(EventType.FILL, self.on_fill)
        bus.subscribe(EventType.DECISION, self.on_decision)
        bus.subscribe(EventType.RISK, self.on_risk)

    # ---------------------------------------------------------------- prices
    def trade_price(self, sym: str) -> float:
        b = self.last_bars.get(sym)
        return b.close if b is not None else np.nan

    def notional_price(self, sym: str) -> float:
        b = self.last_bars.get(sym)
        if b is None:
            return np.nan
        return b.aux.get("f1", b.close) if self.registry.get(sym).is_future else b.close

    # -------------------------------------------------------------- handlers
    def on_market(self, ev: MarketEvent) -> None:
        self.history.update(ev.bars)
        self.last_bars.update(ev.bars)
        self.last_data_ts = ev.ts
        self.aggregator.tick()
        for m in self.models:
            fc = m.on_bar(ev.ts, ev.bars, self.history)
            if fc:
                self.aggregator.update(m.name, fc)
                self.bus.publish(SignalEvent(ev.ts, m.name, fc))
        self.bus.publish(DecisionEvent(ev.ts))

    def on_fill(self, ev: FillEvent) -> None:
        self.oms.on_fill(ev)
        self.portfolio.on_fill(ev)
        if self.metrics:
            self.metrics.on_fill(ev)

    def on_risk(self, ev: RiskEvent) -> None:
        self.risk_log.append({"ts": ev.ts, "level": ev.level.value, "kind": ev.kind, "message": ev.message})
        self.breaker.on_risk_event(ev)

    def on_decision(self, ev: DecisionEvent) -> None:
        ts = ev.ts
        self.bar_count += 1
        prices = {s: b.close for s, b in self.last_bars.items()}
        notional = {s: self.notional_price(s) for s in self.last_bars}
        self.portfolio.mark(prices, ts, notional_prices=notional)

        ms = self.margin.status(self.portfolio)
        if ms.liquidation and self.portfolio.gross_exposure > 0:
            self.bus.publish(RiskEvent(ts, RiskLevel.HALT, "margin", "maintenance margin breached",
                                       ms.equity, ms.maintenance_required))
        for rev in self.monitor.update(ts, self.portfolio.equity, self.portfolio.weights(), self.optimizer.last_cov):
            self.bus.publish(rev)
        self.breaker.on_bar()
        if self.breaker.state is not BreakerState.HALTED:
            self._flatten_reason = None
        elif self._flatten_reason is not None:
            self._ensure_flat()

        vols = {s: b.volume for s, b in self.last_bars.items()}
        if self.breaker.trading_allowed:
            self.ems.on_bar(ts, notional, vols)
        if (self.trading_enabled and self.breaker.trading_allowed and self.bar_count >= self.warmup_bars
                and self.bar_count % self.rebalance_every == 0):
            self.rebalance(ts)
        if self.metrics:
            self.metrics.on_decision(self)

    # ------------------------------------------------------------- rebalance
    def compute_targets(self) -> Dict[str, float]:
        forecasts = {s: f for s, f in self.aggregator.combined().items() if s in self.universe}
        if not forecasts:
            return {}
        R = returns_frame(self.history, list(forecasts), self.optimizer.lookback)
        if R.empty or len(R) < 60:
            return {}
        raw = self.optimizer.optimize(forecasts, R)
        target = self.sizer.size(raw, self.optimizer.last_cov, forecasts)
        # risk gate on target weights (independent of the sizer's own caps)
        lim = self.limits
        w = pd.Series(target, dtype=float).clip(-lim.max_position_pct_equity * 0.98, lim.max_position_pct_equity * 0.98)
        gross = w.abs().sum()
        if gross > lim.max_gross_leverage * 0.98:
            w *= lim.max_gross_leverage * 0.98 / gross
        net = w.sum()
        if abs(net) > lim.max_net_leverage * 0.98:
            w -= (net - np.sign(net) * lim.max_net_leverage * 0.98) / len(w)
        for s in w.index:
            ins = self.registry.get(s)
            if w[s] < 0 and not ins.is_future and not ins.shortable:
                w[s] = 0.0
        return {s: float(v) for s, v in w.items()}

    def rebalance(self, ts: pd.Timestamp) -> None:
        targets = self.compute_targets()
        if not targets:
            return
        self.signal_log.append({"ts": ts, **{f"f_{k}": v for k, v in self.aggregator.combined().items()}})
        self.targets_log.append({"ts": ts, **targets})
        self.bus.publish(TargetEvent(ts, targets, dict(self.sizer.last_info)))
        equity = self.portfolio.equity
        if equity <= 0:
            return
        current_w = self.portfolio.weights()
        pending = self.oms.pending_qty()
        outstanding = self.ems.outstanding_qty()
        for sym in sorted(set(targets) | set(current_w)):
            tw = targets.get(sym, 0.0)
            px_n = self.notional_price(sym)
            if not np.isfinite(px_n) or px_n <= 0:
                continue
            ins = self.registry.get(sym)
            cur_qty = self.portfolio.positions[sym].quantity if sym in self.portfolio.positions else 0.0
            in_flight = pending.get(sym, 0.0) + outstanding.get(sym, 0.0)
            tgt_qty = ins.round_qty(tw * equity / (px_n * ins.multiplier))
            delta = tgt_qty - cur_qty - in_flight
            if delta == 0:
                continue
            dw = abs(ins.notional(delta, px_n)) / equity
            closing = tgt_qty == 0 and cur_qty != 0
            if not closing and (dw < self.rebalance_threshold or abs(ins.notional(delta, px_n)) < self.min_trade_notional):
                continue
            if in_flight:
                # replace in-flight orders for this symbol with a fresh parent
                for o in self.oms.open_orders():
                    if o.symbol == sym:
                        self.oms.cancel(o.order_id)
                for pid, p in list(self.ems.parents.items()):
                    if p.symbol == sym:
                        del self.ems.parents[pid]
                delta = tgt_qty - cur_qty
                if delta == 0:
                    continue
            sigma = float(np.std(self.history.get(sym, "close", 21)[-21:]) / px_n) if self.history.length(sym) > 21 else 0.02
            self.ems.submit_parent(sym, delta, ts, px_n, lot=ins.lot_size, tag="rebalance",
                                   bar_volume=self.last_bars[sym].volume if sym in self.last_bars else None,
                                   decision_price=self.trade_price(sym), sigma=sigma)

    def flatten_all(self, reason: str = "") -> None:
        """Liquidate every position with persistent (GTC) market orders. While the
        breaker stays HALTED, ``_ensure_flat`` re-issues for any residual."""
        self._flatten_reason = reason
        self._ensure_flat()

    def _ensure_flat(self) -> None:
        ts = self.last_data_ts
        working = {o.symbol for o in self.oms.open_orders() if o.status.value != "PENDING_CANCEL"}
        for sym, pos in list(self.portfolio.positions.items()):
            if pos.is_flat or sym in working:
                continue
            px = self.notional_price(sym)
            self.ems.submit_parent(sym, -pos.quantity, ts, px, lot=self.registry.get(sym).lot_size,
                                   tag=f"flatten:{self._flatten_reason}", algo=make_algo("immediate"),
                                   decision_price=self.trade_price(sym), tif=TimeInForce.GTC)

    def warmup(self, events) -> int:
        """Replay historical events with order generation disabled (live/paper
        start-up) so models and risk state are primed through the same code."""
        prev = self.trading_enabled
        self.trading_enabled = False
        n = 0
        for ev in events:
            self.bus.publish(ev)
            self.bus.drain()
            n += 1
        self.trading_enabled = prev
        return n
