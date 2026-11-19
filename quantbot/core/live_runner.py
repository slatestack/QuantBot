"""Paper / live trading loop.

The live runner reuses the exact ``TradingSystem`` used in backtests. Only the
edges differ: a polling market-data feed, a wall clock, real (or simulated)
broker gateways, start-up warm-up replay, broker reconciliation, staleness
watchdog and Prometheus telemetry.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from quantbot.config import universe_symbols
from quantbot.core.bus import InMemoryEventBus
from quantbot.core.clock import WallClock
from quantbot.core.engine import TradingSystem
from quantbot.core.events import EventType
from quantbot.core.instruments import InstrumentRegistry
from quantbot.data_engineering.connectors.historical import HistoricalDataFeed
from quantbot.data_engineering.connectors.live_feeds import PollingBarFeed
from quantbot.execution.gateways.simulated import SimulatedGateway
from quantbot.risk_management.limits import RiskLimits
from quantbot.simulation.backtest_engine import build_slippage

log = logging.getLogger(__name__)


class LiveRunner:
    def __init__(self, cfg: Dict[str, Any], registry: InstrumentRegistry, limits: RiskLimits, mode: str = "paper",
                 broker: str = "sim"):
        if mode not in ("paper", "live"):
            raise ValueError(mode)
        self.cfg, self.registry, self.limits, self.mode, self.broker = cfg, registry, limits, mode, broker
        self.clock = WallClock()
        self.bus = InMemoryEventBus()
        self.running = False
        venues = yaml.safe_load(Path(cfg["venues_file"]).read_text()) if Path(cfg["venues_file"]).exists() else {}
        gws_cfg = venues.get("gateways", {})
        self.gateways: List = []
        self.sim: Optional[SimulatedGateway] = None
        if broker == "sim":
            if mode == "live":
                raise ValueError("live mode requires a real broker (--broker alpaca|ibkr)")
            self.sim = SimulatedGateway(self.bus, registry, build_slippage(cfg),
                                        cfg["execution"].get("max_participation", 0.1))
            self.bus.subscribe(EventType.MARKET, self.sim.on_market)
            self.gateways.append(self.sim)
        elif broker == "alpaca":
            from quantbot.execution.gateways.alpaca import AlpacaGateway

            gc = dict(gws_cfg.get("ALPACA", {}))
            if mode == "live":
                gc["base_url"] = "https://api.alpaca.markets"
            elif "paper-api" not in gc.get("base_url", "https://paper-api.alpaca.markets"):
                raise ValueError("paper mode must use the Alpaca paper endpoint")
            self.gateways.append(AlpacaGateway(self.bus, **{k: v for k, v in gc.items() if k not in ("type", "paper")}))
        elif broker == "ibkr":
            from quantbot.execution.gateways.ibkr import IBKRGateway

            gc = dict(gws_cfg.get("IBKR", {}))
            if mode == "paper" and gc.get("port") in (7496, 4001):
                raise ValueError("paper mode must point at an IBKR paper port (7497/4002)")
            self.gateways.append(IBKRGateway(self.bus, registry, **{k: v for k, v in gc.items() if k != "type"}))
        else:
            raise ValueError(broker)

        self.metrics = None
        mon = cfg.get("monitoring", {})
        if mon.get("enabled"):
            from quantbot.monitoring import MetricsRegistry, MetricsServer

            self.metrics = MetricsRegistry()
            self.metrics_server = MetricsServer(self.metrics, mon.get("prometheus_port", 9108))
        if mon.get("redis_url"):
            from quantbot.core.bus import RedisStreamBridge

            self.bus.subscribe(None, RedisStreamBridge(mon["redis_url"]))

        self.system = TradingSystem(cfg, self.bus, registry, self.gateways, limits,
                                    routes=None, is_backtest=False, metrics=self.metrics)
        if self.sim is not None:
            pf = self.system.portfolio
            self.sim.position_lookup = lambda s: pf.positions[s].quantity if s in pf.positions else 0.0
        self.feed = PollingBarFeed(universe_symbols(cfg), self._fetcher())
        self.state_dir = Path(cfg["live"].get("state_dir", f"state/{cfg['system']['name']}_{mode}_{broker}"))
        self._n_trades_logged = 0

    # ----------------------------------------------------------------- data
    def _fetcher(self):
        if self.broker == "alpaca":
            gw = self.gateways[0]
            return lambda s: gw.latest_bars(s)
        if self.broker == "ibkr":
            gw = self.gateways[0]
            return lambda s: gw.latest_bars(s)
        from quantbot.data_engineering.connectors.yahoo import fetch_daily

        start = str((pd.Timestamp.now() - pd.DateOffset(years=2)).date())
        return lambda s: fetch_daily(s, start)

    # -------------------------------------------------------------- startup
    def start(self) -> None:
        for gw in self.gateways:
            gw.connect()
        if self.metrics is not None:
            self.metrics_server.start()
        frames = self.feed.warmup_frames()
        n = self.system.warmup(iter(HistoricalDataFeed(frames)))
        for s, f in frames.items():
            if len(f):
                self.feed._last_ts[s] = f.index[-1]
        log.info("warm-up replayed %d bars across %d symbols", n, len(frames))
        if self.broker != "sim":
            self._sync_from_broker()
        else:
            self._restore_state()
        # Decide on the latest completed bar now instead of waiting a whole bar: the warm-up
        # replay consumed it with order generation off. Orders fill on the next bar.
        sysm = self.system
        if sysm.last_data_ts is not None and sysm.breaker.trading_allowed and sysm.bar_count >= sysm.warmup_bars:
            sysm.rebalance(sysm.last_data_ts)
            self.bus.drain()
            log.info("start-up rebalance on %s close: %d open orders", sysm.last_data_ts.date(),
                     len(sysm.oms.open_orders()))
            if self.broker == "sim":
                self.save_state()

    # ------------------------------------------------------ persistence (sim paper book)
    def _state_file(self) -> Path:
        return self.state_dir / "portfolio.json"

    def save_state(self) -> None:
        """Persist the simulated paper book so it survives restarts. Broker-backed
        modes don't need this: the broker is the source of truth."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        pf = self.system.portfolio
        doc = {"saved_at": str(self.clock.now()), "cash": pf.cash, "high_water_mark": pf.high_water_mark,
               "initial_cash": pf.initial_cash, "breaker": self.system.breaker.state.value,
               "positions": {s: {"quantity": p.quantity, "avg_price": p.avg_price, "realized_pnl": p.realized_pnl,
                                 "last_settle": None if p.last_settle != p.last_settle else p.last_settle}
                             for s, p in pf.positions.items() if not p.is_flat},
               "last_bar": str(self.system.last_data_ts),
               "target_weights": {k: v for k, v in (self.system.targets_log[-1] if self.system.targets_log else {}).items()
                                  if k != "ts"},
               "signals": self.system.aggregator.combined(),
               "open_orders": [{"order_id": o.order_id, "symbol": o.symbol, "quantity": o.quantity,
                                "filled": o.filled_quantity, "status": o.status.value, "tag": o.tag,
                                "decision_price": o.decision_price} for o in self.system.oms.open_orders()]}
        tmp = self._state_file().with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, indent=2))
        tmp.replace(self._state_file())
        new_trades = pf.trades[self._n_trades_logged:]
        if new_trades:
            f = self.state_dir / "trades.csv"
            pd.DataFrame([t.__dict__ for t in new_trades]).to_csv(f, mode="a", header=not f.exists(), index=False)
            self._n_trades_logged = len(pf.trades)
        f = self.state_dir / "equity.csv"
        row = pd.DataFrame([{"ts": self.system.last_data_ts, "equity": pf.equity, "cash": pf.cash,
                             "gross_exposure": pf.gross_exposure, "breaker": self.system.breaker.state.value}])
        row.to_csv(f, mode="a", header=not f.exists(), index=False)

    def _restore_state(self) -> None:
        f = self._state_file()
        if not f.exists():
            log.info("no saved paper book at %s - starting fresh with %.2f", f, self.system.portfolio.cash)
            return
        doc = json.loads(f.read_text())
        pf = self.system.portfolio
        pf.cash = float(doc["cash"])
        for sym, p in doc["positions"].items():
            pos = pf.position(sym)
            pos.quantity, pos.avg_price, pos.realized_pnl = p["quantity"], p["avg_price"], p["realized_pnl"]
            pos.last_price = self.system.trade_price(sym)
            pos.last_settle = p["last_settle"] if p["last_settle"] is not None else pos.last_price
        pf.high_water_mark = float(doc["high_water_mark"])
        pf.mark({s: self.system.trade_price(s) for s in pf.positions}, self.clock.now(), settle=False)
        self.system.monitor.reset_peak(pf.high_water_mark)
        log.info("restored paper book from %s: equity=%.2f positions=%s", f, pf.equity, pf.quantities())

    def _sync_from_broker(self) -> None:
        gw = self.gateways[0]
        acct = gw.account()
        pf = self.system.portfolio
        pf.cash = float(acct.get("cash", pf.cash))
        for sym, qty in gw.positions().items():
            pos = pf.position(sym)
            pos.quantity = qty
            pos.last_price = self.system.trade_price(sym)
        pf.high_water_mark = pf.equity
        self.system.monitor.reset_peak(pf.equity)
        log.info("synced from broker: equity=%.2f positions=%s", pf.equity, pf.quantities())

    # ----------------------------------------------------------------- loop
    def run(self, max_iterations: Optional[int] = None) -> None:
        if self.mode == "live" and os.environ.get("QUANTBOT_LIVE_CONFIRM") != "YES":
            raise RuntimeError("Refusing to trade real money: set QUANTBOT_LIVE_CONFIRM=YES after paper validation.")
        self.start()
        self.running = True
        signal.signal(signal.SIGINT, lambda *_: self.stop())
        signal.signal(signal.SIGTERM, lambda *_: self.stop())
        poll = float(self.cfg["live"]["poll_seconds"])
        reconcile_every = int(self.cfg["live"].get("reconcile_every_polls", 10))
        it = 0
        while self.running:
            t0 = time.time()
            try:
                ev = self.feed.poll()
                if ev is not None:
                    self.bus.publish(ev)
                for gw in self.gateways:
                    if hasattr(gw, "poll"):
                        gw.poll()
                    if not getattr(gw, "connected", True):
                        self.system.breaker.on_disconnect(self.clock.now(), gw.name)
                self.bus.drain()
                self.system.breaker.check_staleness(self.clock.now(), self.feed.last_data_wallclock)
                if self.broker != "sim" and it % reconcile_every == 0:
                    self.system.oms.reconcile(self.gateways[0].positions())
                if self.metrics is not None:
                    self.metrics.decision_latency.observe(time.time() - t0)
                    if self.feed.last_data_wallclock is not None:
                        self.metrics.last_bar_age.set((self.clock.now() - self.feed.last_data_wallclock).total_seconds())
                if ev is not None:
                    if self.broker == "sim":
                        self.save_state()
                    pf = self.system.portfolio
                    log.info("%s equity=%.2f gross=%.2fx positions=%d breaker=%s", ev.ts, pf.equity,
                             pf.gross_exposure / max(pf.equity, 1e-9), len(pf.quantities()),
                             self.system.breaker.state.value)
            except Exception:
                log.exception("live loop error - pausing new risk")
                self.system.breaker.pause(self.clock.now(), "unhandled exception in live loop")
            it += 1
            if max_iterations is not None and it >= max_iterations:
                break
            deadline = t0 + poll
            while self.running and time.time() < deadline:  # wake promptly on SIGINT/SIGTERM
                self.clock.sleep(min(1.0, deadline - time.time()))
        self.shutdown()

    def stop(self) -> None:
        log.warning("shutdown requested")
        self.running = False

    def shutdown(self) -> None:
        self.system.ems.cancel_all()
        if self.cfg["live"].get("flatten_on_shutdown"):
            self.system.flatten_all("shutdown")
        self.bus.drain()
        if self.broker == "sim":
            self.save_state()
        log.info("stopped. final equity=%.2f", self.system.portfolio.equity)
