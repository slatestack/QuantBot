"""Dependency-free Prometheus metrics exporter (text exposition format 0.0.4).

Exposes engine telemetry (equity, PnL, drawdown, VaR, exposure, fills,
slippage, rejections, decision latency, breaker state) on ``/metrics``.
"""
from __future__ import annotations

import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


class _Metric:
    def __init__(self, name: str, help_: str, mtype: str):
        self.name, self.help, self.type = name, help_, mtype
        self._values: Dict[Tuple[Tuple[str, str], ...], float] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(labels: Optional[Dict[str, str]]):
        return tuple(sorted((labels or {}).items()))

    def render(self) -> List[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} {self.type}"]
        with self._lock:
            for k, v in self._values.items():
                lab = ",".join(f'{a}="{b}"' for a, b in k)
                lines.append(f"{self.name}{{{lab}}} {v}" if lab else f"{self.name} {v}")
        return lines


class Gauge(_Metric):
    def __init__(self, name, help_):
        super().__init__(name, help_, "gauge")

    def set(self, v: float, labels=None):
        with self._lock:
            self._values[self._key(labels)] = float(v)


class Counter(_Metric):
    def __init__(self, name, help_):
        super().__init__(name, help_, "counter")

    def inc(self, v: float = 1.0, labels=None):
        with self._lock:
            k = self._key(labels)
            self._values[k] = self._values.get(k, 0.0) + float(v)


class Histogram(_Metric):
    def __init__(self, name, help_, buckets=(0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0)):
        super().__init__(name, help_, "histogram")
        self.buckets = buckets
        self._counts = [0] * (len(buckets) + 1)
        self._sum = 0.0
        self._n = 0

    def observe(self, v: float):
        with self._lock:
            self._sum += v
            self._n += 1
            for i, b in enumerate(self.buckets):
                if v <= b:
                    self._counts[i] += 1
            self._counts[-1] += 1

    def render(self) -> List[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        with self._lock:
            for b, c in zip(self.buckets, self._counts):
                lines.append(f'{self.name}_bucket{{le="{b}"}} {c}')
            lines.append(f'{self.name}_bucket{{le="+Inf"}} {self._counts[-1]}')
            lines.append(f"{self.name}_sum {self._sum}")
            lines.append(f"{self.name}_count {self._n}")
        return lines


class MetricsRegistry:
    def __init__(self, prefix: str = "quantbot"):
        p = prefix
        self.equity = Gauge(f"{p}_equity", "Portfolio equity (NAV)")
        self.cash = Gauge(f"{p}_cash", "Cash balance")
        self.drawdown = Gauge(f"{p}_drawdown", "Drawdown from high-water mark (fraction)")
        self.daily_pnl = Gauge(f"{p}_daily_pnl_pct", "PnL since start of day (fraction)")
        self.gross = Gauge(f"{p}_gross_leverage", "Gross exposure / equity")
        self.net = Gauge(f"{p}_net_leverage", "Net exposure / equity")
        self.var = Gauge(f"{p}_var_99", "1-day 99% VaR (fraction of equity)")
        self.cvar = Gauge(f"{p}_cvar_99", "1-day 99% CVaR (fraction of equity)")
        self.position = Gauge(f"{p}_position_weight", "Position weight by symbol")
        self.signal = Gauge(f"{p}_signal", "Combined forecast by symbol")
        self.breaker = Gauge(f"{p}_breaker_state", "0=NORMAL 1=PAUSED 2=HALTED")
        self.fills = Counter(f"{p}_fills_total", "Fills by symbol")
        self.slippage = Counter(f"{p}_slippage_cost_total", "Slippage cost (currency)")
        self.commission = Counter(f"{p}_commission_total", "Commission (currency)")
        self.rejections = Gauge(f"{p}_order_rejections", "Pre-trade rejections by reason")
        self.open_orders = Gauge(f"{p}_open_orders", "Open orders")
        self.decision_latency = Histogram(f"{p}_decision_latency_seconds", "Decision step latency")
        self.last_bar_age = Gauge(f"{p}_last_bar_age_seconds", "Seconds since last market data")
        self._all = [v for v in self.__dict__.values() if isinstance(v, _Metric)]

    def render(self) -> str:
        out: List[str] = []
        for m in self._all:
            out += m.render()
        return "\n".join(out) + "\n"

    # ---- engine hooks
    def on_fill(self, ev) -> None:
        self.fills.inc(1, {"symbol": ev.symbol})
        self.slippage.inc(ev.slippage_cost)
        self.commission.inc(ev.commission)

    def on_decision(self, system) -> None:
        pf = system.portfolio
        eq = pf.equity
        self.equity.set(eq)
        self.cash.set(pf.cash)
        self.drawdown.set(pf.drawdown)
        if eq > 0:
            self.gross.set(pf.gross_exposure / eq)
            self.net.set(pf.net_exposure / eq)
        snap = system.monitor.snapshot
        self.daily_pnl.set(snap.get("daily_pnl_pct", 0.0))
        v = snap.get("param_var")
        if v == v and v is not None:
            self.var.set(v)
        c = snap.get("param_cvar")
        if c == c and c is not None:
            self.cvar.set(c)
        for s, w in pf.weights().items():
            self.position.set(w, {"symbol": s})
        for s, f in system.aggregator.combined().items():
            self.signal.set(f, {"symbol": s})
        self.breaker.set({"NORMAL": 0, "PAUSED": 1, "HALTED": 2}[system.breaker.state.value])
        for r, n in system.pre_trade.rejections.items():
            self.rejections.set(n, {"reason": r})
        self.open_orders.set(len(system.oms.open_orders()))


class MetricsServer:
    def __init__(self, registry: MetricsRegistry, port: int = 9108, host: str = "0.0.0.0"):
        self.registry = registry
        reg = registry

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                if self.path.startswith("/metrics"):
                    body = reg.render().encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; version=0.0.4")
                elif self.path.startswith("/health"):
                    body = b"ok\n"
                    self.send_response(200)
                else:
                    body = b"not found\n"
                    self.send_response(404)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        self._srv = ThreadingHTTPServer((host, port), Handler)
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)

    def start(self) -> None:
        self._thread.start()
        log.info("Prometheus metrics on :%d/metrics", self._srv.server_address[1])

    def stop(self) -> None:
        self._srv.shutdown()
