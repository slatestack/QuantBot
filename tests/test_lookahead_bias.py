"""Strict temporal-integrity tests: nothing the system decides at time T may
depend on data stamped after T."""
import numpy as np
import pandas as pd

from quantbot.core.bus import InMemoryEventBus
from quantbot.core.events import EventType, FillEvent, MarketEvent, Order
from quantbot.core.state.history import HistoryBuffer
from quantbot.data_engineering.normalizers.schemas import Bar
from quantbot.execution.gateways.simulated import SimulatedGateway
from quantbot.quant_models.alpha.ml_forecaster import MLForecaster
from quantbot.risk_management.limits import RiskLimits
from quantbot.simulation.backtest_engine import BacktestEngine


def _run(cfg, frames, registry):
    eng = BacktestEngine(cfg, frames, registry, RiskLimits(backtest_auto_reset_bars=5))
    res = eng.run()
    return res, eng


def test_future_perturbation_does_not_change_past_decisions(universe, registry, small_cfg):
    """Scramble every bar after T; all targets and equity up to T must be identical."""
    frames = {s: f for s, f in universe.bars.items() if s in small_cfg["universe"]["equities"] + small_cfg["universe"]["futures"]}
    cut = pd.Timestamp("2018-03-01", tz="UTC")
    rng = np.random.default_rng(0)
    perturbed = {}
    for s, f in frames.items():
        g = f.copy()
        after = g.index > cut
        shock = np.exp(rng.normal(0, 0.2, after.sum()))
        for c in ("open", "high", "low", "close"):
            g.loc[after, c] = g.loc[after, c] * shock
        g.loc[after, "volume"] = g.loc[after, "volume"] * rng.uniform(0.1, 3, after.sum())
        for c in ("f1", "f2"):
            if c in g:
                g.loc[after, c] = g.loc[after, c] * shock
        perturbed[s] = g
    r1, _ = _run(small_cfg, frames, registry)
    r2, _ = _run(small_cfg, perturbed, registry)
    e1 = r1.equity["equity"][r1.equity.index <= cut]
    e2 = r2.equity["equity"][r2.equity.index <= cut]
    pd.testing.assert_series_equal(e1, e2)
    t1 = r1.targets[r1.targets.index <= cut]
    t2 = r2.targets[r2.targets.index <= cut]
    assert len(t1) > 50
    pd.testing.assert_frame_equal(t1, t2)
    # and the futures do diverge, proving the perturbation was real
    assert not np.allclose(r1.equity["equity"].iloc[-1], r2.equity["equity"].iloc[-1])


def test_orders_never_fill_on_the_decision_bar(registry):
    bus = InMemoryEventBus()
    gw = SimulatedGateway(bus, registry)
    fills = []
    bus.subscribe(EventType.FILL, fills.append)
    t0 = pd.Timestamp("2020-01-02 21:00", tz="UTC")
    b0 = Bar("SPY", t0, 100, 101, 99, 100, 1e7)
    o = Order("SPY", 10)
    o.created_ts = t0
    gw.send(o)
    gw.on_market(MarketEvent(t0, {"SPY": b0}))  # same-timestamp bar: must not fill
    bus.drain()
    assert fills == []
    t1 = t0 + pd.Timedelta(days=1)
    gw.on_market(MarketEvent(t1, {"SPY": Bar("SPY", t1, 105, 106, 104, 105.5, 1e7)}))
    bus.drain()
    assert len(fills) == 1 and fills[0].ts == t1
    assert fills[0].price >= 105.0  # filled at next open (+ costs), not at the decision close


def test_history_buffer_rejects_out_of_order_bars():
    h = HistoryBuffer(10)
    t = pd.Timestamp("2020-01-02", tz="UTC")
    h.update({"A": Bar("A", t, 1, 1, 1, 1, 1)})
    try:
        h.update({"A": Bar("A", t - pd.Timedelta(days=1), 1, 1, 1, 1, 1)})
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_ml_labels_are_purged_until_horizon_elapses():
    m = MLForecaster(["A"], horizon=5, retrain_every=10_000, min_samples=10**9)
    h = HistoryBuffer(1000)
    rng = np.random.default_rng(1)
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400)))
    ts0 = pd.Timestamp("2020-01-01", tz="UTC")
    for i, p in enumerate(px):
        b = Bar("A", ts0 + pd.Timedelta(days=i), p, p, p, p, 1e6)
        h.update({"A": b})
        m.on_bar(b.ts_event, {"A": b}, h)
        # every realized training row must be at least `horizon` bars old
        n_feature_rows = max(0, (i + 1) - 253)
        assert len(m._y) <= max(0, n_feature_rows - 5 + 1)
