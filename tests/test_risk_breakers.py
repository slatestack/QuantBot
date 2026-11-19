"""Chaos tests for the independent risk path: flash crashes, feed drops,
fat fingers, margin and the kill switch."""
import numpy as np
import pandas as pd

from quantbot.core.bus import InMemoryEventBus
from quantbot.core.events import EventType, MarketEvent, Order, OrderType
from quantbot.core.instruments import AssetClass, Instrument, InstrumentRegistry
from quantbot.core.state.portfolio import Portfolio
from quantbot.data_engineering.normalizers.schemas import Bar
from quantbot.risk_management import BreakerState, CircuitBreaker, PreTradeRisk, RealTimeRiskMonitor, RiskLimits
from quantbot.simulation.backtest_engine import BacktestEngine


def _portfolio_with_position(registry, qty=1000, px=100.0):
    pf = Portfolio(1_000_000, registry)
    from quantbot.core.events import FillEvent
    pf.on_fill(FillEvent(pd.Timestamp("2020-01-01", tz="UTC"), "x", "SPY", qty, px))
    pf.mark({"SPY": px}, pd.Timestamp("2020-01-01", tz="UTC"))
    return pf


def test_flash_crash_trips_breaker_flattens_and_locks(universe, registry, small_cfg):
    frames = {s: f.copy() for s, f in universe.bars.items()
              if s in small_cfg["universe"]["equities"] + small_cfg["universe"]["futures"]}
    crash_day = frames["SYN_EQ1"].index[600]
    for s, f in frames.items():  # -15% gap down across every asset
        after = f.index >= crash_day
        for c in ("open", "high", "low", "close"):
            f.loc[after, c] *= 0.85
        if "ret" in f:
            f.loc[crash_day, "ret"] = -0.15 if crash_day in f.index else f["ret"]
    lim = RiskLimits(backtest_auto_reset_bars=0)  # never auto-reset -> stays locked
    eng = BacktestEngine(small_cfg, frames, registry, lim)
    res = eng.run()
    assert res.breaker_trips, "breaker should trip on the crash"
    trip = res.breaker_trips[0]
    assert trip["ts"] == crash_day and trip["flatten"]
    # after flatten orders fill on the next bar, the book is flat and stays flat
    later = res.equity[res.equity.index > crash_day + pd.Timedelta(days=5)]
    assert (later["gross_exposure"] == 0).all()
    assert eng.system.breaker.state is BreakerState.HALTED
    flatten_orders = [o for o in eng.system.oms.orders.values() if o.tag.startswith("flatten")]
    assert flatten_orders and all(o.status.value != "REJECTED" for o in flatten_orders)


def test_locked_gate_rejects_new_risk_but_allows_reduction(registry):
    pf = _portfolio_with_position(registry)
    pt = PreTradeRisk(RiskLimits(), registry)
    pt.locked = True
    assert not pt.check(Order("SPY", 100), pf, 100.0).approved
    assert not pt.check(Order("QQQ", 10), pf, 100.0).approved
    assert pt.check(Order("SPY", -500), pf, 100.0).approved
    assert pt.check(Order("SPY", -1000), pf, 100.0).approved


def test_fat_finger_and_collar_and_concentration(registry):
    pf = _portfolio_with_position(registry, qty=0.0 + 1, px=100)
    pt = PreTradeRisk(RiskLimits(max_order_notional=1e6, price_collar_pct=0.05, max_position_pct_equity=0.2), registry)
    r = pt.check(Order("SPY", 20_000), pf, 100.0)
    assert not r.approved and r.reason.startswith("max_order_notional")
    r = pt.check(Order("SPY", 100, OrderType.LIMIT, limit_price=120.0), pf, 100.0)
    assert not r.approved and r.reason.startswith("price_collar")
    r = pt.check(Order("SPY", 3000), pf, 100.0)
    assert not r.approved and r.reason.startswith("concentration")
    assert pt.check(Order("SPY", 1000), pf, 100.0).approved


def test_futures_margin_check(registry):
    reg = InstrumentRegistry([Instrument("CL", AssetClass.FUTURE, multiplier=100, initial_margin=50_000,
                                         maintenance_margin=45_000)])
    pf = Portfolio(100_000, reg)
    pt = PreTradeRisk(RiskLimits(max_position_pct_equity=100, max_gross_leverage=100, max_net_leverage=100,
                                 max_order_pct_equity=100), reg)
    assert pt.check(Order("CL", 1), pf, 70.0).approved
    r = pt.check(Order("CL", 2), pf, 70.0)
    assert not r.approved and r.reason.startswith("margin")


def test_daily_loss_and_drawdown_monitor():
    mon = RealTimeRiskMonitor(RiskLimits(daily_loss_limit_pct=0.025, max_drawdown_pct=0.10))
    t = pd.Timestamp("2020-01-02 21:00", tz="UTC")
    assert mon.update(t, 1_000_000) == []
    ev = mon.update(t + pd.Timedelta(days=1), 970_000)
    assert [e.kind for e in ev] == ["daily_loss"]
    for k, eq in enumerate([960_000, 950_000, 940_000, 930_000, 899_000]):
        ev = mon.update(t + pd.Timedelta(days=2 + k), eq)
    assert "max_drawdown" in [e.kind for e in ev]


def test_feed_drop_pauses_then_resumes(registry):
    pt = PreTradeRisk(RiskLimits(stale_data_seconds=60), registry)
    calls = {"cancel": 0, "flatten": 0}
    cb = CircuitBreaker(pt.limits, pt, lambda: calls.__setitem__("cancel", calls["cancel"] + 1),
                        lambda r: calls.__setitem__("flatten", calls["flatten"] + 1))
    now = pd.Timestamp("2024-01-01 15:00", tz="UTC")
    cb.check_staleness(now, now - pd.Timedelta(seconds=300))
    assert cb.state is BreakerState.PAUSED and pt.locked and calls["flatten"] == 0
    cb.check_staleness(now, now - pd.Timedelta(seconds=5))
    assert cb.state is BreakerState.NORMAL and not pt.locked


def test_manual_reset_required_in_live(registry):
    pt = PreTradeRisk(RiskLimits(), registry)
    cb = CircuitBreaker(pt.limits, pt, lambda: None, lambda r: None, is_backtest=False)
    cb.trip(pd.Timestamp("2024-01-01", tz="UTC"), "test")
    for _ in range(100):
        cb.on_bar()
    assert cb.state is BreakerState.HALTED
    cb.reset("operator")
    assert cb.state is BreakerState.NORMAL and not pt.locked
