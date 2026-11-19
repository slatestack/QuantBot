import numpy as np
import pandas as pd
import pytest

from quantbot.core.events import FillEvent, Order, OrderStatus, OrderUpdateEvent
from quantbot.core.instruments import AssetClass, Instrument, InstrumentRegistry
from quantbot.core.state.portfolio import Portfolio
from quantbot.execution.oms import InvalidTransition, OrderManager
from quantbot.risk_management import PreTradeRisk, RiskLimits

T = pd.Timestamp("2024-01-02 21:00", tz="UTC")


def test_equity_accounting_round_trip():
    reg = InstrumentRegistry([Instrument("A")])
    pf = Portfolio(10_000, reg)
    pf.on_fill(FillEvent(T, "o1", "A", 10, 100.0, commission=1.0))
    pf.mark({"A": 110.0}, T)
    assert pf.equity == pytest.approx(10_000 - 1 + 100)
    pf.on_fill(FillEvent(T, "o2", "A", -10, 120.0, commission=1.0))
    assert pf.positions["A"].is_flat
    assert pf.cash == pytest.approx(10_000 + 200 - 2)
    assert pf.positions["A"].realized_pnl == pytest.approx(200)


def test_futures_variation_margin_matches_contract_pnl():
    reg = InstrumentRegistry([Instrument("CL", AssetClass.FUTURE, multiplier=100, initial_margin=1000)])
    pf = Portfolio(100_000, reg)
    pf.on_fill(FillEvent(T, "o1", "CL", 2, 70.0))
    assert pf.cash == pytest.approx(100_000)  # no notional exchanged
    pf.mark({"CL": 71.5}, T)
    assert pf.cash == pytest.approx(100_000 + 2 * 1.5 * 100)
    pf.mark({"CL": 69.0}, T + pd.Timedelta(days=1))
    assert pf.equity == pytest.approx(100_000 - 2 * 1.0 * 100)
    pf.on_fill(FillEvent(T, "o2", "CL", -2, 69.5))
    assert pf.equity == pytest.approx(100_000 - 2 * 0.5 * 100)


def test_short_borrow_accrues():
    reg = InstrumentRegistry([Instrument("A", short_borrow_rate=0.0365)])
    pf = Portfolio(10_000, reg)
    pf.on_fill(FillEvent(T, "o", "A", -100, 100.0))
    pf.mark({"A": 100.0}, T)
    pf.mark({"A": 100.0}, T + pd.Timedelta(days=10))
    assert pf.total_borrow_cost == pytest.approx(10_000 * 0.0365 * 10 / 365)


class _GW:
    name = "G"
    connected = True

    def __init__(self):
        self.sent, self.cancelled = [], []

    def send(self, o):
        self.sent.append(o)

    def cancel(self, o):
        self.cancelled.append(o)


class _Router:
    def __init__(self, gw):
        self.gw = gw

    def route(self, o):
        return self.gw

    def gateway_by_name(self, n):
        return self.gw


def test_oms_state_machine_and_partial_fills():
    reg = InstrumentRegistry([Instrument("A")])
    pf = Portfolio(1_000_000, reg)
    gw = _GW()
    oms = OrderManager(_Router(gw), PreTradeRisk(RiskLimits(), reg), pf)
    o = Order("A", 100)
    assert oms.submit(o, T, 50.0)
    oms.on_order_update(OrderUpdateEvent(T, o.order_id, OrderStatus.ACKNOWLEDGED))
    oms.on_fill(FillEvent(T, o.order_id, "A", 40, 50.0))
    assert o.status is OrderStatus.PARTIALLY_FILLED and o.remaining == 60
    oms.on_fill(FillEvent(T, o.order_id, "A", 60, 51.0))
    assert o.status is OrderStatus.FILLED and o.avg_fill_price == pytest.approx(50.6)
    with pytest.raises(InvalidTransition):
        oms._transition(o, OrderStatus.ACKNOWLEDGED)
    assert oms.pending_qty() == {}


def test_oms_rejects_and_reconciles():
    reg = InstrumentRegistry([Instrument("A")])
    pf = Portfolio(1_000_000, reg)
    oms = OrderManager(_Router(_GW()), PreTradeRisk(RiskLimits(max_order_qty=10), reg), pf)
    o = Order("A", 100)
    assert not oms.submit(o, T, 50.0)
    assert o.status is OrderStatus.REJECTED and "max_order_qty" in o.reject_reason
    pf.on_fill(FillEvent(T, "x", "A", 5, 50.0))
    assert oms.reconcile({"A": 5}) == {}
    assert oms.reconcile({"A": 4, "B": 1}) == {"A": (5, 4), "B": (0.0, 1)}


def test_instrument_rounding_never_increases_exposure():
    ins = Instrument("X", lot_size=1)
    assert ins.round_qty(2.9) == 2 and ins.round_qty(-2.9) == -2
