"""Interactive Brokers gateway (equities via SMART routing, futures via CME
group exchanges) built on ``ib_async`` (the maintained successor of
``ib_insync``). Install with ``pip install ib_async`` and run TWS or IB Gateway
with the API enabled.

Futures: strategies trade the *root* (e.g. ``CL``); this gateway routes to the
dated contract chosen by the ``RollManager`` (front contract until N business
days before first notice), and the root's broker symbol (e.g. ``MCL``) comes
from the instrument's ``root`` field.

NOTE: this adapter has not been exercised against a live TWS session in CI;
validate it in IBKR paper trading (port 7497) before any real-money use.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import pandas as pd

from quantbot.core.events import FillEvent, Order, OrderStatus, OrderType, OrderUpdateEvent
from quantbot.core.instruments import InstrumentRegistry

log = logging.getLogger(__name__)

_STATUS = {
    "PendingSubmit": OrderStatus.PENDING_NEW, "ApiPending": OrderStatus.PENDING_NEW,
    "PreSubmitted": OrderStatus.ACKNOWLEDGED, "Submitted": OrderStatus.ACKNOWLEDGED,
    "PendingCancel": OrderStatus.PENDING_CANCEL, "ApiCancelled": OrderStatus.CANCELED,
    "Cancelled": OrderStatus.CANCELED, "Filled": OrderStatus.FILLED, "Inactive": OrderStatus.REJECTED,
}


class IBKRGateway:
    def __init__(self, bus, registry: InstrumentRegistry, host: str = "127.0.0.1", port: int = 7497,
                 client_id: int = 17, name: str = "IBKR", roll_manager=None, **_):
        self.bus = bus
        self.registry = registry
        self.host, self.port, self.client_id = host, port, client_id
        self.name = name
        self.roll_manager = roll_manager
        self.connected = False
        self.ib = None
        self._trades: Dict[str, object] = {}

    def connect(self) -> None:
        try:
            from ib_async import IB
        except ImportError:  # pragma: no cover
            try:
                from ib_insync import IB  # type: ignore
            except ImportError as exc:
                raise RuntimeError("IBKR gateway requires `pip install ib_async`") from exc
        self.ib = IB()
        self.ib.connect(self.host, self.port, clientId=self.client_id, readonly=False)
        self.ib.disconnectedEvent += self._on_disconnect
        self.connected = True
        log.info("Connected to IBKR %s:%s", self.host, self.port)

    def _on_disconnect(self, *_):
        self.connected = False
        log.error("IBKR disconnected")

    def _contract(self, symbol: str):
        from ib_async import Future, Stock

        ins = self.registry.get(symbol)
        if not ins.is_future:
            return Stock(symbol, "SMART", ins.currency)
        if self.roll_manager is None:
            raise RuntimeError("futures routing needs a RollManager")
        local = self.roll_manager.active(symbol, pd.Timestamp.now(tz="UTC"))
        # local like 'CLZ26' -> YYYYMM
        month = "FGHJKMNQUVXZ".index(local[-3]) + 1
        year = 2000 + int(local[-2:])
        return Future(ins.root or symbol, f"{year}{month:02d}", ins.exchange, currency=ins.currency)

    def send(self, order: Order) -> None:
        from ib_async import LimitOrder, MarketOrder

        contract = self._contract(order.symbol)
        self.ib.qualifyContracts(contract)
        action = "BUY" if order.quantity > 0 else "SELL"
        ib_order = (MarketOrder(action, abs(order.quantity)) if order.order_type is OrderType.MARKET
                    else LimitOrder(action, abs(order.quantity), order.limit_price))
        ib_order.orderRef = order.order_id
        trade = self.ib.placeOrder(contract, ib_order)
        self._trades[order.order_id] = trade
        oid = order.order_id

        def on_fill(trade_, fill):
            ex = fill.execution
            sign = 1 if ex.side == "BOT" else -1
            comm = fill.commissionReport.commission if fill.commissionReport else 0.0
            self.bus.put_threadsafe(FillEvent(pd.Timestamp(ex.time).tz_convert("UTC"), oid, order.symbol,
                                              sign * float(ex.shares), float(ex.price), float(comm or 0.0), 0.0,
                                              self.name))

        def on_status(trade_):
            st = _STATUS.get(trade_.orderStatus.status)
            if st in (OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.ACKNOWLEDGED):
                self.bus.put_threadsafe(OrderUpdateEvent(pd.Timestamp.now(tz="UTC"), oid, st,
                                                         broker_order_id=str(trade_.order.orderId)))

        trade.fillEvent += on_fill
        trade.statusEvent += on_status

    def cancel(self, order: Order) -> None:
        t = self._trades.get(order.order_id)
        if t is not None:
            self.ib.cancelOrder(t.order)

    def positions(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for p in self.ib.positions():
            sym = p.contract.symbol
            # map broker futures root back to our root symbol
            for ins in self.registry:
                if ins.is_future and ins.root == sym:
                    sym = ins.symbol
            out[sym] = out.get(sym, 0.0) + float(p.position)
        return out

    def account(self) -> dict:
        vals = {v.tag: v.value for v in self.ib.accountSummary()}
        return {"equity": float(vals.get("NetLiquidation", 0)), "cash": float(vals.get("TotalCashValue", 0))}

    def poll(self) -> None:
        if self.ib is not None:
            self.ib.sleep(0)  # pump the ib event loop

    def latest_bars(self, symbol: str, duration: str = "2 Y", bar_size: str = "1 day") -> Optional[pd.DataFrame]:
        contract = self._contract(symbol)
        bars = self.ib.reqHistoricalData(contract, "", duration, bar_size, "TRADES", useRTH=True)
        if not bars:
            return None
        df = pd.DataFrame([{"ts": pd.Timestamp(b.date), "open": b.open, "high": b.high, "low": b.low,
                            "close": b.close, "volume": b.volume} for b in bars]).set_index("ts")
        df.index = pd.to_datetime(df.index, utc=True) + pd.Timedelta(hours=21)
        return df
