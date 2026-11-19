"""Alpaca Markets REST gateway (US equities/ETFs) - paper or live.

Credentials are read from environment variables (``APCA_API_KEY_ID`` /
``APCA_API_SECRET_KEY`` by default); nothing is stored in config.

Fills are discovered by polling each open order and diffing ``filled_qty`` /
``filled_avg_price`` (robust to missed websocket messages). Every fill and
status change is pushed onto the event bus with ``put_threadsafe``.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

import pandas as pd

from quantbot.core.events import FillEvent, Order, OrderStatus, OrderType, OrderUpdateEvent

log = logging.getLogger(__name__)

_STATUS = {
    "new": OrderStatus.ACKNOWLEDGED, "accepted": OrderStatus.ACKNOWLEDGED, "pending_new": OrderStatus.PENDING_NEW,
    "accepted_for_bidding": OrderStatus.ACKNOWLEDGED, "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "filled": OrderStatus.FILLED, "canceled": OrderStatus.CANCELED, "expired": OrderStatus.CANCELED,
    "done_for_day": OrderStatus.CANCELED, "replaced": OrderStatus.CANCELED, "rejected": OrderStatus.REJECTED,
    "pending_cancel": OrderStatus.PENDING_CANCEL, "stopped": OrderStatus.ACKNOWLEDGED,
    "suspended": OrderStatus.ACKNOWLEDGED, "calculated": OrderStatus.ACKNOWLEDGED,
}


class AlpacaGateway:
    def __init__(self, bus, base_url: str = "https://paper-api.alpaca.markets",
                 data_url: str = "https://data.alpaca.markets", key_env: str = "APCA_API_KEY_ID",
                 secret_env: str = "APCA_API_SECRET_KEY", name: str = "ALPACA", timeout: float = 10.0, **_):
        self.bus = bus
        self.base_url = base_url.rstrip("/")
        self.data_url = data_url.rstrip("/")
        self.key = os.environ.get(key_env, "")
        self.secret = os.environ.get(secret_env, "")
        self.name = name
        self.timeout = timeout
        self.connected = False
        self._orders: Dict[str, Order] = {}
        self._seen_fill: Dict[str, float] = {}
        self._seen_status: Dict[str, str] = {}

    # ------------------------------------------------------------------ http
    def _req(self, method: str, url: str, body: Optional[dict] = None, params: Optional[dict] = None) -> Any:
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method, headers={
            "APCA-API-KEY-ID": self.key, "APCA-API-SECRET-KEY": self.secret, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw = r.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="ignore")
            raise RuntimeError(f"Alpaca {method} {url} -> {e.code}: {msg}") from None

    # ------------------------------------------------------------ gateway API
    def connect(self) -> None:
        if not self.key or not self.secret:
            raise RuntimeError("Alpaca credentials missing: set APCA_API_KEY_ID and APCA_API_SECRET_KEY")
        acct = self.account()
        if acct.get("trading_blocked"):
            raise RuntimeError("Alpaca account is trading-blocked")
        self.connected = True
        log.info("Connected to Alpaca (%s), equity=%s", self.base_url, acct.get("equity"))

    def account(self) -> dict:
        return self._req("GET", f"{self.base_url}/v2/account")

    def positions(self) -> Dict[str, float]:
        return {p["symbol"]: float(p["qty"]) for p in self._req("GET", f"{self.base_url}/v2/positions")}

    def send(self, order: Order) -> None:
        body = {"symbol": order.symbol, "qty": str(abs(order.quantity)), "side": "buy" if order.quantity > 0 else "sell",
                "type": "market" if order.order_type is OrderType.MARKET else "limit",
                "time_in_force": order.tif.value.lower(), "client_order_id": order.order_id}
        if order.order_type is OrderType.LIMIT:
            body["limit_price"] = str(order.limit_price)
        resp = self._req("POST", f"{self.base_url}/v2/orders", body)
        self._orders[order.order_id] = order
        self.bus.put_threadsafe(OrderUpdateEvent(pd.Timestamp.now(tz="UTC"), order.order_id,
                                                 _STATUS.get(resp.get("status"), OrderStatus.ACKNOWLEDGED),
                                                 broker_order_id=resp.get("id")))

    def cancel(self, order: Order) -> None:
        if order.broker_order_id:
            self._req("DELETE", f"{self.base_url}/v2/orders/{order.broker_order_id}")

    def cancel_all(self) -> None:
        self._req("DELETE", f"{self.base_url}/v2/orders")

    def poll(self) -> None:
        """Diff open orders' fill state and emit Fill/OrderUpdate events."""
        for oid, order in list(self._orders.items()):
            try:
                o = self._req("GET", f"{self.base_url}/v2/orders:by_client_order_id",
                              params={"client_order_id": oid})
            except Exception as exc:
                log.warning("poll %s failed: %s", oid, exc)
                continue
            now = pd.Timestamp.now(tz="UTC")
            filled = float(o.get("filled_qty") or 0)
            avg = float(o.get("filled_avg_price") or 0)
            prev = self._seen_fill.get(oid, 0.0)
            if filled > prev + 1e-9:
                prev_notional = prev * self._seen_fill.get(oid + ":avg", 0.0)
                inc = filled - prev
                px = (filled * avg - prev_notional) / inc if inc else avg
                sign = 1 if order.quantity > 0 else -1
                self.bus.put_threadsafe(FillEvent(now, oid, order.symbol, sign * inc, px, 0.0, 0.0, self.name))
                self._seen_fill[oid] = filled
                self._seen_fill[oid + ":avg"] = avg
            st = o.get("status")
            if st != self._seen_status.get(oid):
                self._seen_status[oid] = st
                mapped = _STATUS.get(st)
                if mapped in (OrderStatus.CANCELED, OrderStatus.REJECTED):
                    self.bus.put_threadsafe(OrderUpdateEvent(now, oid, mapped, reason=st))
            if st in ("filled", "canceled", "expired", "rejected", "done_for_day", "replaced"):
                self._orders.pop(oid, None)

    # --------------------------------------------------------------- market data
    def latest_bars(self, symbol: str, timeframe: str = "1Day", limit: int = 400) -> pd.DataFrame:
        start = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=int(limit * 1.6))).isoformat()
        doc = self._req("GET", f"{self.data_url}/v2/stocks/{symbol}/bars",
                        params={"timeframe": timeframe, "limit": limit, "adjustment": "all", "start": start,
                                "feed": "iex"})
        rows = doc.get("bars") or []
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        idx = pd.to_datetime(df["t"], utc=True)
        if timeframe == "1Day":
            idx = idx.normalize() + pd.Timedelta(hours=21)  # stamp at session close
        return pd.DataFrame({"open": df["o"].values, "high": df["h"].values, "low": df["l"].values,
                             "close": df["c"].values, "volume": df["v"].values}, index=idx).astype(float)
