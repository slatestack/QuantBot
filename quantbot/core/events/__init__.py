"""Event definitions flowing through the message bus.

Every state change in the system is represented as an immutable-ish event so the
exact same handlers run in backtest, paper and live mode.
"""
from __future__ import annotations

import enum
import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd

from quantbot.data_engineering.normalizers.schemas import Bar, OrderBookSnapshot, Tick

_order_ids = itertools.count(1)


def next_order_id(prefix: str = "O") -> str:
    return f"{prefix}{next(_order_ids):08d}"


class EventType(enum.Enum):
    MARKET = "MARKET"
    TICK = "TICK"
    BOOK = "BOOK"
    DECISION = "DECISION"
    SIGNAL = "SIGNAL"
    TARGET = "TARGET"
    ORDER = "ORDER"
    ORDER_UPDATE = "ORDER_UPDATE"
    FILL = "FILL"
    RISK = "RISK"
    TIMER = "TIMER"


class Side(enum.IntEnum):
    BUY = 1
    SELL = -1


class OrderType(enum.Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class TimeInForce(enum.Enum):
    DAY = "DAY"
    GTC = "GTC"
    IOC = "IOC"


class OrderStatus(enum.Enum):
    PENDING_NEW = "PENDING_NEW"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    PENDING_CANCEL = "PENDING_CANCEL"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"

    @property
    def is_terminal(self) -> bool:
        return self in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED)


@dataclass
class Order:
    symbol: str
    quantity: float  # signed: + buy, - sell
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    tif: TimeInForce = TimeInForce.DAY
    order_id: str = field(default_factory=next_order_id)
    parent_id: Optional[str] = None
    tag: str = ""
    created_ts: Optional[pd.Timestamp] = None
    status: OrderStatus = OrderStatus.PENDING_NEW
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    broker_order_id: Optional[str] = None
    venue: Optional[str] = None
    reject_reason: str = ""
    decision_price: Optional[float] = None  # arrival price for implementation-shortfall accounting

    @property
    def side(self) -> Side:
        return Side.BUY if self.quantity > 0 else Side.SELL

    @property
    def remaining(self) -> float:
        return self.quantity - self.filled_quantity

    @property
    def is_open(self) -> bool:
        return not self.status.is_terminal


@dataclass
class Event:
    ts: pd.Timestamp

    @property
    def type(self) -> EventType:  # pragma: no cover - overridden
        raise NotImplementedError


@dataclass
class MarketEvent(Event):
    """Cross-sectional bar snapshot: every bar that closed at ``ts``."""

    bars: Dict[str, Bar] = field(default_factory=dict)

    @property
    def type(self) -> EventType:
        return EventType.MARKET


@dataclass
class TickEvent(Event):
    tick: Tick = None  # type: ignore[assignment]

    @property
    def type(self) -> EventType:
        return EventType.TICK


@dataclass
class BookEvent(Event):
    book: OrderBookSnapshot = None  # type: ignore[assignment]

    @property
    def type(self) -> EventType:
        return EventType.BOOK


@dataclass
class DecisionEvent(Event):
    """Emitted after a MarketEvent has been fully processed (fills settled), so
    alpha/portfolio logic sees a consistent post-fill state."""

    @property
    def type(self) -> EventType:
        return EventType.DECISION


@dataclass
class SignalEvent(Event):
    model: str = ""
    forecasts: Dict[str, float] = field(default_factory=dict)

    @property
    def type(self) -> EventType:
        return EventType.SIGNAL


@dataclass
class TargetEvent(Event):
    weights: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def type(self) -> EventType:
        return EventType.TARGET


@dataclass
class OrderEvent(Event):
    order: Order = None  # type: ignore[assignment]

    @property
    def type(self) -> EventType:
        return EventType.ORDER


@dataclass
class OrderUpdateEvent(Event):
    order_id: str = ""
    status: OrderStatus = OrderStatus.ACKNOWLEDGED
    reason: str = ""
    broker_order_id: Optional[str] = None

    @property
    def type(self) -> EventType:
        return EventType.ORDER_UPDATE


@dataclass
class FillEvent(Event):
    order_id: str = ""
    symbol: str = ""
    quantity: float = 0.0  # signed
    price: float = 0.0
    commission: float = 0.0
    slippage_cost: float = 0.0  # currency cost vs reference price (positive = cost)
    venue: str = "SIM"
    is_roll: bool = False

    @property
    def type(self) -> EventType:
        return EventType.FILL


class RiskLevel(enum.Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    BREACH = "BREACH"
    HALT = "HALT"


@dataclass
class RiskEvent(Event):
    level: RiskLevel = RiskLevel.INFO
    kind: str = ""
    message: str = ""
    value: float = 0.0
    limit: float = 0.0

    @property
    def type(self) -> EventType:
        return EventType.RISK


@dataclass
class TimerEvent(Event):
    name: str = ""

    @property
    def type(self) -> EventType:
        return EventType.TIMER
