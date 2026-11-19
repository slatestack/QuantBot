"""Pre-trade risk checks: the independent gatekeeper between the portfolio
layer and the OMS. Every order passes ``check`` before reaching a gateway."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from quantbot.core.events import Order, OrderType
from quantbot.core.instruments import InstrumentRegistry
from quantbot.core.state.portfolio import Portfolio
from quantbot.risk_management.limits import RiskLimits


@dataclass
class CheckResult:
    approved: bool
    reason: str = ""


class PreTradeRisk:
    def __init__(self, limits: RiskLimits, registry: InstrumentRegistry):
        self.limits = limits
        self.registry = registry
        self.locked = False  # set by circuit breaker
        self.rejections: Dict[str, int] = {}

    def _reject(self, reason: str) -> CheckResult:
        key = reason.split(":")[0]
        self.rejections[key] = self.rejections.get(key, 0) + 1
        return CheckResult(False, reason)

    def check(self, order: Order, portfolio: Portfolio, ref_price: float, pending_qty: Optional[Dict[str, float]] = None,
              bar_volume: Optional[float] = None, risk_reducing_allowed: bool = True) -> CheckResult:
        lim = self.limits
        ins = self.registry.get(order.symbol)
        cur_qty = portfolio.positions[order.symbol].quantity if order.symbol in portfolio.positions else 0.0
        pend = (pending_qty or {}).get(order.symbol, 0.0)
        post_qty = cur_qty + pend + order.quantity
        reduces = abs(post_qty) < abs(cur_qty + pend) - 1e-12 and np.sign(post_qty) * np.sign(cur_qty + pend) >= 0

        if not np.isfinite(ref_price) or ref_price <= 0:
            return self._reject(f"no_reference_price: {order.symbol}")
        if order.quantity == 0 or not np.isfinite(order.quantity):
            return self._reject("invalid_quantity")
        if self.locked and not (reduces and risk_reducing_allowed):
            return self._reject("kill_switch_locked: trading halted")
        if order.symbol in lim.restricted_symbols and not reduces:
            return self._reject(f"restricted_symbol: {order.symbol}")
        if order.quantity < 0 and post_qty < 0 and not ins.is_future and not ins.shortable:
            return self._reject(f"not_shortable: {order.symbol}")

        equity = portfolio.equity
        if equity <= 0:
            return self._reject("non_positive_equity") if not reduces else CheckResult(True)
        notional = abs(ins.notional(order.quantity, ref_price))
        if reduces and self.locked:
            return CheckResult(True)  # kill-switch liquidation must never be blocked by size limits
        # fat-finger checks
        if abs(order.quantity) > lim.max_order_qty:
            return self._reject(f"max_order_qty: {abs(order.quantity)} > {lim.max_order_qty}")
        if notional > lim.max_order_notional:
            return self._reject(f"max_order_notional: {notional:,.0f} > {lim.max_order_notional:,.0f}")
        if notional > lim.max_order_pct_equity * equity and not reduces:
            return self._reject(f"max_order_pct_equity: {notional / equity:.2%}")
        # price collar for limit orders
        if order.order_type is OrderType.LIMIT and order.limit_price is not None:
            dev = abs(order.limit_price / ref_price - 1.0)
            if dev > lim.price_collar_pct:
                return self._reject(f"price_collar: {dev:.2%} from reference")
        if reduces:
            return CheckResult(True)
        if bar_volume is not None and bar_volume > 0 and abs(order.quantity) > lim.max_participation * bar_volume * 5:
            return self._reject(f"liquidity: order {abs(order.quantity):.0f} vs bar volume {bar_volume:.0f}")
        # concentration
        post_mv = abs(ins.notional(post_qty, ref_price))
        if post_mv > lim.max_position_pct_equity * equity * 1.0001:
            return self._reject(f"concentration: {order.symbol} {post_mv / equity:.2%} of equity")
        # leverage
        gross = portfolio.gross_exposure - abs(portfolio.market_value(order.symbol)) + post_mv
        if gross > lim.max_gross_leverage * equity * 1.0001:
            return self._reject(f"gross_leverage: {gross / equity:.2f}x")
        net = portfolio.net_exposure - portfolio.market_value(order.symbol) + ins.notional(post_qty, ref_price)
        if abs(net) > lim.max_net_leverage * equity * 1.0001:
            return self._reject(f"net_leverage: {net / equity:.2f}x")
        # margin availability
        im_after = portfolio.initial_margin_used() - ins.initial_margin_req(cur_qty, ref_price) \
            + ins.initial_margin_req(post_qty, ref_price)
        if equity - im_after < lim.margin_buffer_pct * equity:
            return self._reject(f"margin: free equity {(equity - im_after):,.0f} below buffer")
        return CheckResult(True)
