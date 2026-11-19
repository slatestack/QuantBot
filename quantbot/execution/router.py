"""Smart order router: picks a gateway by symbol / asset class with failover."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from quantbot.core.events import Order
from quantbot.core.instruments import InstrumentRegistry

log = logging.getLogger(__name__)


class SmartOrderRouter:
    def __init__(self, registry: InstrumentRegistry, gateways: List, routes: Optional[Dict[str, List[str]]] = None):
        """``routes`` maps a symbol or asset class name (EQUITY/FUTURE) to an
        ordered list of gateway names (first healthy one wins)."""
        self.registry = registry
        self.gateways = {g.name: g for g in gateways}
        self.default = gateways[0].name
        self.routes = routes or {}

    def gateway_by_name(self, name: Optional[str]):
        return self.gateways.get(name or self.default, self.gateways[self.default])

    def route(self, order: Order):
        ins = self.registry.get(order.symbol)
        prefs = self.routes.get(order.symbol) or self.routes.get(ins.asset_class.value) or [self.default]
        for name in prefs:
            gw = self.gateways.get(name)
            if gw is not None and getattr(gw, "connected", True):
                return gw
        log.warning("no healthy preferred gateway for %s, using default", order.symbol)
        return self.gateways[self.default]
