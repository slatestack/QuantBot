"""Continuous futures contract stitching and roll scheduling.

Given a set of dated contracts, builds a continuous series that holds the front
contract until ``roll_days_before`` business days before its first-notice /
last-safe day, then rolls to the next contract.

Adjustment methods
------------------
* ``panama`` (additive back-adjustment): preserves *dollar* PnL per contract
  exactly - used for PnL accounting in the backtester.
* ``ratio`` (multiplicative): preserves *percentage* returns; can't go negative.
* ``none``: raw spliced series (has artificial roll gaps - research only).

The output frame also carries term-structure auxiliary columns used by the
commodity carry model: ``f1`` (unadjusted held-contract close), ``f2`` (next
contract close), ``t1``/``t2`` (calendar days to expiry), ``roll`` (1 on roll
days) and ``ret`` (true one-day return of the held position, roll-aware).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from quantbot.core.instruments import FutureContract


@dataclass
class RollEvent:
    date: pd.Timestamp
    from_contract: str
    to_contract: str
    gap: float  # new_close - old_close on roll date
    ratio: float  # new_close / old_close


class FuturesRoller:
    def __init__(self, roll_days_before: int = 5, method: str = "panama"):
        if method not in ("panama", "ratio", "none"):
            raise ValueError(method)
        self.roll_days_before = roll_days_before
        self.method = method

    def roll_date(self, contract: FutureContract) -> pd.Timestamp:
        return (contract.last_safe_day - pd.offsets.BDay(self.roll_days_before)).normalize()

    def active_contract_schedule(self, contracts: Sequence[FutureContract], dates: pd.DatetimeIndex) -> pd.Series:
        """Which contract to hold on each date (pure function of the calendar -> no lookahead)."""
        cs = sorted(contracts, key=lambda c: c.expiry)
        if not cs:
            return pd.Series([None] * len(dates), index=dates, dtype=object)
        naive = dates.tz_localize(None) if dates.tz is not None else dates
        dn = naive.normalize().to_numpy()
        roll_dates = np.array([pd.Timestamp(self.roll_date(c)).tz_localize(None)
                               if pd.Timestamp(self.roll_date(c)).tzinfo else self.roll_date(c) for c in cs],
                              dtype="datetime64[ns]")
        # roll dates are monotone in expiry; first contract whose roll date is strictly after the date
        order = np.argsort(roll_dates, kind="stable")
        rd_sorted = roll_dates[order]
        pos = np.searchsorted(rd_sorted, dn, side="right")
        syms = np.array([cs[i].symbol for i in order] + [None], dtype=object)
        return pd.Series(syms[pos], index=dates, dtype=object)

    def stitch(self, contracts: Sequence[FutureContract], frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        cs = sorted(contracts, key=lambda c: c.expiry)
        by_sym = {c.symbol: c for c in cs}
        all_dates = sorted(set().union(*[frames[c.symbol].index for c in cs if c.symbol in frames]))
        dates = pd.DatetimeIndex(all_dates)
        schedule = self.active_contract_schedule(cs, dates)
        rows = []
        pos_of = {c.symbol: i for i, c in enumerate(cs)}
        prev_sym: Optional[str] = None
        prev_close_same: Optional[float] = None
        rolls: List[RollEvent] = []
        for d, sym in schedule.items():
            if sym is None or sym not in frames or d not in frames[sym].index:
                continue
            bar = frames[sym].loc[d]
            idx = pos_of[sym]
            nxt = cs[idx + 1] if idx + 1 < len(cs) else None
            f2 = frames[nxt.symbol]["close"].get(d, np.nan) if nxt is not None and nxt.symbol in frames else np.nan
            dn = d.tz_localize(None) if d.tzinfo is not None else d
            t1 = (by_sym[sym].expiry - dn).days
            t2 = (nxt.expiry - dn).days if nxt is not None else np.nan
            rolled = prev_sym is not None and sym != prev_sym
            if rolled:
                old_close = frames[prev_sym]["close"].get(d, np.nan)
                if np.isnan(old_close):
                    old_close = rows[-1]["close"]
                rolls.append(RollEvent(d, prev_sym, sym, bar["close"] - old_close, bar["close"] / old_close))
                # return of the position: held old contract until today's close, then switched
                ret = old_close / prev_close_same - 1.0 if prev_close_same else 0.0
            else:
                ret = bar["close"] / prev_close_same - 1.0 if prev_close_same else 0.0
            rows.append({
                "ts_event": d, "open": bar["open"], "high": bar["high"], "low": bar["low"], "close": bar["close"],
                "volume": bar.get("volume", 0.0), "contract": sym, "f1": bar["close"], "f2": f2,
                "t1": float(t1), "t2": float(t2), "roll": 1.0 if rolled else 0.0, "ret": ret,
            })
            prev_sym = sym
            prev_close_same = bar["close"]
        df = pd.DataFrame(rows).set_index("ts_event")
        df = self._adjust(df, rolls)
        df.attrs["rolls"] = rolls
        return df

    def _adjust(self, df: pd.DataFrame, rolls: List[RollEvent]) -> pd.DataFrame:
        if self.method == "none" or not rolls:
            return df
        px = ["open", "high", "low", "close"]
        out = df.copy()
        for r in rolls:
            before = out.index < r.date
            if self.method == "panama":
                out.loc[before, px] = out.loc[before, px] + r.gap
            else:
                out.loc[before, px] = out.loc[before, px] * r.ratio
        if self.method == "panama" and (out[px] <= 0).any().any():
            # shift the whole series up so the level stays positive; differences (dollar PnL) are unchanged
            shift = -out[px].min().min() + 1.0
            out[px] = out[px] + shift
            out.attrs["panama_shift"] = shift
        return out


class RollManager:
    """Live mapping from a futures root (what strategies trade) to the dated
    contract the gateway must route to, plus roll-order generation."""

    def __init__(self, contracts: Dict[str, Sequence[FutureContract]], roll_days_before: int = 5):
        self.roller = FuturesRoller(roll_days_before)
        self.contracts = {root: sorted(cs, key=lambda c: c.expiry) for root, cs in contracts.items()}

    def active(self, root: str, ts: pd.Timestamp) -> Optional[str]:
        s = self.roller.active_contract_schedule(self.contracts[root], pd.DatetimeIndex([ts]))
        return s.iloc[0]

    def roll_orders(self, root: str, ts: pd.Timestamp, held_contract: str, qty: float) -> List[tuple]:
        """Return [(contract, signed_qty), ...] needed to roll ``qty`` from ``held_contract``."""
        target = self.active(root, ts)
        if target is None or target == held_contract or qty == 0:
            return []
        return [(held_contract, -qty), (target, qty)]
