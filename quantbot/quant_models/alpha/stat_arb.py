"""Kalman-filter pairs trading with Ornstein-Uhlenbeck mean-reversion gating.

Observation model on log prices: y_t = beta_t * x_t + alpha_t + e_t, with the
state (beta, alpha) following a random walk. The one-step-ahead innovation e_t
(computed from the *prior* state - no lookahead) is the tradable spread and
z_t its value standardized against the trailing window. Trades are only taken when the
recent spread exhibits OU mean reversion with a half-life inside bounds.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from quantbot.core.state.history import HistoryBuffer
from quantbot.data_engineering.normalizers.schemas import Bar
from quantbot.quant_models.alpha.base import AlphaModel
from quantbot.research.statistical_tests import engle_granger, ou_fit


class KalmanHedgeRatio:
    def __init__(self, delta: float = 1e-7, obs_var: float = 1e-3, adaptive: bool = False, lam: float = 0.995):
        self.theta = np.zeros(2)
        self.P = np.eye(2)
        self.Q = delta / (1 - delta) * np.eye(2)
        self.R = obs_var
        self.adaptive = adaptive
        self.lam = lam
        self.n = 0

    def step(self, y: float, x: float) -> Tuple[float, float, float]:
        """Returns (innovation e, innovation variance S, beta_prior)."""
        H = np.array([x, 1.0])
        P_pred = self.P + self.Q
        beta_prior = self.theta[0]
        y_hat = H @ self.theta
        e = y - y_hat
        S = float(H @ P_pred @ H + self.R)
        K = P_pred @ H / S
        self.theta = self.theta + K * e
        self.P = P_pred - np.outer(K, H) @ P_pred
        if self.adaptive and self.n > 20:
            # Mehra-style adaptive measurement noise using the post-fit residual
            post = y - H @ self.theta
            self.R = max(self.lam * self.R + (1 - self.lam) * (post * post + H @ self.P @ H), 1e-8)
        self.n += 1
        return float(e), S, float(beta_prior)

    @property
    def beta(self) -> float:
        return float(self.theta[0])


class _PairState:
    def __init__(self, kf: KalmanHedgeRatio, window: int):
        self.kf = kf
        self.spread: Deque[float] = deque(maxlen=window)
        self.position = 0  # +1 long spread (long Y / short X), -1 short spread
        self.ly: Deque[float] = deque(maxlen=504)
        self.lx: Deque[float] = deque(maxlen=504)
        self.coint_p = np.nan
        self.z = 0.0
        self.half_life = np.nan


class KalmanPairsStatArb(AlphaModel):
    name = "stat_arb"

    def __init__(self, pairs: Sequence[Tuple[str, str]], entry_z: float = 2.0, exit_z: float = 0.5,
                 stop_z: float = 4.0, ou_window: int = 120, min_half_life: float = 1.0,
                 max_half_life: float = 40.0, delta: float = 1e-7, obs_var: float = 1e-3, warmup: int = 60,
                 coint_pvalue: Optional[float] = 0.10, coint_window: int = 252, coint_every: int = 21,
                 name: Optional[str] = None):
        syms: List[str] = []
        for y, x in pairs:
            syms += [y, x]
        super().__init__(dict.fromkeys(syms), name)
        self.pairs = [tuple(p) for p in pairs]
        self.entry_z, self.exit_z, self.stop_z = entry_z, exit_z, stop_z
        self.min_hl, self.max_hl = min_half_life, max_half_life
        self.warmup = warmup
        self.delta = delta
        self.obs_var = obs_var
        self.ou_window = ou_window
        self.coint_pvalue = coint_pvalue
        self.coint_window = coint_window
        self.coint_every = coint_every
        self._st: Dict[Tuple[str, str], _PairState] = {}

    @property
    def warmup_bars(self) -> int:
        return self.warmup

    def diagnostics(self) -> Dict[str, dict]:
        return {f"{y}/{x}": {"beta": st.kf.beta, "z": st.z, "half_life": st.half_life, "position": st.position,
                             "coint_p_trailing": st.coint_p}
                for (y, x), st in self._st.items()}

    def on_bar(self, ts, bars: Dict[str, Bar], history: HistoryBuffer) -> Optional[Dict[str, float]]:
        out: Dict[str, float] = {}
        for y_sym, x_sym in self.pairs:
            by, bx = bars.get(y_sym), bars.get(x_sym)
            if by is None or bx is None:
                continue
            st = self._st.setdefault((y_sym, x_sym), _PairState(KalmanHedgeRatio(self.delta, self.obs_var),
                                                                          self.ou_window))
            ly, lx = np.log(by.close), np.log(bx.close)
            e, S, beta = st.kf.step(ly, lx)
            st.ly.append(ly)
            st.lx.append(lx)
            if self.coint_pvalue is not None and len(st.ly) >= self.coint_window and st.kf.n % self.coint_every == 0:
                # rolling Engle-Granger test on trailing data only
                st.coint_p = engle_granger(np.asarray(st.ly)[-self.coint_window:],
                                           np.asarray(st.lx)[-self.coint_window:])["p_value"]
            st.spread.append(e)
            if len(st.spread) >= 30:
                sp = np.asarray(st.spread)
                sd = sp[:-1].std(ddof=1)
                # standardize against the *prior* window (excludes today's value from the stats)
                st.z = (e - sp[:-1].mean()) / sd if sd > 0 else 0.0
            else:
                st.z = e / np.sqrt(S) if S > 0 else 0.0
            if st.kf.n < self.warmup:
                continue
            if len(st.spread) >= 30 and st.kf.n % 5 == 0:
                st.half_life = ou_fit(np.asarray(st.spread))["half_life"]
            mean_reverting = np.isfinite(st.half_life) and self.min_hl <= st.half_life <= self.max_hl
            if self.coint_pvalue is not None:
                mean_reverting = mean_reverting and np.isfinite(st.coint_p) and st.coint_p <= self.coint_pvalue
            z = st.z
            if st.position == 0:
                if mean_reverting and z > self.entry_z and z < self.stop_z:
                    st.position = -1
                elif mean_reverting and z < -self.entry_z and z > -self.stop_z:
                    st.position = 1
            else:
                if abs(z) < self.exit_z or abs(z) > self.stop_z or not mean_reverting \
                        or (st.position == 1 and z > self.exit_z) or (st.position == -1 and z < -self.exit_z):
                    st.position = 0
            b = float(np.clip(st.kf.beta, 0.1, 3.0))
            scale = 1.0 / max(1.0, b)
            out[y_sym] = out.get(y_sym, 0.0) + st.position * scale
            out[x_sym] = out.get(x_sym, 0.0) - st.position * b * scale
        return {k: self.clip(v) for k, v in out.items()} or None
