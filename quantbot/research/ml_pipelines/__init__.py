"""Offline ML research pipeline (Lopez de Prado, *Advances in Financial ML*).

* Triple-barrier labelling with volatility-scaled barriers.
* Purged K-Fold CV with embargo - removes training samples whose label
  window overlaps the test fold, preventing leakage across overlapping horizons.
* Combinatorial Purged CV (CPCV) - yields a distribution of out-of-sample
  paths instead of a single backtest path.
* Permutation (MDA) feature importance on purged folds.
* Model registry exporting ONNX artefacts consumed by ``MLForecaster``.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from quantbot.quant_models.alpha.ml_forecaster import FEATURE_NAMES, LinearOnnxModel, compute_features, fit_ridge


# ----------------------------------------------------------------- labelling
def triple_barrier_labels(close: pd.Series, horizon: int = 10, pt_mult: float = 2.0, sl_mult: float = 2.0,
                          vol_window: int = 50) -> pd.DataFrame:
    """Label each bar with the first barrier touched: +1 (profit-take),
    -1 (stop-loss) or sign of return at the vertical barrier. Returns columns
    [label, ret, t1] where ``t1`` is the label end time (for purging)."""
    r = np.log(close).diff()
    vol = r.ewm(span=vol_window).std()
    px = close.to_numpy()
    idx = close.index
    out = []
    for i in range(len(close) - 1):
        v = vol.iloc[i]
        if not np.isfinite(v) or v == 0:
            continue
        end = min(i + horizon, len(close) - 1)
        path = np.log(px[i + 1: end + 1] / px[i])
        up, dn = pt_mult * v, -sl_mult * v
        hit_up = np.where(path >= up)[0]
        hit_dn = np.where(path <= dn)[0]
        first_up = hit_up[0] if len(hit_up) else np.inf
        first_dn = hit_dn[0] if len(hit_dn) else np.inf
        if first_up < first_dn:
            lab, j = 1, int(first_up)
        elif first_dn < first_up:
            lab, j = -1, int(first_dn)
        else:
            j = len(path) - 1
            lab = int(np.sign(path[-1]))
        out.append({"ts": idx[i], "label": lab, "ret": float(path[j]), "t1": idx[i + 1 + j]})
    return pd.DataFrame(out).set_index("ts")


# ------------------------------------------------------------ cross-validation
class PurgedKFold:
    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(self, t0: pd.Index, t1: pd.Series) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """``t0`` = sample start times (sorted), ``t1`` = label end times."""
        n = len(t0)
        embargo = int(n * self.embargo_pct)
        bounds = np.array_split(np.arange(n), self.n_splits)
        t0v, t1v = _as_ns(t0), _as_ns(t1)
        for test_idx in bounds:
            yield _purged_train(t0v, t1v, [test_idx], embargo), test_idx


def _as_ns(t) -> np.ndarray:
    """Timestamps (tz-aware or naive, Index or Series) -> int64 ns array."""
    idx = pd.DatetimeIndex(pd.Series(t).reset_index(drop=True) if not isinstance(t, pd.Index) else t)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx.as_unit("ns").asi8


def _purged_train(t0: np.ndarray, t1: np.ndarray, test_blocks: List[np.ndarray], embargo: int) -> np.ndarray:
    n = len(t0)
    keep = np.ones(n, bool)
    for blk in test_blocks:
        start, end = t0[blk].min(), t1[blk].max()
        keep[blk] = False
        # purge: training labels overlapping [start, end]
        overlap = (t0 <= end) & (t1 >= start)
        keep &= ~overlap
        # embargo after the test block
        e0 = blk[-1] + 1
        keep[e0: min(n, e0 + embargo)] = False
    return np.where(keep)[0]


class CombinatorialPurgedCV:
    """CPCV(N, k): split into N groups, test on every combination of k groups.
    Produces C(N, k) splits and phi = k/N * C(N,k) out-of-sample paths."""

    def __init__(self, n_groups: int = 6, k_test: int = 2, embargo_pct: float = 0.01):
        self.n_groups, self.k_test, self.embargo_pct = n_groups, k_test, embargo_pct

    @property
    def n_paths(self) -> int:
        from math import comb

        return comb(self.n_groups, self.k_test) * self.k_test // self.n_groups

    def split(self, t0: pd.Index, t1: pd.Series):
        n = len(t0)
        embargo = int(n * self.embargo_pct)
        groups = np.array_split(np.arange(n), self.n_groups)
        t0v, t1v = _as_ns(t0), _as_ns(t1)
        for combo in itertools.combinations(range(self.n_groups), self.k_test):
            test_blocks = [groups[g] for g in combo]
            yield combo, _purged_train(t0v, t1v, test_blocks, embargo), np.concatenate(test_blocks)

    def backtest_paths(self, t0, t1, fit_predict: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> np.ndarray:
        """Assemble the phi OOS prediction paths (shape [n_paths, n_samples])."""
        n = len(t0)
        groups = np.array_split(np.arange(n), self.n_groups)
        preds_by_group: Dict[int, List[np.ndarray]] = {g: [] for g in range(self.n_groups)}
        for combo, tr, te in self.split(t0, t1):
            p = fit_predict(tr, te)
            pos = 0
            for g in combo:
                k = len(groups[g])
                preds_by_group[g].append(p[pos: pos + k])
                pos += k
        paths = np.full((self.n_paths, n), np.nan)
        for g in range(self.n_groups):
            for j, arr in enumerate(preds_by_group[g][: self.n_paths]):
                paths[j, groups[g]] = arr
        return paths


# ------------------------------------------------------------------ dataset
def build_dataset(frames: Dict[str, pd.DataFrame], horizon: int = 5) -> pd.DataFrame:
    """Pooled panel of FEATURE_NAMES + vol-scaled forward return label + t1."""
    rows = []
    for sym, df in frames.items():
        r = (df["ret"] if "ret" in df.columns else df["close"].pct_change()).fillna(0.0).to_numpy()
        idx = df.index
        for i in range(260, len(df) - horizon):
            x = compute_features(r[: i + 1])
            if x is None:
                continue
            vol = np.std(r[i - 62: i + 1], ddof=1)
            y = np.clip(np.sum(r[i + 1: i + 1 + horizon]) / (vol * np.sqrt(horizon)), -3, 3)
            rows.append({"ts": idx[i], "t1": idx[i + horizon], "symbol": sym, "y": y,
                         **dict(zip(FEATURE_NAMES, x))})
    return pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)


def cv_score(ds: pd.DataFrame, cv, l2: float = 50.0) -> Dict[str, float]:
    X = ds[FEATURE_NAMES].to_numpy()
    y = ds["y"].to_numpy()
    t0, t1 = pd.Index(ds["ts"]), ds["t1"]
    ics = []
    splits = cv.split(t0, t1)
    for sp in splits:
        tr, te = (sp[1], sp[2]) if len(sp) == 3 else sp
        m = fit_ridge(X[tr], y[tr], l2)
        p = m.predict(X[te])
        if np.std(p) > 0:
            ics.append(np.corrcoef(p, y[te])[0, 1])
    ics = np.array(ics)
    return {"mean_ic": float(ics.mean()), "std_ic": float(ics.std()), "n_splits": len(ics),
            "ic_t": float(ics.mean() / (ics.std() / np.sqrt(len(ics)))) if ics.std() > 0 else np.nan}


def permutation_importance(ds: pd.DataFrame, cv: PurgedKFold, l2: float = 50.0, seed: int = 0) -> pd.Series:
    """Mean decrease in OOS IC when each feature is shuffled (MDA)."""
    rng = np.random.default_rng(seed)
    X = ds[FEATURE_NAMES].to_numpy()
    y = ds["y"].to_numpy()
    drops = {f: [] for f in FEATURE_NAMES}
    for tr, te in cv.split(pd.Index(ds["ts"]), ds["t1"]):
        m = fit_ridge(X[tr], y[tr], l2)
        base = np.corrcoef(m.predict(X[te]), y[te])[0, 1]
        for j, f in enumerate(FEATURE_NAMES):
            Xp = X[te].copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drops[f].append(base - np.corrcoef(m.predict(Xp), y[te])[0, 1])
    return pd.Series({f: float(np.mean(v)) for f, v in drops.items()}).sort_values(ascending=False)


# ----------------------------------------------------------------- registry
class ModelRegistry:
    def __init__(self, root: str | Path = "models"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, model: LinearOnnxModel, meta: dict) -> Path:
        path = self.root / f"{name}.onnx"
        path.write_bytes(model.to_onnx_bytes())
        (self.root / f"{name}.json").write_text(json.dumps({**meta, "features": FEATURE_NAMES}, indent=2,
                                                           default=str))
        return path

    def list(self) -> List[str]:
        return sorted(p.stem for p in self.root.glob("*.onnx"))


def train_and_register(frames: Dict[str, pd.DataFrame], name: str = "ridge_forecaster", horizon: int = 5,
                       l2: float = 50.0, registry_root: str = "models") -> dict:
    ds = build_dataset(frames, horizon)
    pk = cv_score(ds, PurgedKFold(5, 0.01), l2)
    cp = cv_score(ds, CombinatorialPurgedCV(6, 2, 0.01), l2)
    X, y = ds[FEATURE_NAMES].to_numpy(), ds["y"].to_numpy()
    model = fit_ridge(X, y, l2)
    meta = {"horizon": horizon, "l2": l2, "n_samples": len(ds), "purged_kfold": pk, "cpcv": cp,
            "train_end": str(ds["ts"].max())}
    path = ModelRegistry(registry_root).save(name, model, meta)
    return {"path": str(path), **meta}
