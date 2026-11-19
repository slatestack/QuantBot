"""Machine-learning forecaster with ONNX Runtime inference.

Two modes:
* ``walk_forward`` (default): a pooled cross-asset ridge model is (re)trained
  inside the event loop every ``retrain_every`` bars, using only samples whose
  forward label is fully realized (purged by the label horizon). The fitted
  model is compiled to an ONNX graph and executed with onnxruntime - the same
  artefact path as production inference.
* ``pretrained``: load an ONNX model produced offline by
  ``quantbot.research.ml_pipelines`` (trained with purged / combinatorial CV).
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np

from quantbot.core.state.history import HistoryBuffer
from quantbot.data_engineering.normalizers.schemas import Bar
from quantbot.quant_models.alpha.base import AlphaModel
from quantbot.quant_models.features import returns as feat_returns

log = logging.getLogger(__name__)

FEATURE_NAMES = ["mom_5", "mom_21", "mom_63", "mom_126", "rev_1", "vol_ratio", "dist_high", "dist_low", "skew_63"]


def compute_features(r: np.ndarray) -> Optional[np.ndarray]:
    """Features from a vector of past returns (most recent last). Needs >= 253 obs."""
    if len(r) < 253 or not np.all(np.isfinite(r[-253:])):
        return None
    vol = np.std(r[-63:], ddof=1)
    if vol <= 0:
        return None
    idx = np.cumprod(1 + r[-252:])

    def mom(h):
        return np.sum(r[-h:]) / (vol * np.sqrt(h))

    vol20 = np.std(r[-20:], ddof=1)
    vol126 = np.std(r[-126:], ddof=1)
    x = r[-63:]
    skew = float(np.mean(((x - x.mean()) / (x.std() + 1e-12)) ** 3))
    return np.array([mom(5), mom(21), mom(63), mom(126), r[-1] / vol, vol20 / vol126 - 1.0,
                     idx[-1] / idx.max() - 1.0, idx[-1] / idx.min() - 1.0, np.clip(skew, -5, 5)], dtype=float)


class LinearOnnxModel:
    """y = ((x - mu) / sd) @ w + b, compiled to ONNX and run with onnxruntime."""

    def __init__(self, mu: np.ndarray, sd: np.ndarray, w: np.ndarray, b: float):
        self.mu, self.sd, self.w, self.b = mu.astype(np.float32), sd.astype(np.float32), w.astype(np.float32), float(b)
        self._sess = None
        try:
            self._sess = self._build_session()
        except Exception as exc:  # onnx/onnxruntime missing -> numpy fallback (identical math)
            log.debug("ONNX unavailable, using numpy inference: %s", exc)

    def to_onnx_bytes(self) -> bytes:
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        f = len(self.w)
        inits = [numpy_helper.from_array(self.mu.reshape(1, f), "mu"),
                 numpy_helper.from_array(self.sd.reshape(1, f), "sd"),
                 numpy_helper.from_array(self.w.reshape(f, 1), "W"),
                 numpy_helper.from_array(np.array([self.b], np.float32), "B")]
        nodes = [helper.make_node("Sub", ["X", "mu"], ["xc"]), helper.make_node("Div", ["xc", "sd"], ["xs"]),
                 helper.make_node("MatMul", ["xs", "W"], ["xw"]), helper.make_node("Add", ["xw", "B"], ["Y"])]
        graph = helper.make_graph(nodes, "linear_forecaster",
                                  [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, f])],
                                  [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, 1])], inits)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], producer_name="quantbot")
        model.ir_version = min(model.ir_version, 9)
        onnx.checker.check_model(model)
        return model.SerializeToString()

    _shared_sessions: Dict[int, object] = {}

    @classmethod
    def _parametric_session(cls, n_features: int):
        """One cached ONNX session per feature count whose weights are *inputs*,
        so walk-forward retraining does not rebuild the runtime graph."""
        if n_features in cls._shared_sessions:
            return cls._shared_sessions[n_features]
        import onnxruntime as ort
        from onnx import TensorProto, helper

        f = n_features
        nodes = [helper.make_node("Sub", ["X", "mu"], ["xc"]), helper.make_node("Div", ["xc", "sd"], ["xs"]),
                 helper.make_node("MatMul", ["xs", "W"], ["xw"]), helper.make_node("Add", ["xw", "B"], ["Y"])]
        inputs = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, f]),
                  helper.make_tensor_value_info("mu", TensorProto.FLOAT, [1, f]),
                  helper.make_tensor_value_info("sd", TensorProto.FLOAT, [1, f]),
                  helper.make_tensor_value_info("W", TensorProto.FLOAT, [f, 1]),
                  helper.make_tensor_value_info("B", TensorProto.FLOAT, [1])]
        graph = helper.make_graph(nodes, "linear_forecaster_param", inputs,
                                  [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, 1])])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], producer_name="quantbot")
        model.ir_version = min(model.ir_version, 9)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        sess = ort.InferenceSession(model.SerializeToString(), opts, providers=["CPUExecutionProvider"])
        cls._shared_sessions[f] = sess
        return sess

    def _build_session(self):
        return self._parametric_session(len(self.w))

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, np.float32)
        if self._sess is not None:
            f = len(self.w)
            return self._sess.run(None, {"X": X, "mu": self.mu.reshape(1, f), "sd": self.sd.reshape(1, f),
                                         "W": self.w.reshape(f, 1), "B": np.array([self.b], np.float32)})[0].ravel()
        return (((X - self.mu) / self.sd) @ self.w + self.b).ravel()


class OnnxFileModel:
    def __init__(self, path: str):
        import onnxruntime as ort

        self._sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self._in = self._sess.get_inputs()[0].name

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._sess.run(None, {self._in: np.asarray(X, np.float32)})[0], float).ravel()


def fit_ridge(X: np.ndarray, y: np.ndarray, l2: float = 10.0) -> LinearOnnxModel:
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    b = y.mean()
    A = Z.T @ Z + l2 * np.eye(Z.shape[1])
    w = np.linalg.solve(A, Z.T @ (y - b))
    return LinearOnnxModel(mu, sd, w, b)


class MLForecaster(AlphaModel):
    name = "ml"

    def __init__(self, symbols: Iterable[str], horizon: int = 5, retrain_every: int = 21, min_samples: int = 750,
                 max_samples: int = 20_000, l2: float = 50.0, mode: str = "walk_forward",
                 model_path: Optional[str] = None, smooth_halflife: float = 0.0, name: Optional[str] = None):
        super().__init__(symbols, name)
        self.h = horizon
        self.retrain_every = retrain_every
        self.min_samples = min_samples
        self.l2 = l2
        self.mode = mode
        self.model = OnnxFileModel(model_path) if mode == "pretrained" and model_path else None
        self.pred_scale = 1.0
        # EWMA smoothing of output forecasts (match signal turnover to the label horizon)
        self.smooth_alpha = 1 - 0.5 ** (1 / smooth_halflife) if smooth_halflife > 0 else 1.0
        self._smoothed: Dict[str, float] = {}
        # per symbol: pending feature rows awaiting label realisation [(bar_idx, x, vol)]
        self._pending: Dict[str, Deque[Tuple[int, np.ndarray, float]]] = {s: deque() for s in self.symbols}
        self._X: Deque[np.ndarray] = deque(maxlen=max_samples)
        self._y: Deque[float] = deque(maxlen=max_samples)
        self._n: Dict[str, int] = {s: 0 for s in self.symbols}
        self._bars_seen = 0
        self.train_log: List[dict] = []

    @property
    def warmup_bars(self) -> int:
        return 260

    def _realize_labels(self, sym: str, r: np.ndarray) -> None:
        """Move pending rows whose forward horizon has fully elapsed into the training set."""
        n = self._n[sym]
        pend = self._pending[sym]
        while pend and n - pend[0][0] >= self.h:
            i, x, vol = pend.popleft()
            k = n - i  # bars elapsed since the feature row
            fwd = r[len(r) - k: len(r) - k + self.h]
            if len(fwd) == self.h and vol > 0:
                self._X.append(x)
                self._y.append(float(np.clip(np.sum(fwd) / (vol * np.sqrt(self.h)), -3, 3)))

    def on_bar(self, ts, bars: Dict[str, Bar], history: HistoryBuffer) -> Optional[Dict[str, float]]:
        self._bars_seen += 1
        feats: Dict[str, np.ndarray] = {}
        for sym in self.symbols:
            if sym not in bars:
                continue
            self._n[sym] += 1
            r = feat_returns(history, sym, 300)
            self._realize_labels(sym, r)
            x = compute_features(r)
            if x is None:
                continue
            self._pending[sym].append((self._n[sym], x, float(np.std(r[-63:], ddof=1))))
            feats[sym] = x
        if self.mode == "walk_forward" and self._bars_seen % self.retrain_every == 0 and len(self._y) >= self.min_samples:
            X, y = np.asarray(self._X), np.asarray(self._y)
            self.model = fit_ridge(X, y, self.l2)
            pred = self.model.predict(X)
            self.pred_scale = float(np.std(pred)) or 1.0
            ic = float(np.corrcoef(pred, y)[0, 1]) if np.std(pred) > 0 else 0.0
            self.train_log.append({"ts": ts, "n": len(y), "in_sample_ic": ic})
        if self.model is None or not feats:
            return None
        syms = list(feats)
        preds = self.model.predict(np.vstack([feats[s] for s in syms]))
        out = {}
        for s, p in zip(syms, preds):
            raw = float(np.tanh(p / (2 * self.pred_scale)))
            prev = self._smoothed.get(s)
            sm = raw if prev is None else (1 - self.smooth_alpha) * prev + self.smooth_alpha * raw
            self._smoothed[s] = sm
            out[s] = self.clip(sm)
        return out
