import numpy as np
import pandas as pd
import pytest

from quantbot.core.instruments import FutureContract
from quantbot.data_engineering.futures_roller import FuturesRoller
from quantbot.execution.algos import TWAP, VWAP, AlmgrenChriss
from quantbot.quant_models.alpha.commodity_term import roll_yield
from quantbot.quant_models.alpha.ml_forecaster import LinearOnnxModel, fit_ridge
from quantbot.quant_models.alpha.stat_arb import KalmanHedgeRatio
from quantbot.quant_models.features.frac_diff import ffd_weights, frac_diff_ffd
from quantbot.quant_models.features.microstructure import order_book_imbalance, order_flow_imbalance
from quantbot.quant_models.features.volatility import yang_zhang
from quantbot.quant_models.portfolio import hrp_weights, ledoit_wolf, marcenko_pastur_denoise
from quantbot.quant_models.portfolio.sizing import PositionSizer
from quantbot.data_engineering.normalizers.schemas import OrderBookSnapshot
from quantbot.research.ml_pipelines import CombinatorialPurgedCV, PurgedKFold, triple_barrier_labels
from quantbot.research.statistical_tests import hurst_exponent, ou_fit
from quantbot.simulation.performance import deflated_sharpe, probabilistic_sharpe

rng = np.random.default_rng(42)


def test_kalman_recovers_hedge_ratio():
    x = np.cumsum(rng.normal(0, 0.01, 3000)) + 4
    y = 0.5 + 1.7 * x + rng.normal(0, 0.01, 3000)
    kf = KalmanHedgeRatio(delta=1e-6, obs_var=1e-4)
    for a, b in zip(y, x):
        kf.step(a, b)
    assert kf.beta == pytest.approx(1.7, abs=0.1)


def test_ou_half_life():
    s = np.zeros(20000)
    for i in range(1, len(s)):
        s[i] = 0.9 * s[i - 1] + rng.normal()
    assert ou_fit(s)["half_life"] == pytest.approx(np.log(2) / -np.log(0.9), rel=0.1)


def test_hurst_regimes():
    rw = np.cumsum(rng.normal(size=5000))
    assert 0.4 < hurst_exponent(rw) < 0.6


def test_yang_zhang_close_to_true_vol():
    n, sig = 3000, 0.02
    c = 100 * np.exp(np.cumsum(rng.normal(0, sig, n)))
    o = np.r_[c[0], c[:-1]] * np.exp(rng.normal(0, sig * 0.3, n))
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, sig * 0.5, n)))
    l = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, sig * 0.5, n)))
    assert 0.5 * sig < yang_zhang(o, h, l, c) < 2 * sig


def test_frac_diff_weights_and_d1_equals_diff():
    w = ffd_weights(1.0, 1e-5)
    assert np.allclose(w, [-1.0, 1.0])
    s = pd.Series(np.arange(10.0))
    assert np.allclose(frac_diff_ffd(s, 1.0).dropna(), 1.0)


def test_obi_ofi():
    b0 = OrderBookSnapshot("A", pd.Timestamp(0, tz="UTC"), ((100, 10),), ((101, 30),))
    b1 = OrderBookSnapshot("A", pd.Timestamp(1, tz="UTC"), ((100, 20),), ((101, 30),))
    assert order_book_imbalance(b0) == pytest.approx(-0.5)
    assert order_flow_imbalance(b0, b1) == pytest.approx(10.0)


def test_ledoit_wolf_and_mp_are_psd_and_hrp_sums_to_one():
    X = rng.normal(size=(120, 15)) * rng.uniform(0.5, 2, 15)
    C = ledoit_wolf(X)
    assert np.all(np.linalg.eigvalsh(C) > 0)
    D = marcenko_pastur_denoise(np.cov(X, rowvar=False), 120)
    assert np.all(np.linalg.eigvalsh(D) > -1e-10)
    w = hrp_weights(pd.DataFrame(C))
    assert w.sum() == pytest.approx(1.0) and (w > 0).all()
    # lower-variance assets receive more weight on average
    assert np.corrcoef(w.values, np.diag(C))[0, 1] < 0


def test_vol_targeting_hits_target():
    cov = pd.DataFrame(np.diag([0.01 ** 2, 0.02 ** 2]), index=["a", "b"], columns=["a", "b"])
    sz = PositionSizer(target_vol=0.10, max_gross_leverage=10, max_weight=10, kelly_fraction=None)
    w = sz.size({"a": 1.0, "b": 0.5}, cov)
    vol = np.sqrt(np.array([w["a"], w["b"]]) @ cov.values @ np.array([w["a"], w["b"]]) * 252)
    assert vol == pytest.approx(0.10, rel=1e-6)


def test_execution_algos_conserve_quantity():
    for algo in (TWAP(7), VWAP(13), AlmgrenChriss(10, risk_aversion=1e-5)):
        s = algo.schedule(1003, lot=1, sigma=0.02, price=50)
        assert sum(s) == pytest.approx(1003) and all(x > 0 for x in s)
        s = algo.schedule(-500, lot=1, sigma=0.02, price=50)
        assert sum(s) == pytest.approx(-500) and all(x < 0 for x in s)
    ac = AlmgrenChriss(10, risk_aversion=1e-3).schedule(1000, lot=1, sigma=0.02, price=50)
    assert ac[0] > ac[-1]  # risk aversion front-loads


def test_roll_yield_sign():
    assert roll_yield(100, 98, 20, 50) > 0  # backwardation
    assert roll_yield(100, 102, 20, 50) < 0  # contango


def test_panama_preserves_contract_pnl():
    idx = pd.bdate_range("2024-01-01", "2024-04-30", tz="UTC")
    c1 = FutureContract("XH24", "X", pd.Timestamp("2024-02-20"))
    c2 = FutureContract("XK24", "X", pd.Timestamp("2024-04-20"))
    c3 = FutureContract("XN24", "X", pd.Timestamp("2024-06-20"))
    frames = {}
    for c, base in ((c1, 100.0), (c2, 105.0), (c3, 110.0)):
        p = base + np.cumsum(rng.normal(0, 1, len(idx)))
        frames[c.symbol] = pd.DataFrame({"open": p, "high": p + 1, "low": p - 1, "close": p, "volume": 1000.0},
                                        index=idx)
    cont = FuturesRoller(5, "panama").stitch([c1, c2, c3], frames)
    assert cont["roll"].sum() == 2
    held = cont.index[cont.index < pd.Timestamp("2024-02-12", tz="UTC")]
    d_adj = cont.loc[held, "close"].diff().dropna()
    d_raw = frames["XH24"].loc[held, "close"].diff().dropna()
    assert np.allclose(d_adj.values, d_raw.values)


def test_onnx_inference_matches_numpy():
    X = rng.normal(size=(500, 9))
    y = X @ rng.normal(size=9) + rng.normal(size=500)
    m = fit_ridge(X, y, 1.0)
    ref = (((X - m.mu) / m.sd) @ m.w + m.b)
    assert np.allclose(m.predict(X), ref, atol=1e-4)
    import onnxruntime as ort
    sess = ort.InferenceSession(m.to_onnx_bytes(), providers=["CPUExecutionProvider"])
    assert np.allclose(sess.run(None, {"X": X.astype(np.float32)})[0].ravel(), ref, atol=1e-4)


def test_purged_kfold_has_no_label_overlap():
    t0 = pd.date_range("2020-01-01", periods=300, freq="D")
    t1 = pd.Series(t0 + pd.Timedelta(days=10))
    for tr, te in PurgedKFold(5, 0.02).split(pd.Index(t0), t1):
        te_start, te_end = t0[te[0]], t1.iloc[te].max()
        tr_t0, tr_t1 = t0[tr], t1.iloc[tr]
        assert not ((tr_t0 <= te_end) & (tr_t1.values >= te_start)).any()
    cv = CombinatorialPurgedCV(6, 2)
    assert sum(1 for _ in cv.split(pd.Index(t0), t1)) == 15 and cv.n_paths == 5


def test_triple_barrier_labels():
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400))),
                      index=pd.date_range("2020-01-01", periods=400, freq="D"))
    lab = triple_barrier_labels(close, horizon=10)
    assert set(lab["label"].unique()) <= {-1, 0, 1}
    assert (lab["t1"] > lab.index).all()


def test_psr_and_dsr():
    r = rng.normal(0.001, 0.01, 2000)
    psr = probabilistic_sharpe(r)
    assert 0.9 < psr <= 1.0
    assert deflated_sharpe(r, n_trials=100) < psr
