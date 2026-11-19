# %% [markdown]
# # Alpha research workflow (jupytext percent-format; open with Jupyter + jupytext, or run as a script)
# 1. Load data  2. Stationarity / memory diagnostics  3. Pair screening
# 4. ML dataset + purged / combinatorial CV  5. Export ONNX model for production.

# %%
import numpy as np
import pandas as pd

from quantbot.config import load_config
from quantbot.data_engineering.loader import load_frames
from quantbot.quant_models.features.frac_diff import min_ffd_order
from quantbot.research.ml_pipelines import (CombinatorialPurgedCV, PurgedKFold, build_dataset, cv_score,
                                            permutation_importance)
from quantbot.research.statistical_tests import engle_granger, hurst_exponent

cfg = load_config("config/strategy_etf.yaml")
frames = load_frames(cfg)

# %% Memory-preserving stationarity: minimum fractional-differencing order per asset
print({s: min_ffd_order(df["close"]) for s, df in list(frames.items())[:5]})

# %% Hurst exponents (trend vs mean reversion)
print(pd.Series({s: hurst_exponent(np.log(df["close"].to_numpy())) for s, df in frames.items()}).round(3))

# %% Pair screening: Engle-Granger on every equity pair in the same sector
syms = ["XOM", "CVX", "XLE", "KO", "PEP", "GLD", "SLV"]
rows = []
for i, a in enumerate(syms):
    for b in syms[i + 1:]:
        j = frames[a]["close"].to_frame("a").join(frames[b]["close"].to_frame("b"), how="inner").iloc[-756:]
        eg = engle_granger(np.log(j["a"].to_numpy()), np.log(j["b"].to_numpy()))
        rows.append({"pair": f"{a}/{b}", "p": eg["p_value"], "beta": eg["beta"]})
print(pd.DataFrame(rows).sort_values("p"))

# %% ML: purged K-fold and CPCV information coefficients
ds = build_dataset({s: frames[s] for s in ["SPY", "QQQ", "TLT", "GLD", "USO", "EEM"]}, horizon=5)
print("purged k-fold:", cv_score(ds, PurgedKFold(5, 0.01)))
print("CPCV(6,2):   ", cv_score(ds, CombinatorialPurgedCV(6, 2, 0.01)))
print(permutation_importance(ds, PurgedKFold(5, 0.01)))
