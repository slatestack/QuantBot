import json

import numpy as np

from quantbot.risk_management import RiskLimits
from quantbot.simulation.backtest_engine import BacktestEngine
from quantbot.monitoring import MetricsRegistry


def test_end_to_end_backtest_runs_and_reports(universe, registry, small_cfg, tmp_path):
    frames = {s: f for s, f in universe.bars.items()}
    metrics = MetricsRegistry()
    eng = BacktestEngine(small_cfg, frames, registry, RiskLimits(), metrics_exporter=metrics)
    res = eng.run()
    assert len(res.equity) == len(eng.feed)
    assert np.isfinite(res.metrics["sharpe"])
    assert res.metrics["n_trades"] > 50
    # every non-roll fill belongs to an order the OMS knows about
    known = set(eng.system.oms.orders)
    assert set(res.trades.loc[~res.trades["is_roll"].astype(bool), "order_id"]) <= known
    # leverage limits respected at every mark (small tolerance for intrabar drift)
    lev = res.equity["gross_exposure"] / res.equity["equity"]
    assert lev.max() < RiskLimits().max_gross_leverage * 1.25
    out = res.save(tmp_path / "rep", plot=False)
    assert json.loads((out / "metrics.json").read_text())["metrics"]["final_equity"] > 0
    text = metrics.render()
    assert "quantbot_equity" in text and "quantbot_fills_total" in text
