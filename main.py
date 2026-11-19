#!/usr/bin/env python3
"""QuantBot entrypoint.

    python main.py --backtest --config config/strategy_etf.yaml
    python main.py --paper    --config config/strategy_etf.yaml --broker sim      # local paper broker, live data
    python main.py --paper    --config config/strategy_etf.yaml --broker alpaca   # Alpaca paper account
    python main.py --live     --config config/strategy_etf.yaml --broker alpaca   # real money (guarded)
    python main.py --download --config config/strategy_etf.yaml                   # fill the Parquet data lake
    python main.py --research --config config/strategy_etf.yaml                   # stat tests on universe/pairs
    python main.py --train-ml --config config/strategy_etf.yaml                   # purged-CV model -> models/*.onnx
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from quantbot.config import load_config
from quantbot.core.instruments import InstrumentRegistry
from quantbot.risk_management.limits import RiskLimits

log = logging.getLogger("quantbot")


def _setup(args):
    overrides = {"data": {}}
    if args.start:
        overrides["data"]["start"] = args.start
    if args.end:
        overrides["data"]["end"] = args.end
    if args.capital:
        overrides["system"] = {"initial_capital": args.capital}
    if args.metrics_port:
        overrides["monitoring"] = {"enabled": True, "prometheus_port": args.metrics_port}
    cfg = load_config(args.config, overrides)
    registry = InstrumentRegistry.from_yaml(*cfg["instrument_files"])
    limits = RiskLimits.from_yaml(cfg["risk_limits_file"])
    return cfg, registry, limits


def cmd_backtest(args) -> int:
    from quantbot.data_engineering.loader import load_frames
    from quantbot.simulation.backtest_engine import BacktestEngine

    cfg, registry, limits = _setup(args)
    frames = load_frames(cfg)
    engine = BacktestEngine(cfg, frames, registry, limits, n_trials=args.n_trials)
    res = engine.run(progress=True)
    out = Path(args.out or f"reports/{cfg['system']['name']}")
    res.save(out)
    m = res.metrics
    print("\n=== Backtest summary ===")
    for k in ("start", "end", "total_return", "cagr", "ann_vol", "sharpe", "sortino", "max_drawdown", "calmar",
              "psr_vs_0", "deflated_sharpe", "avg_gross_leverage", "n_trades", "cost_drag_ann", "breaker_trips",
              "final_equity", "elapsed_sec"):
        v = m.get(k)
        if isinstance(v, (float, np.floating)):
            pct = k in ("total_return", "cagr", "ann_vol", "max_drawdown", "cost_drag_ann")
            print(f"  {k:<20} {v:>14.2%}" if pct else f"  {k:<20} {v:>14.3f}")
        else:
            print(f"  {k:<20} {v!s:>14}")
    print(f"\nreport written to {out}/ (equity_curve.csv, trades.csv, metrics.json, tearsheet.png)")
    return 0


def cmd_trade(args, mode: str) -> int:
    from quantbot.core.live_runner import LiveRunner

    cfg, registry, limits = _setup(args)
    if args.poll:
        cfg["live"]["poll_seconds"] = args.poll
    runner = LiveRunner(cfg, registry, limits, mode=mode, broker=args.broker)
    runner.run(max_iterations=args.iterations)
    return 0


def cmd_download(args) -> int:
    from quantbot.data_engineering.loader import load_frames
    from quantbot.data_engineering.storage import ParquetStore

    cfg, _, _ = _setup(args)
    store = ParquetStore(cfg["data"].get("path", "data/lake"))
    for sym, df in load_frames(cfg).items():
        store.write_bars(sym, df)
        print(f"{sym:<10} {len(df):>6} bars  {df.index[0].date()} -> {df.index[-1].date()}")
    return 0


def cmd_research(args) -> int:
    from quantbot.data_engineering.loader import load_frames
    from quantbot.research.statistical_tests import adf_test, engle_granger, hurst_exponent, ou_fit, variance_ratio

    cfg, _, _ = _setup(args)
    frames = load_frames(cfg)
    print(f"{'symbol':<10}{'ADF p(logP)':>12}{'Hurst':>8}{'VR(5)':>8}{'VR z':>8}")
    for s, df in frames.items():
        lp = np.log(df["close"].to_numpy())
        r = np.diff(lp)
        vr = variance_ratio(r, 5)
        print(f"{s:<10}{adf_test(lp)['p_value']:>12.3f}{hurst_exponent(lp):>8.3f}{vr['vr']:>8.3f}{vr['z']:>8.2f}")
    pairs = (cfg["models"].get("stat_arb") or {}).get("pairs", [])
    if pairs:
        print(f"\n{'pair':<16}{'EG p':>8}{'beta':>8}{'half-life':>11}")
        for y, x in pairs:
            if y not in frames or x not in frames:
                continue
            j = frames[y]["close"].to_frame("y").join(frames[x]["close"].to_frame("x"), how="inner").dropna()
            ly, lx = np.log(j["y"].to_numpy()), np.log(j["x"].to_numpy())
            eg = engle_granger(ly, lx)
            hl = ou_fit(ly - eg["beta"] * lx)["half_life"]
            print(f"{y + '/' + x:<16}{eg['p_value']:>8.3f}{eg['beta']:>8.3f}{hl:>11.1f}")
    return 0


def cmd_train_ml(args) -> int:
    from quantbot.data_engineering.loader import load_frames
    from quantbot.research.ml_pipelines import train_and_register

    cfg, _, _ = _setup(args)
    frames = load_frames(cfg)
    syms = (cfg["models"].get("ml") or {}).get("symbols", list(frames))
    syms = list(frames) if syms == "all" else syms
    info = train_and_register({s: frames[s] for s in syms if s in frames}, name=args.model_name)
    print(json.dumps(info, indent=2, default=str))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="QuantBot - event-driven quant trading system")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--backtest", action="store_true")
    mode.add_argument("--paper", action="store_true")
    mode.add_argument("--live", action="store_true")
    mode.add_argument("--download", action="store_true")
    mode.add_argument("--research", action="store_true")
    mode.add_argument("--train-ml", action="store_true")
    p.add_argument("--config", default="config/strategy_etf.yaml")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--capital", type=float)
    p.add_argument("--out", help="report directory (backtest)")
    p.add_argument("--n-trials", type=int, default=1, help="number of strategy variants tried (for deflated Sharpe)")
    p.add_argument("--broker", default="sim", choices=["sim", "alpaca", "ibkr"])
    p.add_argument("--poll", type=float, help="seconds between polls (paper/live)")
    p.add_argument("--iterations", type=int, help="stop after N polling iterations (paper/live)")
    p.add_argument("--metrics-port", type=int, help="enable Prometheus exporter on this port")
    p.add_argument("--model-name", default="ridge_forecaster")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    if args.backtest:
        return cmd_backtest(args)
    if args.paper:
        return cmd_trade(args, "paper")
    if args.live:
        return cmd_trade(args, "live")
    if args.download:
        return cmd_download(args)
    if args.research:
        return cmd_research(args)
    if args.train_ml:
        return cmd_train_ml(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
