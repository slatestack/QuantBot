# QuantBot

An event-driven quantitative trading system for **equities and commodity futures**, implementing the
*Quantitative Trading System Architecture & Codebase Blueprint*. A single production engine
(`quantbot/core/engine.py`) runs unchanged in **backtest**, **paper** and **live** modes.

> **Read this first.** No trading system can guarantee profits. The results below are historical
> simulations with modelled costs. They are not a forecast. Paper-trade for months before risking
> capital, start small, and keep the kill switch on.

## Results (as of 2026-11-19)

Cross-asset ETF book (`config/strategy_etf.yaml`). Real daily data for 2008-01 to 2026-11.
$1M start, trading begins after a 260-bar warm-up. Costs are included: commission,
half-spread, square-root market impact and short-borrow fees.

| Metric | Value |
|---|---|
| CAGR | **4.7 %** |
| Annualised vol | 7.6 % |
| Sharpe | **0.64** |
| Max drawdown | **−16.5 %** |
| Deflated Sharpe (8 variants tried) | 0.90 |
| Cost drag | 0.62 %/yr |
| Kill-switch trips (−2.5 % day) | 4 (2018-02, 2020-03, 2021-11, 2022-11) |

Evidence from development:

* **Trend-following** produces most of the return when run alone (Sharpe 0.48).
* **Stat-arb** lost money on XOM/CVX, KO/PEP and GLD/SLV (Sharpe −0.34). None of these
  pairs passes a full-sample Engle-Granger test. A rolling cointegration gate now
  blocks most trades on them, but the sleeve is still slightly negative.
* **ML forecaster:** the out-of-sample IC under purged K-fold is about 0. Price-only
  features don't carry a reliable daily edge on these ETFs, so ML has a small weight
  and its forecast is smoothed to cut turnover.
* **Performance has been roughly flat since 2021.**

The synthetic config (`config/strategy_synthetic.yaml`) shows a Sharpe of about 2.2. **Ignore that
as evidence of edge.** The generator plants trends, carry and a cointegrated pair on purpose, so that
run only shows the machinery can capture structure when structure exists.

## Blueprint → code map

| Blueprint layer | Module |
|---|---|
| Events / bus / clocks | `quantbot/core/events`, `core/bus` (deterministic in-memory bus, Redis Streams bridge), `core/clock` |
| Ledger | `core/state/portfolio.py` (equities cash-settled; futures daily variation margin), `core/state/history.py` (hot ring buffer) |
| Instrument abstraction | `core/instruments.py`, `config/instruments/*.yaml` |
| Data ingestion | `data_engineering/connectors` (Yahoo, Alpaca, CSV/Parquet, synthetic), `normalizers/schemas.py` (`ts_event`/`ts_recv`), `storage` (Parquet lake, QuestDB ILP) |
| Futures roll | `data_engineering/futures_roller.py`: Panama/ratio stitching, roll before FND, roll-aware returns, `RollManager` for live routing |
| Features | `quant_models/features`: Yang-Zhang, Parkinson, Garman-Klass, OBI, OFI, VPIN, Kyle λ, Roll / Corwin-Schultz spreads, fractional differencing |
| Alpha | `quant_models/alpha`: Kalman-filter pairs with OU and rolling-cointegration gates; commodity roll-yield carry and seasonality; Baz et al. multi-horizon trend with breakout filter; walk-forward ML with ONNX Runtime inference |
| Aggregation | `SignalAggregator`: model weights, stale-signal decay, diversification multiplier |
| Portfolio | `quant_models/portfolio`: Ledoit-Wolf, Marcenko-Pastur denoising, HRP, fractional-Kelly MVO, vol targeting with a Kelly cap |
| Risk | `risk_management`: pre-trade fat-finger, collar, concentration, leverage and margin checks; VaR/CVaR and drawdown monitor; circuit breaker (cancel → flatten → lock) with staleness and disconnect pauses |
| Execution | `execution/oms.py` (order state machine and reconciliation), `ems.py`, `algos` (TWAP, VWAP, Almgren-Chriss), `router.py`, `gateways` (simulated, Alpaca, IBKR) |
| Simulation | `simulation/backtest_engine.py`, `slippage_models.py` (spread and square-root impact), `margin_engine.py`, `performance.py` (PSR, Deflated Sharpe) |
| Research | `research/statistical_tests` (ADF, Engle-Granger, OU, Hurst, variance ratio), `research/ml_pipelines` (triple-barrier labels, purged K-fold, CPCV, permutation importance, ONNX model registry), `research/notebooks` |
| Monitoring | `monitoring/metrics_exporter.py` (Prometheus, no dependencies), Grafana dashboard, `docker-compose.yml` |

### How lookahead is prevented
Bars are only visible once delivered, through `HistoryBuffer`. Orders created at bar *t* can only
fill on a later bar, at its open plus costs. Fills are applied before the next decision step. ML
labels join the training set only once their horizon has fully elapsed. `tests/test_lookahead_bias.py`
scrambles all data after a cutoff and asserts that every earlier decision and equity value is
identical.

## Quick start

```bash
pip install -r requirements.txt
```

Backtest on real ETF data (downloads and caches it into `data/cache`):

```bash
python main.py --backtest --config config/strategy_etf.yaml --n-trials 8
```

Backtest on the synthetic multi-asset universe, including commodity futures curves:

```bash
python main.py --backtest --config config/strategy_synthetic.yaml
```

Run the research diagnostics (stationarity, Hurst exponent, pair cointegration):

```bash
python main.py --research --config config/strategy_etf.yaml
```

Train the offline ML model with purged and combinatorial CV, exported to `models/*.onnx`:

```bash
python main.py --train-ml --config config/strategy_etf.yaml
```

Paper-trade with live data against the local simulated broker, with Prometheus metrics on :9108:

```bash
python main.py --paper --config config/strategy_etf.yaml --broker sim --metrics-port 9108
```

Paper-trade against an Alpaca paper account (keys stay in the environment, never in config):

```bash
export APCA_API_KEY_ID=... APCA_API_SECRET_KEY=...
python main.py --paper --config config/strategy_etf.yaml --broker alpaca
```

Run the full stack (engine, Redis, QuestDB, Prometheus, Grafana on :3000):

```bash
docker compose up --build
```

Run the tests:

```bash
python -m pytest
```

Live trading needs a real broker **and** an explicit opt-in:

```bash
QUANTBOT_LIVE_CONFIRM=YES python main.py --live --config config/strategy_etf.yaml --broker alpaca
```

Each backtest writes `reports/<name>/`: `equity_curve.csv`, `trades.csv`, `target_weights.csv`,
`risk_events.csv`, `implementation_shortfall.csv`, `metrics.json` and `tearsheet.png`.

## Status

| Component | Status |
|---|---|
| Backtester, models, risk, OMS/EMS, simulated broker | Covered by 33 tests (`tests/`) |
| Paper loop with live Yahoo data and the simulated broker | Run end-to-end |
| Alpaca gateway | Written against the REST v2 API. **Not yet run against a real account.** Validate in paper first. |
| IBKR gateway (`ib_async`) | Written. **Not yet run against TWS or IB Gateway.** Validate on the paper port 7497. |
| Commodity futures on real data | Needs a futures data source, such as IBKR historical data. Yahoo's `CL=F`-style series aren't roll-adjusted, so the real-data config uses commodity ETFs instead. |

Risk limits are in `config/risk_limits.yaml`: a −2.5 % daily-loss kill switch, a −20 % maximum
drawdown, 3× gross leverage, a 40 % position cap and a 99 % VaR limit. In backtests the breaker
automatically "resets" after 5 bars to stand in for a human. In live trading it stays locked until
`CircuitBreaker.reset()` is called.
