# crypto-trader

An **end-to-end machine-learning crypto strategy platform** for both **directional classification** and **return regression** models.  
Built for **research, backtesting, and live trading** on Binance.

---

## Overview

This framework lets you:

* **Backtest** combinations of models × intervals × start windows with **fee/slippage-aware threshold sweeps**.
* **Calibrate live trading thresholds** dynamically on recent data.
* **Switch seamlessly** between:
  - **Classification (directional):** predict up/down probabilities.
  - **Regression (expected return):** predict next-bar return in basis points.
* **Live trading bot** retrains automatically and applies **Kelly-capped position sizing**.
* **Add new models and features modularly** — only one file each.

---

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install poetry
poetry install --no-root
```

Before doing anything, verify all tests pass:

```bash
pytest -q
```
You should see **all tests passed** — do NOT start trading if any test fails.

---

## Configuration (`config.py`)

Example minimal setup:

```python
# Binance credentials
api_key = "YOUR_BINANCE_API_KEY"
api_secret = "YOUR_BINANCE_API_SECRET"
is_test_net = False

# Symbol and timeframe
symbol = "BTCUSDT"
trade_interval = "1h"     # e.g. "4h", "8h", "1d"
start_str = "720 days ago UTC"
timelag = 16

# Model and task
model_name = "hgb"        # any from model_manager
model_task = "classify"   # "classify" or "regress"

# Performance calibration
best_metric = "total_net_return"
threshold_sweep = "0.50:0.90:0.01"  # for classify
ret_bps_sweep = "-10:50:2"          # for regress (in bps)

# Trading economics
fees_bps = 10
slippage_bps = 5
retrain_every = 30

# Optional regression behavior
pred_ret_scale_bps = 20.0  # converts predicted return → pseudo-probability
calib_window_bars = 2000
```

---

## Classification vs Regression

| Mode | Target | What it Predicts | Decision Logic | When to Use |
|------|---------|------------------|----------------|--------------|
| **Classification** | 0/1 (`UP_DOWN`) | Probability that next bar closes higher | Go long if p(up) ≥ threshold | Simpler, more robust; good for low data or noisy magnitude |
| **Regression** | Continuous (`FWD_RET_BPS`) | Next bar return in bps | Go long if predicted bps > (fees+slippage) | Captures expected *magnitude*; better for optimizing real P&L |
| **Hybrid idea** | – | Combine classification filter and regression intensity | Trade when both agree | Advanced setups only |
 
If your hit rate is high but P&L is low → switch to **regression**.  
If your regression is unstable → fall back to **classification**.

---

## Backtesting

Run a **multi-interval, multi-model** grid backtest with full sweep:

```bash
python backtest.py \
  --symbol BTCUSDT \
  --intervals "4h,8h,1d" \
  --start-list "120d,365d,720d" \
  --models "logreg,sgdlog,rf,hgb,linsvc,bilstm,gru_lstm,hybrid_transformer,voting_soft,stacking,metalabel" \
  --best-metric total_net_return \
  --fees-bps 10 --slippage-bps 3 \
  --split-mode time --test-size 0.2 \
  --task regress
```

Results land in `backtest_output/metrics/`.

Key outputs:
- `total_net_return` – final equity return after costs.
- `avg_net_ret_per_bar` – mean bar return.
- `avg_net_ret_per_trade` – mean return while in position.
- `sharpe_like` – risk-adjusted per-bar performance.
- `hit_rate` – % of profitable trades.

---

## Running the Live Bot

Start the auto-retraining trading bot:

```bash
python bot.py
```

The bot will:
1. Load historical data.
2. Train the selected model.
3. Calibrate threshold (probability or bps).
4. Continuously trade on bar closes.
5. Retrain after `retrain_every` bars.

You’ll see logs like:
```
[CALIBRATION] threshold=0.645 metric=total_net_return value=0.025 trades=108
Predicted up/down: 0.671/0.329 thr=0.645
Computed qty (kelly-capped): 0.0041
```

---

## Metrics Explained

| Metric | Meaning |
|--------|----------|
| **hit_rate** | % of profitable trades |
| **avg_net_ret_per_bar** | Avg. per-bar return (after fees/slippage) |
| **avg_net_ret_per_trade** | Avg. return during active trades |
| **total_net_return** | Total compounded return across test bars |
| **sharpe_like** | Mean/std × sqrt(bars/year) |
| **cost_roundtrip** | 2 × (fees + slippage)/10,000 |

> A high probability doesn’t mean profit. Calibrate using **net P&L metrics**, not accuracy.

---

## Add a New Model

All changes go into **`model_manager.py`**.

1️⃣ **Register name:**

```python
ModelName = Literal["logreg", "sgdlog", "rf", "hgb", "xgb", "linsvc"]
```

**Add to builder:**

```python
if name == "xgb":
    from xgboost import XGBRegressor
    return XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        subsample=0.8,
        random_state=self.random_state,
        tree_method="hist",
    )
```

**Add scaling rule if needed:**

```python
def _needs_scaling(model_name: ModelName) -> bool:
    return model_name in {"logreg", "sgdlog", "linsvc"}
```

---

## Add a New Feature (HistoryManager)

Edit `_recompute_features()` in `history_manager.py` and append new columns:

```python
# Example: add Williams %R and EMA spread
willr = talib.WILLR(high, low, close, timeperiod=timelag)
ema_fast = talib.EMA(close, timeperiod=12)
ema_slow = talib.EMA(close, timeperiod=26)
ema_spread = (ema_fast - ema_slow) / ema_slow

feat = pd.DataFrame({
    "WILLR": willr,
    "EMA_SPREAD": ema_spread,
}, index=df.index)
```

Then extend the predictors:

```python
self.predictor_cols += ["WILLR", "EMA_SPREAD"]
```

Handle NaNs:
```python
feat = feat.replace([np.inf, -np.inf], np.nan).dropna()
```

---

## Testing

```bash
pytest -q
```

All tests must pass before any live trading.

---

## Choosing Between Models & Tasks

| Goal | Recommended Setup |
|------|--------------------|
| Robust, simple directional bot | `model_task = "classify"`, `model_name = "hgb"` |
| Probabilistic timing + P&L optimization | `model_task = "regress"`, `model_name = "hgb"` |
| Deep learning on sequences | `model_name = "bilstm"` or `"gru_lstm"` |
| Ensemble averaging | `model_name = "voting_soft"` |
| Meta-labeling (entry refinement) | `model_name = "metalabel"` |

---

## Kelly Position Sizing

For classification:
```math
f = 2p - 1
```

For regression:
```math
f = sigmoid(pred_bps / pred_ret_scale_bps) - 0.5
```

Then clipped to a max size (usually 1.0 = full balance).

---

## Safety

* Always test on **TESTNET** first (`is_test_net=True`).
* Ensure all **tests pass**.
* Check **threshold calibration** results before going live.
* Consider **dry-run mode** for first deployment.

More tips:
* Use **shorter starts (e.g. 60d)** for rapid prototyping.
* Increase `timelag` to smooth noisy patterns.
* Use **`--best-metric total_net_return`** for trading-style evaluation.
* Avoid overfitting thresholds; recalibrate frequently.
* Backtest intervals separately; don’t mix 1h/4h/1d into one model.

---

## File Summary

| File | Purpose |
|------|----------|
| `managers/history_manager.py` | Build features and labels |
| `managers/model_manager.py` | Define model pipelines |
| `managers/position_manager.py` | Position sizing and execution |
| `evaluations/evaluator.py` | Threshold sweeps and performance metrics |
| `backtest.py` | Backtest grid runner |
| `bot.py` | Live trading loop |

---

## License

Private under **Reagan Chan (rrr.chanhiulok@gmail.com)**  
For personal and research use only.

---

## TLDR Quick Start

```bash
# Run backtest
python backtest.py --symbol BTCUSDT --intervals "1h,4h,8h,1d" --start-list "120d,365d,720d" --models "logreg,rf,hgb" --best-metric total_net_return

# Run live bot (classification)
python bot.py

# Switch to regression mode
# config.py: model_task = "regress"
python bot.py
```
