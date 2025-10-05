# crypto-trader

End-to-end crypto strategy playground:

* **Backtests** across (interval × start window × model) with fee/slippage-aware **threshold sweep** and rich P\&L stats.
* **Live bot** that retrains on schedule and calibrates thresholds from recent history with Kelly-capped sizing.
* Modular code: add models/pipelines in one place; add features in one place.

---

## Quick Start

### Install

```bash
python3 -m venv venv
source venv/bin/activate
pip install poetry
poetry install --no-root
```

`.env`
```
# binance creds
BINANCE_API_KEY=
BINANCE_API_SECRET=
BINANCE_API_INTERVAL=1h
TESTNET=False
```

before u start running anything, make sure the system is up to date
```
pytest .
```
you should see all test passed! Do not start trading without all the test showing pass!

### Run the backtest grid

```bash
# all args are optional, or you can adjust in the test_model.py script's default
python test_model.py \
  --symbol BTCUSDT \
  --start-list "120d,365d" \
  --intervals "30m,1h" \
  --models "logreg,hgb,rf" \
  --threshold-sweep "0.50:0.90:0.01" \
  --best-metric total_net_return \
  --fees-bps 10 --slippage-bps 5 \
  --split-mode time --test-size 0.2
```
or more comprehensively:
```
python test_model.py  --symbol BTCUSDT --intervals "4h,8h,1d"   --start-list "120d,365d,720d"   --models "logreg,sgdlog,rf,hgb,linsvc,bilstm,gru_lstm,hybrid_transformer,voting_soft,stacking,metalabel"   --best-metric total_net_return --fees-bps 10 --slippage-bps 5 \
  --split-mode time --test-size 0.2
```

Artifacts land in `backtest_output/metrics/`. Key columns include:

### Running the live bot

```bash
python bot.py
```
---


## Understanding Key Metrics

* **hit\_rate**: fraction of profitable trades.
* **avg\_net\_ret\_per\_bar**: average per-bar net return (after fees/slippage).
* **avg\_net\_ret\_per\_trade**: average return *on bars when in a position*.
* **total\_net\_return**: equity curve final return over test bars.
* **sharpe\_like**: mean/SD of per-bar returns × sqrt(bars/year).
* **cost\_roundtrip**: `2 × (fees_bps + slippage_bps)/10,000`.

> A high probability of being right doesn’t guarantee profit—moves must clear **costs**. 
<br>Use `--label-mode ret_gt_bps --ret-bps <costs>` to train for “up big enough”, not just “up”.

---

## Add a New Model (and use it in backtests)

You only need to touch **`model_manager.py`**.

### 1) Register the name

Extend the union and (optionally) scaling rule:

```python
# model_manager.py
ModelName = Literal["logreg", "sgdlog", "rf", "hgb", "linsvc", "xgb"]  # NEW

def _needs_scaling(model_name: ModelName) -> bool:
    return model_name in {"logreg", "sgdlog", "linsvc"}  # trees/boosters usually False
```

### 2) Build the estimator branch

Wrap non-probabilistic models with **CalibratedClassifierCV** to expose `predict_proba`.

```python
from sklearn.calibration import CalibratedClassifierCV
# optional: from xgboost import XGBClassifier

if self.model_name == "xgb":
    # Example hyperparams; tune for your data
    base = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        learning_rate=0.05,
        random_state=self.random_state,
        tree_method="hist",
        # scale_pos_weight or class_weight if class imbalance
    )
    return base  # already supports predict_proba
```

> If the model lacks native `predict_proba` (e.g., `LinearSVC`), wrap with `CalibratedClassifierCV(..., method="isotonic"|"sigmoid")`.

### 3) (Optional) Custom pre-processing pipeline

Edit `_build_pipeline` to add steps (e.g., PCA):

```python
from sklearn.decomposition import PCA
...
pipe = Pipeline([
  ("prep", prep),
  ("pca", PCA(n_components=20)),
  ("clf", self._build_estimator()),
])
```

### 4) Run it

Backtests automatically pick it up:

```bash
python test_model.py --models "logreg,xgb,hgb"
```

…and the bot uses whatever you configure in `config.py`.

> **Dependencies:** Add any external model lib to `pyproject.toml` (e.g., `xgboost`) and run `poetry install`.

---

## Add a New Feature (in `history_manager.py`)

You’ll edit **one place**: `_recompute_features()` and the `predictor_cols` list.

### 1) Compute the feature

Example: Williams %R and a fast/slow EMA spread.

```python
# inside _recompute_features, after you define close/high/low/open/vol
willr = talib.WILLR(high, low, close, timeperiod=timelag)
ema_fast = talib.EMA(close, timeperiod=12)
ema_slow = talib.EMA(close, timeperiod=26)
ema_spread = (ema_fast - ema_slow) / ema_slow
```

### 2) Add to the features table

```python
feat = pd.DataFrame({
    # ... existing columns ...
    "WILLR": willr,
    "EMA_SPREAD": ema_spread,
}, index=df.index)
```

### 3) Add to `predictor_cols`

```python
self.predictor_cols = [
    # ... existing names ...
    "WILLR",
    "EMA_SPREAD",
]
```

### 4) Handle NaNs and alignment

* All arrays must align with `df.index`.
* Replace inf with NaN and drop or fill early rows with insufficient lookback:

```python
feat = feat.replace([np.inf, -np.inf], np.nan).dropna()
```

* Keep the label aligned: we compute `UP_DOWN` from **next bar** close; ensure the final `feat` has no look-ahead leakage.

### 5) (Important) Scaling and binary patterns

`model_manager.py` currently scales “numeric” feature columns and passes through others. Instead of relying on **position** in the list, consider selecting pattern columns by name:

```python
# Recommended improvement in model_manager.py
pattern_cols = [c for c in self.predictor_cols if c.startswith("PATT_")]
numeric_cols = [c for c in self.predictor_cols if c not in pattern_cols]
```

This makes adding features order-agnostic.

### 6) Verify, then run

* `python test_model.py ...` to ensure the new features flow through.
* If you added many new indicators, consider increasing `timelag` or lookbacks accordingly.

---

## Testing

Run the test suite:

```bash
pytest -q
```

---

## FAQ

**Why is my hit rate high but P\&L negative?**  Costs and small moves. Direction ≠ profit. Train with `ret_gt_bps`, calibrate probabilities, and prefer P\&L-driven objectives.

**How do I run only one combo quickly?**  Use a single start, interval, model, and a shorter `start_str` (e.g., `30d`) while iterating.

**Can I use other assets?**  Yes: `--symbol ETHUSDT` etc. Ensure symbol filters exist and liquidity is sufficient.

## License

Privated under Reagan Chan rrr.chanhiulok@gmail.com
