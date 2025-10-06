## The repo consist on two main portions:
1. project goal (Backtesting)
    1. script is backtest.py
    1. please ask chatgpt to interpret the script if unclear
    1. The script is a CLI script that test all iterations of settings (time frame, time interval, and models)
1. trading bot
    1. script is bot.py
    1. running this would start a live trade session

## Interpretation of results:

1. To see the details of the backtest is in: `backtest_output/metrics/SUMMARY_BTCUSDT_20251005_125402.csv` and `backtest_output/metrics/SUMMARY_BTCUSDT_20251005_054449.csv`
    1. the reason there are two files is that each of them are testing different things, the former is `REGRESSION`, the latter is `CLASSIFICATION` - these concepts are very important in understanding the data
    1. `Classification`
        1. Predicts a category/label (e.g., up vs down, spam/not spam, class A/B/C).
        1. Output is often a probability per class; final label comes from a threshold (e.g., p≥0.6 ⇒ “up”).
        1. Choose `classification` when:
            1. The decision is inherently discrete (buy/skip, approve/deny).
            1. You care about hit rate, precision/recall, or meeting a probability threshold.
            1. Your action is triggered by event occurrence (e.g., “return > 10 bps”).
    1. `Regression`
        1. Predicts a number (a continuous value), e.g., next-bar return in bps, price, demand, time.
        1. Choose `regression` when:
            1. You need the magnitude or direction + size (e.g., +15 bps vs −5 bps).
            1. You want to rank opportunities by expected value (sort by ŷ).
            1. You’ll convert ŷ into actions with a return threshold minus costs.
    
1. Quick rule of thumb
    1. If your action is “Do I act or not?” → start with classification.
    1. If your action is “How much edge do I have?” (and you’ll sort/pick sizes) → use regression.

In our repo, for project goal (UP/DOWN Prediction), we shd look at classification, BUT for practical trading, we shd look at regression, we need to narrate this idea very clearly!

1. here’s a plain-English cheatsheet for every column in your results. I’ll note when a field only applies to `classification` or `regression`, the units, and what the number represents
    1. symbol — market pair you evaluated (e.g., BTCUSDT).
    1. interval — bar timeframe used (e.g., 1d, 8h).
    1. start_str — where the history window starts (e.g., 365 days ago UTC).
    1. timelag — lookback length used to build features / sequences (e.g., 16 bars).
    1. model — model used (classifier name or mapped regressor, e.g., hgb_reg, rf_reg).
    1. rows — number of samples actually used for training/eval after any windowing.
    1. split_mode — "time" (train first chunk, test last chunk) or "random" (shuffle split).
    1. test_size — fraction of rows held out for testing (e.g., 0.2).
    1. class_weight — classifier class weighting ("" or "balanced"). (`classification`)
    1. task — "classify" or "regress".
    1. label_mode — how targets were formed:
    1. "direction" = next bar up/down class. (`classification`)
    1. "ret_gt_bps" = class is 1 if next return > ret_bps. (`classification`)
    1. "return_bps" = continuous next-bar return in bps. (`regression`)
    1. ret_bps (bps) — threshold used when label_mode="ret_gt_bps". Else 0.0.
    1. Predictive quality (standard ML metrics)
    1. (Only meaningful for `classification` unless noted)
    1. accuracy — share of correct class predictions.
    1. precision — of predicted ups, how many were actually up.
    1. recall — of actual ups, how many you caught.
    1. f1 — harmonic mean of precision & recall.
    1. auc — ROC AUC using predicted “up” probabilities.
    1. confusion_matrix — [[TN, FP],[FN, TP]] for the test set.
    1. r2 — coefficient of determination for predicted returns. (`regression`)
    1. mae_bps (bps) — mean absolute error of predicted vs. true next-bar return. (`regression`)
    1. mape_pct (%) — mean absolute percentage error (safe/robust variant). (`regression`)

---

1. we want to select the best model, here are the judgement criterias we used to filter down till the best model remaining:
    1. Collect runs: Gather all model results across intervals and time windows.
    1. Sanity check: Keep datasets with at least 200 rows and valid metrics.
    1. Profit filter: Require positive total return and Sharpe > 0.3.
    1. Quality filter: Drop regressors with R² ≤ 0 and classifiers with AUC < 0.52.
    1. Stability check: Keep models showing consistent Sharpe > 0 across runs.
    1. Activity filter: Require at least 8 trades per backtest.
    1. Cost check: Ensure returns exceed 1.2× trading cost.
    1. Scoring: Rank models by weighted Sharpe, return, and stability.
    1. Select winners: Pick top 3 and most stable models per interval.


1. final results are here: `backtest_output/metrics/SHORTLIST_stable.csv`
    1. the two models results are explained in slide (Results (Trading bot))

