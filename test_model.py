# test_models.py
# ------------------------------------------------------------
# Validator for your OOP stack with:
# - multi-interval eval
# - time/random split
# - threshold sweep + one-bar P&L (net of fees/slippage)
# - CSV exports (datasets, predictions, threshold results, summary)
# ------------------------------------------------------------

import os
import time
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from binance.client import Client

import config
from history_manager import HistoryManager, INTERVAL_TO_MS
from model_manager import ModelManager  # safe: bot.run() is guarded by __main__


# --- mapping for shorthand intervals ---
def map_interval(code: str) -> str:
    from binance.client import Client
    m = {
        "1m": Client.KLINE_INTERVAL_1MINUTE,
        "3m": Client.KLINE_INTERVAL_3MINUTE,
        "5m": Client.KLINE_INTERVAL_5MINUTE,
        "15m": Client.KLINE_INTERVAL_15MINUTE,
        "30m": Client.KLINE_INTERVAL_30MINUTE,
        "1h": Client.KLINE_INTERVAL_1HOUR,
        "2h": Client.KLINE_INTERVAL_2HOUR,
        "4h": Client.KLINE_INTERVAL_4HOUR,
        "6h": Client.KLINE_INTERVAL_6HOUR,
        "8h": Client.KLINE_INTERVAL_8HOUR,
        "12h": Client.KLINE_INTERVAL_12HOUR,
        "1d": Client.KLINE_INTERVAL_1DAY,
    }
    if code not in m:
        raise ValueError(f"Unsupported interval code: {code}")
    return m[code]


def ensure_dirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    for sub in ["datasets", "metrics", "predictions", "thresholds"]:
        os.makedirs(os.path.join(path, sub), exist_ok=True)


def time_split_indices(n_rows: int, test_frac: float) -> Tuple[np.ndarray, np.ndarray]:
    test_len = max(1, int(n_rows * test_frac))
    train_len = max(1, n_rows - test_len)
    idx = np.arange(n_rows)
    return idx[:train_len], idx[train_len:]


def parse_sweep(arg: str) -> Tuple[float, float, float]:
    # "0.5:0.7:0.01" -> (0.5, 0.7, 0.01)
    parts = [p.strip() for p in arg.split(":")]
    if len(parts) != 3:
        raise ValueError("threshold sweep must be 'start:stop:step', e.g. 0.5:0.7:0.02")
    a, b, c = map(float, parts)
    if not (0.0 <= a < b <= 1.0) or c <= 0:
        raise ValueError("invalid threshold sweep bounds")
    return a, b, c


@dataclass
class EvalConfig:
    symbol: str
    start_str: str
    intervals: List[str]               # ["5m","15m","1h",...]
    timelag: int = 20
    split_mode: str = "time"           # "time" or "random"
    test_size: float = 0.2
    out_dir: str = "backtest_output"
    save_datasets: bool = True
    save_predictions: bool = True
    threshold_sweep: Tuple[float, float, float] | None = None  # (start, stop, step)
    fees_bps: float = 10.0             # taker fee per SIDE in bps (0.1% => 10 bps)
    slippage_bps: float = 0.0          # extra per SIDE in bps
    # strategy: at each bar close, if p_up >= thr, go long for ONE bar, else flat


def one_bar_pnl(signal: np.ndarray, fwd_returns: np.ndarray, cost_per_roundtrip: float) -> np.ndarray:
    """
    signal: 0/1 array aligned to test rows; 1 => be long next bar
    fwd_returns: array of close_{t+1}/close_t - 1 aligned to test rows
    cost_per_roundtrip: total cost (both sides) as decimal (e.g., 0.002 = 20 bps)
    """
    gross = signal * fwd_returns
    net = gross - signal * cost_per_roundtrip
    return net


def bars_per_year_from_interval(interval_code: str) -> float:
    # crude but fine for scaling Sharpe-ish stats
    minutes_map = {
        "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480, "12h": 720,
        "1d": 24*60
    }
    m = minutes_map[interval_code]
    return (365.0 * 24.0 * 60.0) / m


def evaluate_one_interval(cfg: EvalConfig, client: Client, interval_code: str) -> Dict:
    b_interval = map_interval(interval_code)
    print(f"\n--- Interval {interval_code} / start {cfg.start_str} ---")

    # 1) Build data/features
    data = HistoryManager(
        client=client, symbol=cfg.symbol, interval=b_interval,
        start_str=cfg.start_str, timelag=cfg.timelag
    )
    data.load_history()
    X, y = data.dataset()
    n = len(X)

    if n < 200:
        print(f"[WARN] Only {n} rows; results may be noisy.")

    # Forward returns aligned to features rows (close_{t+1}/close_t - 1)
    # df_features indexes align with df_ohlcv rows except last row dropped
    fwd_ret = data.df_ohlcv["close"].pct_change().shift(-1)
    fwd_ret = fwd_ret.reindex(data.df_features.index).values  # numpy array aligned to X

    # 2) Fit/Eval
    model = ModelManager(predictor_cols=data.predictor_cols)

    if cfg.split_mode == "random":
        # random split inside ModelManager for test accuracy
        test_acc = model.train(X, y)
        pipe = model.pipeline
        # For reporting & sweeps we’ll get probs for all rows
        y_prob_all = pipe.predict_proba(X)[:, 1]
        y_pred_all = (y_prob_all >= 0.5).astype(int)
        # "global" metrics on all rows (reference only)
        acc = test_acc
        prec = precision_score(y, y_pred_all, zero_division=0)
        rec = recall_score(y, y_pred_all, zero_division=0)
        f1 = f1_score(y, y_pred_all, zero_division=0)
        try:
            auc = roc_auc_score(y, y_prob_all)
        except Exception:
            auc = float("nan")
        cm = confusion_matrix(y, y_pred_all).tolist()
        test_mask = np.ones_like(y, dtype=bool)  # NOTE: not the true random holdout
        p_up_test = y_prob_all
        y_test = y.values
        fwd_test = fwd_ret
    else:
        # time split: last fraction is the test set
        idx_train, idx_test = time_split_indices(n, cfg.test_size)
        X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]
        X_test,  y_test  = X.iloc[idx_test],  y.iloc[idx_test]

        model.pipeline = model._build_pipeline()
        model.pipeline.fit(X_train, y_train)

        p_up_test = model.pipeline.predict_proba(X_test)[:, 1]
        y_pred = (p_up_test >= 0.5).astype(int)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, p_up_test)
        except Exception:
            auc = float("nan")
        cm = confusion_matrix(y_test, y_pred).tolist()

        test_mask = np.zeros(n, dtype=bool)
        test_mask[idx_test] = True
        fwd_test = fwd_ret[idx_test]

    # 3) Save datasets/preds
    ts_tag = time.strftime("%Y%m%d_%H%M%S")
    base = f"{cfg.symbol}_{interval_code}_{ts_tag}"

    if cfg.save_datasets:
        data.df_ohlcv.to_csv(os.path.join(cfg.out_dir, "datasets", f"OHLCV_{base}.csv"), index=False)
        data.df_features.to_csv(os.path.join(cfg.out_dir, "datasets", f"FEATURES_{base}.csv"), index=True)

    if cfg.save_predictions:
        pred_df = pd.DataFrame({
            "is_test": test_mask.astype(int),
            "p_up": np.where(test_mask, p_up_test, np.nan),
            "y_true": np.where(test_mask, y_test, np.nan),
            "fwd_ret": np.where(test_mask, fwd_test, np.nan),
        })
        pred_df.to_csv(os.path.join(cfg.out_dir, "predictions", f"PRED_{base}.csv"), index=False)

    # 4) Threshold sweep with one-bar P&L
    thr_results_path = ""
    if cfg.threshold_sweep is not None:
        a, b, step = cfg.threshold_sweep
        thresholds = np.round(np.arange(a, b + 1e-12, step), 6)
        cost_rt = 2.0 * ((cfg.fees_bps + cfg.slippage_bps) / 10_000.0)  # roundtrip as decimal

        rows = []
        bpy = bars_per_year_from_interval(interval_code)

        for thr in thresholds:
            signal = (p_up_test >= thr).astype(int)
            net = one_bar_pnl(signal, fwd_test, cost_rt)

            trades = int(signal.sum())
            hit_rate = float((net[signal == 1] > 0).mean()) if trades > 0 else np.nan
            avg_gross = float((signal * fwd_test).mean()) if trades > 0 else 0.0
            avg_net = float(net.mean()) if trades > 0 else 0.0
            # equity curve on test subset only
            eq = np.cumprod(1.0 + np.nan_to_num(net, nan=0.0))
            total_ret = float(eq[-1] - 1.0) if len(eq) else 0.0
            # simple “daily” Sharpe-ish scaled by bars/year (no risk-free, using mean/std per bar)
            mu = np.nanmean(net) if trades > 1 else 0.0
            sd = np.nanstd(net, ddof=1) if trades > 1 else 0.0
            sharpe = float((mu / sd) * np.sqrt(bpy)) if sd > 0 else np.nan

            rows.append({
                "threshold": thr,
                "trades": trades,
                "hit_rate": hit_rate,
                "avg_gross_ret_per_bar": avg_gross,
                "avg_net_ret_per_bar": avg_net,
                "total_net_return": total_ret,
                "sharpe_like": sharpe,
            })

        thr_df = pd.DataFrame(rows)
        thr_results_path = os.path.join(cfg.out_dir, "thresholds", f"THRESH_{base}.csv")
        thr_df.to_csv(thr_results_path, index=False)

    # 5) Return summary
    return {
        "symbol": cfg.symbol,
        "interval": interval_code,
        "rows": n,
        "split_mode": cfg.split_mode,
        "test_size": cfg.test_size,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": float(auc),
        "confusion_matrix": cm,
        "predictions_file": f"PRED_{base}.csv" if cfg.save_predictions else "",
        "thresholds_file": os.path.basename(thr_results_path) if thr_results_path else "",
    }


def main():
    p = argparse.ArgumentParser(description="Validate models across intervals, with threshold sweep & CSV outputs.")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--start-str", default="720 days ago UTC",
                   help="e.g., '365 days ago UTC' or '2021-01-01' (longer => more training data).")
    p.add_argument("--intervals", default="5m,15m,1h,4h,1d",
                   help="Comma list: e.g., '5m,15m,1h,4h,1d'.")
    p.add_argument("--timelag", type=int, default=20)
    p.add_argument("--split-mode", choices=["time","random"], default="time")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--out-dir", default="backtest_output")
    p.add_argument("--save-datasets", action="store_true")
    p.add_argument("--save-predictions", action="store_true")
    p.add_argument("--threshold-sweep", default=None,
                   help="Format 'start:stop:step', e.g. '0.5:0.7:0.02'")
    p.add_argument("--fees-bps", type=float, default=10.0,
                   help="Taker fee per SIDE in bps (0.1%% => 10).")
    p.add_argument("--slippage-bps", type=float, default=0.0,
                   help="Extra per SIDE in bps for slippage/spread.")
    args = p.parse_args()

    thr = parse_sweep(args.threshold_sweep) if args.threshold_sweep else None
    cfg = EvalConfig(
        symbol=args.symbol,
        start_str=args.start_str,
        intervals=[s.strip() for s in args.intervals.split(",") if s.strip()],
        timelag=args.timelag,
        split_mode=args.split_mode,
        test_size=args.test_size,
        out_dir=args.out_dir,
        save_datasets=args.save_datasets,
        save_predictions=args.save_predictions,
        threshold_sweep=thr,
        fees_bps=args.fees_bps,
        slippage_bps=args.slippage_bps,
    )

    ensure_dirs(cfg.out_dir)
    client = Client(getattr(config, "api_key", None),
                    getattr(config, "api_secret", None),
                    testnet=getattr(config, "is_test_net", False))

    results = []
    for code in cfg.intervals:
        try:
            res = evaluate_one_interval(cfg, client, code)
            results.append(res)
        except Exception as e:
            print(f"[ERROR] {code}: {e}")

    if results:
        df = pd.DataFrame(results)
        ts_tag = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(cfg.out_dir, "metrics", f"SUMMARY_{cfg.symbol}_{ts_tag}.csv")
        df.to_csv(out_path, index=False)
        print("\nSummary written to:", out_path)
        print(df[["interval","rows","split_mode","accuracy","precision","recall","f1","auc","thresholds_file"]])
    else:
        print("No successful evaluations.")


if __name__ == "__main__":
    main()
