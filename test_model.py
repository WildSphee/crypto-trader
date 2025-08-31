# test_models.py
# ------------------------------------------------------------
# Grid validator across intervals × start_strs × models.
# - time-based or random split
# - optional label mode: 'direction' or 'ret_gt_bps'
# - threshold P&L sweep (fees/slippage)
# - CSV exports
# ------------------------------------------------------------

import argparse
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from binance.client import Client
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import config
from history_manager import HistoryManager
from model_manager import ModelManager

# ---------- helpers ----------


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
    a, b, c = map(float, arg.split(":"))
    if not (0.0 <= a < b <= 1.0) or c <= 0:
        raise ValueError(
            "threshold sweep must be 'start:stop:step' with 0<=start<stop<=1 and step>0"
        )
    return a, b, c


def parse_start_list(arg: str) -> List[str]:
    # allow "365d,720d,2021-01-01,90 days ago UTC"
    out = []
    for token in [s.strip() for s in arg.split(",") if s.strip()]:
        if token.lower().endswith("d") and token[:-1].isdigit():
            out.append(f"{int(token[:-1])} days ago UTC")
        else:
            out.append(token)
    return out


def bars_per_year_from_interval(interval_code: str) -> float:
    minutes_map = {
        "1m": 1,
        "3m": 3,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "2h": 120,
        "4h": 240,
        "6h": 360,
        "8h": 480,
        "12h": 720,
        "1d": 24 * 60,
    }
    m = minutes_map[interval_code]
    return (365.0 * 24.0 * 60.0) / m


def one_bar_pnl(
    signal: np.ndarray, fwd_returns: np.ndarray, cost_per_roundtrip: float
) -> np.ndarray:
    gross = signal * fwd_returns
    net = gross - signal * cost_per_roundtrip
    return net


# ---------- core eval ----------


def evaluate_combo(
    symbol: str,
    start_str: str,
    interval_code: str,
    timelag: int,
    model_name: str,
    class_weight: Optional[str],
    split_mode: str,
    test_size: float,
    out_dir: str,
    save_datasets: bool,
    save_predictions: bool,
    threshold_sweep: Optional[Tuple[float, float, float]],
    fees_bps: float,
    slippage_bps: float,
    label_mode: str,
    ret_bps: float,
    client: Client,
) -> Dict:
    b_interval = map_interval(interval_code)
    print(
        f"\n--- Interval {interval_code} / start {start_str} / model {model_name} ---"
    )

    # 1) Build features
    data = HistoryManager(
        client=client,
        symbol=symbol,
        interval=b_interval,
        start_str=start_str,
        timelag=timelag,
    )
    data.load_history()
    X, y_dir = data.dataset()  # default direction label from HistoryManager

    # optional label override: ret_gt_bps
    fwd_ret = (
        data.df_ohlcv["close"]
        .pct_change()
        .shift(-1)
        .reindex(data.df_features.index)
        .values
    )
    if label_mode == "ret_gt_bps":
        thr = ret_bps / 10_000.0
        y = (fwd_ret > thr).astype(int)
        # remove any NaNs alignments just in case
        mask_ok = ~np.isnan(fwd_ret)
        X, y, fwd_ret = X[mask_ok], y[mask_ok], fwd_ret[mask_ok]
    else:
        y = y_dir.values

    n = len(X)
    if n < 200:
        print(f"[WARN] Only {n} rows; results may be noisy.")

    # 2) Fit/Eval
    model = ModelManager(
        predictor_cols=list(X.columns),
        model_name=model_name,
        class_weight=class_weight,
        random_state=1,
    )

    if split_mode == "random":
        # Train internally with random split to get a fair test accuracy
        test_acc = model.train(X, pd.Series(y))
        pipe = model.pipeline
        # For sweep/report, we’ll compute probs on the entire set (reference)
        p_up_all = pipe.predict_proba(X)[:, 1]
        y_pred_all = (p_up_all >= 0.5).astype(int)

        acc = test_acc
        prec = precision_score(y, y_pred_all, zero_division=0)
        rec = recall_score(y, y_pred_all, zero_division=0)
        f1 = f1_score(y, y_pred_all, zero_division=0)
        try:
            auc = roc_auc_score(y, p_up_all)
        except Exception:
            auc = float("nan")
        cm = confusion_matrix(y, y_pred_all).tolist()

        test_mask = np.ones(n, dtype=bool)  # not the true random holdout
        p_up_test, y_test, fwd_test = p_up_all, y, fwd_ret
    else:
        idx_train, idx_test = time_split_indices(n, test_size)
        X_train, y_train = X.iloc[idx_train], y[idx_train]
        X_test, y_test = X.iloc[idx_test], y[idx_test]

        model.pipeline = model._build_pipeline()
        model.pipeline.fit(X_train, y_train)

        p_up_test = model.pipeline.predict_proba(X_test)[:, 1]
        y_pred = (p_up_test >= 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, p_up_test)
        except Exception:
            auc = float("nan")
        cm = confusion_matrix(y_test, y_pred).tolist()

        test_mask = np.zeros(n, dtype=bool)
        test_mask[idx_test] = True
        fwd_test = fwd_ret[idx_test]

    # 3) Save artifacts
    ts_tag = time.strftime("%Y%m%d_%H%M%S")
    tag = f"{symbol}_{interval_code}_{model_name}_{ts_tag}"

    if save_datasets:
        data.df_ohlcv.to_csv(
            os.path.join(
                out_dir, "datasets", f"OHLCV_{start_str.replace(' ', '_')}_{tag}.csv"
            ),
            index=False,
        )
        data.df_features.to_csv(
            os.path.join(
                out_dir, "datasets", f"FEATURES_{start_str.replace(' ', '_')}_{tag}.csv"
            ),
            index=True,
        )

    if save_predictions:
        pred_df = pd.DataFrame(
            {
                "is_test": test_mask.astype(int),
                "p_up": np.where(test_mask, p_up_test, np.nan),
                "y_true": np.where(test_mask, y_test, np.nan),
                "fwd_ret": np.where(test_mask, fwd_test, np.nan),
            }
        )
        pred_df.to_csv(
            os.path.join(
                out_dir, "predictions", f"PRED_{start_str.replace(' ', '_')}_{tag}.csv"
            ),
            index=False,
        )

    cost_rt = 2.0 * ((fees_bps + slippage_bps) / 10_000.0)
    best_metric = "sharpe_like"  # or "total_net_return" / "avg_net_ret_per_bar"

    def _pnl_for_threshold(thr_val: float):
        signal = (p_up_test >= thr_val).astype(int)
        # net returns per test bar (nan-safe stats)
        net = signal * fwd_test - signal * cost_rt

        trades = int(signal.sum())
        hit_rate = float((net[signal == 1] > 0).mean()) if trades > 0 else float("nan")
        avg_net_ret_per_bar   = float(np.nanmean(net)) if trades > 0 else 0.0
        avg_net_ret_per_trade = float(np.nanmean(net[signal == 1])) if trades > 0 else float("nan")

        eq = np.cumprod(1.0 + np.nan_to_num(net, nan=0.0))
        total_ret = float(eq[-1] - 1.0) if len(eq) else 0.0

        mu = np.nanmean(net) if trades > 1 else 0.0
        sd = np.nanstd(net, ddof=1) if trades > 1 else 0.0
        bpy = bars_per_year_from_interval(interval_code)
        sharpe = float((mu / sd) * np.sqrt(bpy)) if sd > 0 else float("nan")

        return {
            "threshold": float(thr_val),
            "trades": trades,
            "hit_rate": hit_rate,
            "avg_net_ret_per_bar": avg_net_ret_per_bar,
            "avg_net_ret_per_trade": avg_net_ret_per_trade,
            "total_net_return": total_ret,
            "sharpe_like": sharpe,
        }

    best = None
    if threshold_sweep:
        a, b, step = threshold_sweep
        thresholds = np.round(np.arange(a, b + 1e-12, step), 6)
        # evaluate & pick best
        candidates = [_pnl_for_threshold(th) for th in thresholds]
        if candidates:
            best = max(candidates, key=lambda r: (-np.inf if np.isnan(r[best_metric]) else r[best_metric]))
    else:
        # no sweep provided → evaluate at 0.50 so summary still has P&L
        best = _pnl_for_threshold(0.50)

    # 5) Summary row (now includes best-threshold P&L)
    return {
        "symbol": symbol,
        "interval": interval_code,
        "start_str": start_str,
        "timelag": timelag,
        "model": model_name,
        "rows": n,
        "split_mode": split_mode,
        "test_size": test_size,
        "class_weight": class_weight or "",
        "label_mode": label_mode,
        "ret_bps": ret_bps if label_mode == "ret_gt_bps" else 0.0,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": float(auc),
        "confusion_matrix": cm,
        # best threshold & money metrics (no separate thresholds file)
        "best_threshold": float(best["threshold"]) if best else 0.50,
        "trades": int(best["trades"]) if best else 0,
        "hit_rate": float(best["hit_rate"]) if best else float("nan"),
        "avg_net_ret_per_bar": float(best["avg_net_ret_per_bar"]) if best else 0.0,
        "avg_net_ret_per_trade": float(best["avg_net_ret_per_trade"]) if best else float("nan"),
        "total_net_return": float(best["total_net_return"]) if best else 0.0,
        "sharpe_like": float(best["sharpe_like"]) if best else float("nan"),
        # for sanity/reference
        "cost_roundtrip": float(cost_rt),
    }


def main():
    p = argparse.ArgumentParser(
        description="Grid-validate models over intervals * start_strs * models."
    )
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument(
        "--start-list",
        default="120d,365d,730d",
        help="Comma list of windows or absolute dates, e.g. '180d,365d,2021-01-01,90 days ago UTC'.",
    )
    p.add_argument(
        "--intervals",
        default="30m,1h,2h,4h",
        help="Comma list, e.g. '5m,15m,1h,4h,1d'.",
    )
    p.add_argument(
        "--models",
        default="logreg,hgb,rf,hgb,linsvc,sgdlog",
        help="Comma list: logreg,sgdlog,rf,hgb,linsvc",
    )
    p.add_argument("--timelag", type=int, default=20)
    p.add_argument("--split-mode", choices=["time", "random"], default="time")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--class-weight", choices=["balanced", "none"], default="none")
    p.add_argument(
        "--label-mode",
        choices=["direction", "ret_gt_bps"],
        default="direction",
        help="Use 'ret_gt_bps' to define UP only when next-bar return > --ret-bps.",
    )
    p.add_argument(
        "--ret-bps",
        type=float,
        default=0.0,
        help="Return threshold in bps for 'ret_gt_bps' mode.",
    )
    p.add_argument("--out-dir", default="backtest_output")
    p.add_argument("--save-datasets", action="store_true")
    p.add_argument("--save-predictions", action="store_true")
    p.add_argument("--threshold-sweep", default="0.50:0.75:0.01", help="start:stop:step for p_up threshold, e.g. '0.50:0.75:0.01'")
    p.add_argument("--fees-bps", type=float, default=10.0)
    p.add_argument("--slippage-bps", type=float, default=5)
    args = p.parse_args()

    start_list = parse_start_list(args.start_list)
    intervals = [s.strip() for s in args.intervals.split(",") if s.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    thr = parse_sweep(args.threshold_sweep) if args.threshold_sweep else None
    class_weight = None if args.class_weight == "none" else "balanced"

    ensure_dirs(args.out_dir)

    # IMPORTANT: use MAINNET for data
    client = Client(
        getattr(config, "api_key", None),
        getattr(config, "api_secret", None),
        testnet=False,
    )

    results = []
    for start_str in start_list:
        for interval_code in intervals:
            for model_name in models:
                try:
                    res = evaluate_combo(
                        symbol=args.symbol,
                        start_str=start_str,
                        interval_code=interval_code,
                        timelag=args.timelag,
                        model_name=model_name,
                        class_weight=class_weight,
                        split_mode=args.split_mode,
                        test_size=args.test_size,
                        out_dir=args.out_dir,
                        save_datasets=args.save_datasets,
                        save_predictions=args.save_predictions,
                        threshold_sweep=thr,
                        fees_bps=args.fees_bps,
                        slippage_bps=args.slippage_bps,
                        label_mode=args.label_mode,
                        ret_bps=args.ret_bps,
                        client=client,
                    )
                    results.append(res)
                except Exception as e:
                    print(
                        f"[ERROR] start={start_str} interval={interval_code} model={model_name}: {e}"
                    )

    if results:
        df = pd.DataFrame(results)
        ts_tag = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(
            args.out_dir, "metrics", f"SUMMARY_{args.symbol}_{ts_tag}.csv"
        )
        df.to_csv(out_path, index=False)
        print("\nSummary written to:", out_path)
        print(
            df[
                [
                    "start_str",
                    "interval",
                    "model",
                    "rows",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "auc",
                    "thresholds_file",
                ]
            ]
        )
    else:
        print("No successful evaluations.")


if __name__ == "__main__":
    main()
