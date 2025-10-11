import argparse
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from binance.client import Client
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    mean_absolute_error,
    log_loss,
    mean_squared_error,
)

import config
from managers.history_manager import HistoryManager
from managers.model_manager import ModelManager

from evaluations.metrics import SweepConfig, safe_mape_pct
from evaluations.evaluator import choose_best_threshold_for_window

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


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


def make_windows_from_df(
    df: pd.DataFrame, feat_cols: list[str], window: int, stride: int = 1
) -> np.ndarray:
    arr = df[feat_cols].to_numpy(dtype=np.float32, copy=False)
    if len(arr) < window:
        return np.empty((0, window, arr.shape[1]), dtype=np.float32)
    sw = sliding_window_view(arr, window_shape=window, axis=0)
    return sw[::stride]


def ensure_dirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    for sub in ["metrics", "datasets", "predictions"]:
        os.makedirs(os.path.join(path, sub), exist_ok=True)


def time_split_indices(n_rows: int, test_frac: float) -> Tuple[np.ndarray, np.ndarray]:
    test_len = max(1, int(n_rows * test_frac))
    train_len = max(1, n_rows - test_len)
    idx = np.arange(n_rows)
    return idx[:train_len], idx[train_len:]


def parse_sweep(arg: str) -> SweepConfig:
    a, b, c = map(float, arg.split(":"))
    if not (0.0 <= a < b <= 1.0) or c <= 0:
        raise ValueError(
            "threshold sweep must be 'start:stop:step' with 0<=start<stop<=1 and step>0"
        )
    return SweepConfig(start=a, stop=b, step=c)


def parse_start_list(arg: str) -> List[str]:
    out = []
    for token in [s.strip() for s in arg.split(",") if s.strip()]:
        if token.lower().endswith("d") and token[:-1].isdigit():
            out.append(f"{int(token[:-1])} days ago UTC")
        else:
            out.append(token)
    return out


def sweep_on_predicted_return(
    yhat_bps: np.ndarray,
    fwd_ret_bps: np.ndarray,
    cost_bps: float,
    best_metric: str,
    thr_grid_bps: np.ndarray | None = None,
) -> Dict:
    """
    Long-only: trade when predicted return >= (threshold + cost).
    Metrics are net of costs (bps).
    """
    if thr_grid_bps is None:
        thr_grid_bps = np.arange(0.0, 60.1, 1.0)  # 0..60 bps

    best = {"threshold": 0.0, "total_net_return": float("-inf")}
    for thr in thr_grid_bps:
        take = yhat_bps >= (thr + cost_bps)
        trades = int(take.sum())
        if trades == 0:
            continue
        net = fwd_ret_bps[take] - cost_bps
        total = float(np.nansum(net))
        avg = float(np.nanmean(net))
        std = float(np.nanstd(net))
        sharpe_like = float(avg / (std + 1e-9))

        # Sortino-like: mean / downside std (only negative net bars)
        downside = net[net < 0.0]
        downside_std = float(np.nanstd(downside)) if downside.size > 0 else np.nan
        sortino_like = float(avg / (downside_std + 1e-9)) if not np.isnan(downside_std) else float("nan")

        metric = {
            "total_net_return": total,
            "avg_net_ret_per_bar": avg,
            "sharpe_like": sharpe_like,
            "sortino_like": sortino_like,
        }.get(best_metric, sharpe_like)

        if metric > best.get(best_metric, float("-inf")):
            best = {
                "threshold": float(thr),
                "trades": trades,
                "total_net_return": total,
                "avg_net_ret_per_bar": avg,
                "sharpe_like": sharpe_like,
                "sortino_like": sortino_like,
            }
    if best["total_net_return"] == float("-inf"):
        best.update(
            {
                "threshold": 0.0,
                "trades": 0,
                "total_net_return": 0.0,
                "avg_net_ret_per_bar": float("nan"),
                "sharpe_like": float("nan"),
                "sortino_like": float("nan"),
            }
        )
    return best

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
    sweep_cfg: SweepConfig,
    fees_bps: float,
    slippage_bps: float,
    label_mode: str,
    ret_bps: float,
    best_metric: str,
    client: Client,
    task: str,
) -> Dict:
    b_interval = map_interval(interval_code)
    print(
        f"\n--- Interval {interval_code} / start {start_str} / model {model_name} / task {task} ---"
    )

    data = HistoryManager(
        client=client,
        symbol=symbol,
        interval=b_interval,
        start_str=start_str,
        timelag=timelag,
        # HistoryManager defaults include FNG/on-chain already enabled
    )
    data.load_history()

    # ================= REGRESSION PATH =================
    if task == "regress":
        # Predict next-bar return in bps
        X, y_ret = data.dataset(target="return_bps")

        # map common classifier names to sensible regressors if provided
        reg_name_map = {
            "hgb": "hgb_reg",
            "rf": "rf_reg",
            "logreg": "linreg",
            "sgdlog": "linreg",
            "linsvc": "svr",
            "voting_soft": "hgb_reg",
            "stacking": "hgb_reg",
            "metalabel": "hgb_reg",
            "hgb_reg": "hgb_reg",
            "rf_reg": "rf_reg",
            "linreg": "linreg",
            "svr": "svr",
            "arima": "arima",
        }
        reg_model_name = reg_name_map.get(model_name, "hgb_reg")

        model = ModelManager(
            predictor_cols=list(X.columns),
            model_name=reg_model_name,
            input_kind="tabular",
            task="regress",
        )

        # time split and fit ONLY on train
        n = len(X)
        idx_train, idx_test = time_split_indices(n, test_size)
        X_train, y_train = X.iloc[idx_train], y_ret.iloc[idx_train]
        X_test, y_test = X.iloc[idx_test], y_ret.iloc[idx_test]

        model.pipeline = model._build_pipeline_reg()
        model.pipeline.fit(X_train, y_train)
        yhat_test = model.pipeline.predict(X_test)

        from sklearn.metrics import r2_score

        r2 = float(r2_score(y_test, yhat_test))

        rmse_bps = float(np.sqrt(mean_squared_error(y_test, yhat_test)))

        cost_bps = 2.0 * (fees_bps + slippage_bps)
        best = sweep_on_predicted_return(
            yhat_bps=np.asarray(yhat_test, dtype=float),
            fwd_ret_bps=np.asarray(y_test, dtype=float),
            cost_bps=cost_bps,
            best_metric=best_metric,
        )
        mape_pct = safe_mape_pct(y_test, yhat_test)

        # artifacts
        ts_tag = time.strftime("%Y%m%d_%H%M%S")
        tag = f"{symbol}_{interval_code}_{reg_model_name}_{ts_tag}"
        if save_datasets:
            os.makedirs(os.path.join(out_dir, "datasets"), exist_ok=True)
            data.df_ohlcv.to_csv(
                os.path.join(
                    out_dir,
                    "datasets",
                    f"OHLCV_{start_str.replace(' ', '_')}_{tag}.csv",
                ),
                index=False,
            )
            data.df_features.to_csv(
                os.path.join(
                    out_dir,
                    "datasets",
                    f"FEATURES_{start_str.replace(' ', '_')}_{tag}.csv",
                ),
                index=True,
            )
        if save_predictions:
            os.makedirs(os.path.join(out_dir, "predictions"), exist_ok=True)
            pred_df = pd.DataFrame(
                {"p_up": np.nan, "y_true": np.nan, "fwd_ret": np.nan}
            )
            pred_df.loc[: len(yhat_test) - 1, "fwd_ret"] = y_test.values
            pred_df.loc[: len(yhat_test) - 1, "p_up"] = (
                yhat_test  # store predicted return in 'p_up' slot for convenience
            )
            pred_df.to_csv(
                os.path.join(
                    out_dir,
                    "predictions",
                    f"PRED_{start_str.replace(' ', '_')}_{tag}.csv",
                ),
                index=False,
            )

        return {
            "symbol": symbol,
            "interval": interval_code,
            "start_str": start_str,
            "timelag": timelag,
            "model": reg_model_name,
            "rows": n,
            "split_mode": split_mode,
            "test_size": test_size,
            "class_weight": "",
            "label_mode": "return_bps",
            "ret_bps": 0.0,
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "auc": float("nan"),
            "confusion_matrix": [],
            "best_threshold": float(best.get("threshold", 0.0)),
            "trades": int(best.get("trades", 0)),
            "hit_rate": float("nan"),
            "avg_net_ret_per_bar": float(best.get("avg_net_ret_per_bar", 0.0)),
            "avg_net_ret_per_trade": float("nan"),
            "total_net_return": float(best.get("total_net_return", 0.0)),
            "sharpe_like": float(best.get("sharpe_like", float("nan"))),
            "sortino_like": float(best.get("sortino_like", float("nan"))),
            "cost_roundtrip": float(2.0 * ((fees_bps + slippage_bps) / 10_000.0)),
            "r2": float(r2),
            "mae_bps": float(mean_absolute_error(y_test, yhat_test)),
            "rmse_bps": float(rmse_bps),
            "mape_pct": float(mape_pct),
        }

    # ================= CLASSIFICATION PATH (original) =================
    X, y_dir = data.dataset(target="direction")

    # forward returns aligned with features (for money metrics)
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
        mask_ok = ~np.isnan(fwd_ret)
        X, y, fwd_ret = X[mask_ok], y[mask_ok], fwd_ret[mask_ok]
    else:
        y = y_dir.values

    seq_models = {"bilstm", "gru_lstm", "hybrid_transformer"}
    use_sequence = model_name in seq_models

    if use_sequence:
        window = max(2, timelag)
        stride = 1
        feat_cols = list(X.columns)
        X_seq = make_windows_from_df(
            data.df_features, feat_cols, window=window, stride=stride
        )
        n_seq = len(X_seq)
        if n_seq == 0:
            raise RuntimeError(
                f"Not enough rows ({len(X)}) to form any {window}-length windows."
            )

        idx_last = (window - 1) + np.arange(n_seq) * stride
        y = np.asarray(y)[idx_last]
        fwd_ret = fwd_ret[idx_last]
        X_nd = X_seq
    else:
        X_nd = X

    n = len(X_nd)
    if n < 200:
        print(f"[WARN] Only {n} rows; results may be noisy.")

    model = ModelManager(
        predictor_cols=list(X.columns),
        model_name=model_name,
        class_weight=class_weight,
        random_state=1,
        input_kind="sequence" if use_sequence else "tabular",
        sequence_maker=None,
        task="classify",
    )

    if split_mode == "random":
        test_acc = model.train(X_nd, pd.Series(y))
        pipe = model.pipeline
        p_up_all = pipe.predict_proba(X_nd)[:, 1]
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

        test_mask = np.ones(n, dtype=bool)
        p_up_test, y_test, fwd_test = p_up_all, y, fwd_ret

    else:  # time split
        idx_train, idx_test = time_split_indices(n, test_size)
        X_test, y_test = (
            (X_nd[idx_test] if use_sequence else X.iloc[idx_test]),
            y[idx_test],
        )

        model.pipeline = model._build_pipeline_clf()
        model.pipeline.fit(
            X_nd[idx_train] if use_sequence else X.iloc[idx_train], y[idx_train]
        )

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

    # threshold sweep (probability) w/ costs
    best = choose_best_threshold_for_window(
        p_up_window=p_up_test,
        fwd_ret_window=fwd_test,
        interval_code=interval_code,
        fees_bps=fees_bps,
        slippage_bps=slippage_bps,
        sweep=sweep_cfg,
        best_metric=best_metric,
    )

    # log loss (guarded)
    try:
        # p_up_test and y_test are already defined in both split modes
        logloss = float(log_loss(y_test, p_up_test, labels=[0, 1]))
    except Exception:
        logloss = float("nan")

    # sortino-like: rebuild per-trade net series at chosen threshold
    try:
        thr = float(best.get("threshold", 0.50))
        take = p_up_test >= thr
        trades = int(np.sum(take))
        if trades > 0:
            # fwd_test is fraction return; convert to bps, then subtract round-trip costs
            cost_bps = 2.0 * (fees_bps + slippage_bps)
            net_bps = (fwd_test[take] * 10_000.0) - cost_bps
            avg_net = float(np.nanmean(net_bps))
            downside = net_bps[net_bps < 0.0]
            downside_std = float(np.nanstd(downside)) if downside.size > 0 else np.nan
            sortino_like = float(avg_net / (downside_std + 1e-9)) if not np.isnan(downside_std) else float("nan")
        else:
            sortino_like = float("nan")
    except Exception:
        sortino_like = float("nan")


    # artifacts
    ts_tag = time.strftime("%Y%m%d_%H%M%S")
    tag = f"{symbol}_{interval_code}_{model_name}_{ts_tag}"

    if save_datasets:
        os.makedirs(os.path.join(out_dir, "datasets"), exist_ok=True)
        HistoryManager_df_ohlcv = data.df_ohlcv.copy()
        HistoryManager_df_features = data.df_features.copy()
        HistoryManager_df_ohlcv.to_csv(
            os.path.join(
                out_dir, "datasets", f"OHLCV_{start_str.replace(' ', '_')}_{tag}.csv"
            ),
            index=False,
        )
        HistoryManager_df_features.to_csv(
            os.path.join(
                out_dir, "datasets", f"FEATURES_{start_str.replace(' ', '_')}_{tag}.csv"
            ),
            index=True,
        )

    if save_predictions:
        os.makedirs(os.path.join(out_dir, "predictions"), exist_ok=True)
        pred_df = pd.DataFrame(
            {
                "is_test": np.where(test_mask, 1, 0),
                "p_up": np.nan,
                "y_true": np.nan,
                "fwd_ret": np.nan,
            }
        )
        if split_mode == "random":
            pred_df.loc[:, "p_up"] = p_up_test
            pred_df.loc[:, "y_true"] = y_test
            pred_df.loc[:, "fwd_ret"] = fwd_test
        else:
            pred_df.loc[test_mask, "p_up"] = p_up_test
            pred_df.loc[test_mask, "y_true"] = y_test
            pred_df.loc[test_mask, "fwd_ret"] = fwd_test

        pred_df.to_csv(
            os.path.join(
                out_dir, "predictions", f"PRED_{start_str.replace(' ', '_')}_{tag}.csv"
            ),
            index=False,
        )

    return {
        "symbol": symbol,
        "interval": interval_code,
        "start_str": start_str,
        "timelag": timelag,
        "model": model_name,
        "rows": len(X_nd),
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
        "best_threshold": float(best.get("threshold", 0.50)),
        "trades": int(best.get("trades", 0)),
        "hit_rate": float(best.get("hit_rate", float("nan"))),
        "avg_net_ret_per_bar": float(best.get("avg_net_ret_per_bar", 0.0)),
        "avg_net_ret_per_trade": float(best.get("avg_net_ret_per_trade", float("nan"))),
        "total_net_return": float(best.get("total_net_return", 0.0)),
        "sharpe_like": float(best.get("sharpe_like", float("nan"))),
        "sortino_like": float(sortino_like),
        "cost_roundtrip": float(2.0 * ((fees_bps + slippage_bps) / 10_000.0)),
        "r2": float("nan"),
        "mae_bps": float("nan"),
        "mape_pct": float("nan"),
        "log_loss": float(logloss),
    }


def main():
    p = argparse.ArgumentParser(
        description="Grid-validate models over intervals * start_strs * models."
    )
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument(
        "--start-list",
        default="365d,720d",
        help="Comma list of windows or absolute dates, e.g. '180d,365d,2021-01-01,90 days ago UTC'.",
    )
    p.add_argument(
        "--intervals", default="1h,4h", help="Comma list, e.g. '5m,15m,1h,4h,1d'."
    )
    p.add_argument(
        "--models",
        default="logreg,rf,hgb,linsvc,bilstm,gru_lstm,hybrid_transformer,voting_soft,stacking,metalabel,arima",
        help="Comma list for classification; for regression you can still pass these and they map to regressors.",
    )
    p.add_argument("--timelag", type=int, default=16)
    p.add_argument("--split-mode", choices=["time", "random"], default="time")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--class-weight", choices=["balanced", "none"], default="none")
    p.add_argument(
        "--label-mode", choices=["direction", "ret_gt_bps"], default="direction"
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
    p.add_argument(
        "--threshold-sweep",
        default="0.50:0.90:0.005",
        help="start:stop:step for p_up threshold",
    )
    p.add_argument(
        "--best-metric",
        choices=["sharpe_like", "total_net_return", "avg_net_ret_per_bar"],
        default="sharpe_like",
    )
    p.add_argument("--fees-bps", type=float, default=10.0)
    p.add_argument("--slippage-bps", type=float, default=0.0)
    p.add_argument("--task", choices=["classify", "regress"], default="regress")

    args = p.parse_args()

    start_list = parse_start_list(args.start_list)
    intervals = [s.strip() for s in args.intervals.split(",") if s.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    sweep_cfg = (
        parse_sweep(args.threshold_sweep) if args.threshold_sweep else SweepConfig()
    )
    class_weight = None if args.class_weight == "none" else "balanced"

    ensure_dirs(args.out_dir)

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
                        sweep_cfg=sweep_cfg,
                        fees_bps=args.fees_bps,
                        slippage_bps=args.slippage_bps,
                        label_mode=args.label_mode,
                        ret_bps=args.ret_bps,
                        best_metric=args.best_metric,
                        client=client,
                        task=args.task,
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

        # print a compact view that works for both tasks
        cols = [
            "start_str",
            "interval",
            "model",
            "rows",
            "accuracy",
            "r2",
            "log_loss",
            "rmse_bps",
            "mape_pct",
            "avg_net_ret_per_bar",
            "total_net_return",
            "sharpe_like",
            "sortino_like",
            "best_threshold",
            "trades",
        ]
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        print(df[cols])
    else:
        print("No successful evaluations.")


if __name__ == "__main__":
    main()
