import time
import numpy as np
import pandas as pd
from binance.client import Client

import config
from managers.history_manager import INTERVAL_TO_MS, HistoryManager
from managers.model_manager import ModelManager
from managers.position_manager import PositionManager, SizingConfig
from evaluations.evaluator import choose_best_threshold_for_window
from evaluations.metrics import SweepConfig

_INTERVAL_MAP = {
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


def _map_trade_interval_to_binance(code: str) -> str:
    if code not in _INTERVAL_MAP:
        raise ValueError(f"Unsupported trade_interval: {code}")
    return _INTERVAL_MAP[code]


def _make_rolling_windows(
    df: pd.DataFrame, feat_cols: list[str], window: int
) -> np.ndarray:
    arr = df[feat_cols].to_numpy(dtype=np.float32, copy=False)
    if len(arr) < window:
        return np.empty(
            (0, window, arr.shape[1] if arr.ndim == 2 else 0), dtype=np.float32
        )
    n = len(arr) - window + 1
    out = np.empty((n, window, arr.shape[1]), dtype=np.float32)
    for i in range(n):
        out[i] = arr[i : i + window]
    return out


def _last_window(df: pd.DataFrame, feat_cols: list[str], window: int) -> np.ndarray:
    if len(df) < window:
        raise RuntimeError(f"Not enough rows ({len(df)}) for window={window}")
    arr = df[feat_cols].to_numpy(dtype=np.float32, copy=False)
    return arr[-window:].reshape(1, window, arr.shape[1])


class TradeBot:
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        trade_interval_code: str | None = None,
        start_str: str | None = None,
        timelag: int = 20,
        retrain_every: int = 50,
        sweep_cfg: SweepConfig | None = None,
        best_metric: str | None = None,
        calib_window_bars: int = 2000,
        fees_bps: float | None = None,
        slippage_bps: float | None = None,
        model_name: str | None = None,
    ) -> None:
        trade_interval_code = trade_interval_code or getattr(
            config, "trade_interval", "5m"
        )
        start_str = start_str or getattr(config, "start_str", "60 days ago UTC")
        self.best_metric = best_metric or getattr(config, "best_metric", "sharpe_like")
        self.sweep_cfg = sweep_cfg or SweepConfig(
            *tuple(
                map(
                    float,
                    (getattr(config, "threshold_sweep", "0.50:0.90:0.005").split(":")),
                )
            )
        )
        self.calib_window_bars = int(
            getattr(config, "calib_window_bars", calib_window_bars)
        )
        self.fees_bps = float(
            getattr(config, "fees_bps", fees_bps if fees_bps is not None else 10.0)
        )
        self.slippage_bps = float(
            getattr(
                config,
                "slippage_bps",
                slippage_bps if slippage_bps is not None else 5.0,
            )
        )
        self.client = Client(
            getattr(config, "api_key", None),
            getattr(config, "api_secret", None),
            testnet=getattr(config, "is_test_net", False),
        )
        self.symbol = symbol
        self.interval_code = trade_interval_code
        self.interval = _map_trade_interval_to_binance(trade_interval_code)
        self.interval_ms = INTERVAL_TO_MS[self.interval]
        self.retrain_every = max(1, retrain_every)
        self.data = HistoryManager(
            client=self.client,
            symbol=self.symbol,
            interval=self.interval,
            start_str=start_str,
            timelag=timelag,
        )

        self.model_name = model_name or getattr(config, "model_name", "hgb")
        self.seq_models = {"bilstm", "gru_lstm", "hybrid_transformer"}
        self.use_sequence = self.model_name in self.seq_models
        self.window = max(2, timelag)
        self.position = PositionManager(
            client=self.client, symbol=self.symbol, sizing=SizingConfig()
        )
        self._bars_since_retrain = 0
        self._current_threshold = 0.50

    def _calibrate_threshold(self) -> None:
        X, _ = self.data.dataset()
        feat_cols = list(X.columns)

        fwd_ret_series = (
            self.data.df_ohlcv["close"]
            .pct_change()
            .shift(-1)
            .reindex(self.data.df_features.index)
        )

        if self.use_sequence:
            X_seq = _make_rolling_windows(self.data.df_features, feat_cols, self.window)
            if len(X_seq) == 0:
                return
            p_all = self.model.pipeline.predict_proba(X_seq)[:, 1]

            # one label per window: take return at the last row of each window
            idx_last = np.arange(self.window - 1, self.window - 1 + len(X_seq))
            r = fwd_ret_series.iloc[idx_last].to_numpy(dtype=float, copy=False)

        else:
            # tabular: one prob per feature row, same index/length as df_features
            p_all = self.model.pipeline.predict_proba(X)[:, 1]
            r = fwd_ret_series.to_numpy(dtype=float, copy=False)

        # drop NaNs using a mask that matches p_all length exactly
        if len(r) != len(p_all):
            # extra safety: align lengths conservatively (shouldn’t trigger with logic above)
            m = min(len(r), len(p_all))
            r = r[-m:]
            p_all = p_all[-m:]

        mask = ~np.isnan(r)
        if not mask.any():
            return
        p_all = p_all[mask]
        r = r[mask]
        if p_all.size == 0:
            return

        start = max(0, p_all.size - self.calib_window_bars)
        p_win = p_all[start:]
        r_win = r[start:]

        best = choose_best_threshold_for_window(
            p_up_window=p_win,
            fwd_ret_window=r_win,
            interval_code=self.interval_code,
            fees_bps=self.fees_bps,
            slippage_bps=self.slippage_bps,
            sweep=self.sweep_cfg,
            best_metric=self.best_metric,
        )
        self._current_threshold = float(best.get("threshold", 0.50))
        print(
            f"[CALIBRATION] threshold={self._current_threshold:.3f} metric={self.best_metric} value={best.get(self.best_metric)} trades={best.get('trades')}"
        )

    def bootstrap(self) -> None:
        print("Loading historical data...")
        self.data.load_history()
        X, y = self.data.dataset()
        self.model = ModelManager(
            predictor_cols=self.data.predictor_cols,
            model_name=self.model_name,
            input_kind="sequence" if self.use_sequence else "tabular",
        )
        print(f"Dataset ready: {len(X)} rows, {len(self.data.predictor_cols)} features")
        if self.use_sequence:
            feat_cols = list(X.columns)
            X_seq = _make_rolling_windows(self.data.df_features, feat_cols, self.window)
            if len(X_seq) == 0:
                raise RuntimeError(f"Not enough data for window={self.window}")
            y_seq = np.asarray(y)[self.window - 1 :]
            y_seq = y_seq[-len(X_seq) :]
            acc = self.model.train(X_seq, y_seq)
        else:
            acc = self.model.train(X, y)
        print(f"Initial test accuracy: {acc:.3f}")
        self._calibrate_threshold()

    def _update_and_maybe_retrain(self) -> None:
        self.data.update_with_latest(limit=3)
        self._bars_since_retrain += 1
        if self._bars_since_retrain >= self.retrain_every:
            X, y = self.data.dataset()
            try:
                if self.use_sequence:
                    feat_cols = list(X.columns)
                    X_seq = _make_rolling_windows(
                        self.data.df_features, feat_cols, self.window
                    )
                    if len(X_seq) == 0:
                        raise RuntimeError(f"Not enough data for window={self.window}")
                    y_seq = np.asarray(y)[self.window - 1 :]
                    y_seq = y_seq[-len(X_seq) :]
                    acc = self.model.train(X_seq, y_seq)
                else:
                    acc = self.model.train(X, y)
                print(f"Retrained. Test accuracy: {acc:.3f}")
                self._calibrate_threshold()
            except Exception as e:
                print(f"[WARN] Retrain skipped: {e}")
            self._bars_since_retrain = 0

    def run(self) -> None:
        self.bootstrap()
        next_trade_time = self.data.next_bar_close_time_ms()
        print(f"Next decision at ~{pd.to_datetime(next_trade_time, unit='ms')}")
        while True:
            now_ms = int(time.time() * 1000)
            if now_ms >= next_trade_time:
                while now_ms >= next_trade_time:
                    print(f"\n[BAR CLOSE] {pd.to_datetime(next_trade_time, unit='ms')}")
                    self._update_and_maybe_retrain()
                    self.position.close_long()
                    if self.use_sequence:
                        X, _ = self.data.dataset()
                        feat_cols = list(X.columns)
                        try:
                            X_win = _last_window(
                                self.data.df_features, feat_cols, self.window
                            )
                            last_price = float(self.data.df_ohlcv["close"].iloc[-1])
                            p_up = self.model.predict_proba_up(X_win)
                        except Exception as e:
                            print(f"[WARN] Prediction skipped: {e}")
                            p_up = 0.5
                    else:
                        X_new = self.data.latest_features_row()
                        last_price = float(X_new["CLOSE"].iloc[0])
                        try:
                            p_up = self.model.predict_proba_up(X_new)
                        except Exception as e:
                            print(f"[WARN] Prediction skipped: {e}")
                            p_up = 0.5
                    p_down = 1.0 - p_up
                    print(
                        f"Predicted up/down: {p_up:.3f}/{p_down:.3f}  thr={self._current_threshold:.3f}"
                    )
                    if p_up >= self._current_threshold:
                        qty = self.position.compute_quantity_kelly(p_up, last_price)
                        print(f"Computed qty (kelly-capped): {qty}")
                        self.position.open_long(qty)
                    else:
                        print("Signal below threshold; staying flat.")
                    next_trade_time += self.interval_ms
                try:
                    usdt_bal = self.client.get_asset_balance(asset="USDT")
                    base_asset = self.symbol.replace("USDT", "")
                    base_bal = self.client.get_asset_balance(asset=base_asset)
                    print(
                        f"Balances: USDT={usdt_bal['free']} {base_asset}={base_bal['free']}"
                    )
                except Exception as e:
                    print(f"[WARN] Balance fetch failed: {e}")
            try:
                time.sleep(1)
            except Exception:
                pass


if __name__ == "__main__":
    bot = TradeBot(
        symbol=getattr(config, "symbol", "BTCUSDT"),
        trade_interval_code=getattr(config, "trade_interval", "1h"),
        start_str=getattr(config, "start_str", "720 days ago UTC"),
        timelag=getattr(config, "timelag", 16),
        retrain_every=getattr(config, "retrain_every", 30),
        model_name=getattr(config, "model_name", "hgb"),
    )
    bot.run()
