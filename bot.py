"""
Key changes:
- Adds fee/slippage-aware threshold sweep on a rolling calibration window.
- Reuses shared metrics/sweep code.
- Uses the selected threshold when deciding to trade long-only.
"""

from __future__ import annotations
import time
import numpy as np
import pandas as pd
from binance.client import Client

import config
from managers.history_manager import INTERVAL_TO_MS, HistoryManager
from managers.model_manager import ModelManager
from managers.position_manager import PositionManager, SizingConfig
from evaluator import choose_best_threshold_for_window
from metrics import SweepConfig


# mapping kept as-is from your original bot
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




class TradeBot:
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        trade_interval_code: str | None = None,
        start_str: str | None = None,
        timelag: int = 20,
        retrain_every: int = 50,
        # NEW: threshold sweep calibration
        sweep_cfg: SweepConfig | None = None,
        best_metric: str = None,
        calib_window_bars: int = 2000,  # how many recent bars to calibrate on
        fees_bps: float = None,
        slippage_bps: float = None,
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
        self.interval_code = trade_interval_code  # keep for metrics annualization
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
        self.model = ModelManager(predictor_cols=self.data.predictor_cols)
        self.position = PositionManager(
            client=self.client, symbol=self.symbol, sizing=SizingConfig()
        )

        self._bars_since_retrain = 0
        self._current_threshold = 0.50  # fallback

    # ---------- Calibration ----------

    def _calibrate_threshold(self) -> None:
        """Run threshold sweep on the most recent calibration window using
        model probabilities vs next-bar returns computed from OHLCV.
        """
        # We'll recompute dataset to align features and build next-bar returns
        X, _y_dir = self.data.dataset()
        # probs over the window
        p_all = self.model.pipeline.predict_proba(X)[:, 1]

        # build forward returns aligned to X
        fwd_ret = (
            self.data.df_ohlcv["close"]
            .pct_change()
            .shift(-1)
            .reindex(self.data.df_features.index)
            .values
        )
        mask_ok = ~np.isnan(fwd_ret)
        p_all = p_all[mask_ok]
        fwd_ret = fwd_ret[mask_ok]

        # take the most recent window
        if p_all.size == 0:
            return
        start = max(0, p_all.size - self.calib_window_bars)
        p_win = p_all[start:]
        r_win = fwd_ret[start:]

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

    # ---------- Lifecycle ----------

    def bootstrap(self) -> None:
        print("Loading historical data...")
        self.data.load_history()
        X, y = self.data.dataset()
        print(f"Dataset ready: {len(X)} rows, {len(self.data.predictor_cols)} features")
        acc = self.model.train(X, y)
        print(f"Initial test accuracy: {acc:.3f}")
        self._calibrate_threshold()

    def _update_and_maybe_retrain(self) -> None:
        self.data.update_with_latest(limit=3)
        self._bars_since_retrain += 1
        if self._bars_since_retrain >= self.retrain_every:
            X, y = self.data.dataset()
            try:
                acc = self.model.train(X, y)
                print(f"Retrained. Test accuracy: {acc:.3f}")
                self._calibrate_threshold()  # recalibrate after retrain
            except Exception as e:
                print(f"[WARN] Retrain skipped: {e}")
            self._bars_since_retrain = 0

    # ---------- Trading Loop ----------

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

                    # Close any existing long at the bar boundary
                    self.position.close_long()

                    # Prepare a single prediction for next bar
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
        trade_interval_code=getattr(config, "trade_interval", "5m"),
        start_str=getattr(config, "start_str", "60 days ago UTC"),
        timelag=getattr(config, "timelag", 20),
        retrain_every=getattr(config, "retrain_every", 50),
    )
    bot.run()
