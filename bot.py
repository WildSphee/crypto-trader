# Trading bot using OOP:
# - Loads history + builds features
# - Trains a Logistic Regression pipeline (proper train/test split)
# - On each closed bar: updates data, optionally re-trains (configurable),
#   gets a single prediction, sizes with capped Kelly, and trades long-only.
#
# IMPORTANT: This is an educational template. Real trading needs more safety checks,
# error handling, and risk controls than this example.

from __future__ import annotations

import time

import pandas as pd
from binance.client import Client

import config
from history_manager import INTERVAL_TO_MS, HistoryManager

# from binance.enums import *
from model_manager import ModelManager
from position_manager import PositionManager, SizingConfig

# ---------- Utilities ----------


def _map_trade_interval_to_binance(interval_code: str) -> str:
    """
    Accepts: "5m", "15m", "1h" (or any that map to Client.KLINE_INTERVAL_*).
    """
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
    if interval_code not in m:
        raise ValueError(f"Unsupported trade_interval: {interval_code}")
    return m[interval_code]


class TradeBot:
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        trade_interval_code: str = None,
        start_str: str = None,
        timelag: int = 20,
        retrain_every: int = 50,  # retrain every N bars (set 1 if you want each bar)
    ) -> None:
        # config fallbacks
        trade_interval_code = trade_interval_code or getattr(
            config, "trade_interval", "5m"
        )
        start_str = start_str or getattr(config, "start_str", "60 days ago UTC")

        self.client = Client(
            getattr(config, "api_key", None),
            getattr(config, "api_secret", None),
            testnet=getattr(config, "is_test_net", False),
        )

        self.symbol = symbol
        self.interval = _map_trade_interval_to_binance(trade_interval_code)
        self.interval_ms = INTERVAL_TO_MS[self.interval]
        self.retrain_every = max(1, retrain_every)

        # data + model + position managers
        self.data = HistoryManager(
            client=self.client,
            symbol=self.symbol,
            interval=self.interval,
            start_str=start_str,
            timelag=timelag,
        )
        self.model = ModelManager(predictor_cols=self.data.predictor_cols)
        self.position = PositionManager(
            client=self.client,
            symbol=self.symbol,
            sizing=SizingConfig(),
        )

        self._bars_since_retrain = 0

    # ---------- Lifecycle ----------

    def bootstrap(self) -> None:
        """
        Load history, build features, and train the model once.
        """
        print("Loading historical data...")
        self.data.load_history()
        X, y = self.data.dataset()

        print(f"Dataset ready: {len(X)} rows, {len(self.data.predictor_cols)} features")
        acc = self.model.train(X, y)
        print(f"Initial test accuracy: {acc:.3f}")

    def _update_and_maybe_retrain(self) -> None:
        """
        Update data with the latest closed bar; retrain according to schedule.
        """
        self.data.update_with_latest(limit=3)
        self._bars_since_retrain += 1

        if self._bars_since_retrain >= self.retrain_every:
            X, y = self.data.dataset()
            try:
                acc = self.model.train(X, y)
                print(f"Retrained. Test accuracy: {acc:.3f}")
            except Exception as e:
                print(f"[WARN] Retrain skipped: {e}")
            self._bars_since_retrain = 0

    # ---------- Trading Loop ----------

    def run(self) -> None:
        """
        Main loop: wait for each closed bar, then decide and act (long-only).
        Closes any prior long at the bar boundary (following your original logic),
        then decides whether to open a new long for the next bar.
        """
        self.bootstrap()

        # compute the first decision time (close of next bar)
        next_trade_time = self.data.next_bar_close_time_ms()
        print(f"Next decision at ~{pd.to_datetime(next_trade_time, unit='ms')}")

        while True:
            now_ms = int(time.time() * 1000)

            # If time passed the expected decision point, process (may need to catch up)
            if now_ms >= next_trade_time:
                # For robustness, catch up if multiple bars elapsed
                while now_ms >= next_trade_time:
                    print(f"\n[BAR CLOSE] {pd.to_datetime(next_trade_time, unit='ms')}")
                    # Update data and (optionally) retrain
                    self._update_and_maybe_retrain()

                    # Close any existing long from the previous signal (your original design)
                    self.position.close_long()

                    # Prepare a single prediction for the new bar
                    X_new = self.data.latest_features_row()
                    # The last price (close of just-closed bar)
                    last_price = float(X_new["CLOSE"].iloc[0])

                    try:
                        p_up = self.model.predict_proba_up(X_new)
                    except Exception as e:
                        print(f"[WARN] Prediction skipped: {e}")
                        p_up = 0.5

                    p_down = 1.0 - p_up
                    print(f"Predicted up/down: {p_up:.3f}/{p_down:.3f}")

                    # Long-only logic
                    if p_up >= 0.5:
                        qty = self.position.compute_quantity_kelly(p_up, last_price)
                        print(f"Computed qty (kelly-capped): {qty}")
                        self.position.open_long(qty)
                    else:
                        print("Signal not long; staying flat (shorting disabled).")

                    # schedule next bar decision
                    next_trade_time += self.interval_ms

                # show current balances
                try:
                    usdt_bal = self.client.get_asset_balance(asset="USDT")
                    base_asset = self.symbol.replace("USDT", "")
                    base_bal = self.client.get_asset_balance(asset=base_asset)
                    print(
                        f"Balances: USDT={usdt_bal['free']} {base_asset}={base_bal['free']}"
                    )
                except Exception as e:
                    print(f"[WARN] Balance fetch failed: {e}")

            # small sleep to avoid busy loop
            try:
                time.sleep(1)
            except Exception:
                pass


# ---------- Entrypoint ----------

if __name__ == "__main__":
    # Choose interval from config.trade_interval ("5m", "15m", "1h", ...)
    bot = TradeBot(
        symbol="BTCUSDT",
        trade_interval_code=getattr(config, "trade_interval", "5m"),
        start_str=getattr(config, "start_str", "60 days ago UTC"),
        timelag=20,
        retrain_every=50,  # set to 1 if you want to retrain on every new bar
    )
    bot.run()
