# Provides HistoricalData, a thin OOP wrapper around Binance klines that:
# - Loads historical OHLCV
# - Computes indicators (EMA, CMO, +DM, -DM) and candlestick patterns
# - Builds a clean features DataFrame with a next-bar target (UP_DOWN)
# - Updates with the latest closed bar and exposes prediction features
#
# Dependencies: numpy, pandas, talib, python-binance, config.py

from typing import Tuple

import numpy as np
import pandas as pd
import talib
from binance.client import Client

INTERVAL_TO_MS = {
    Client.KLINE_INTERVAL_1MINUTE: 60_000,
    Client.KLINE_INTERVAL_3MINUTE: 3 * 60_000,
    Client.KLINE_INTERVAL_5MINUTE: 5 * 60_000,
    Client.KLINE_INTERVAL_15MINUTE: 15 * 60_000,
    Client.KLINE_INTERVAL_30MINUTE: 30 * 60_000,
    Client.KLINE_INTERVAL_1HOUR: 60 * 60_000,
    Client.KLINE_INTERVAL_2HOUR: 2 * 60 * 60_000,
    Client.KLINE_INTERVAL_4HOUR: 4 * 60 * 60_000,
    Client.KLINE_INTERVAL_6HOUR: 6 * 60 * 60_000,
    Client.KLINE_INTERVAL_8HOUR: 8 * 60 * 60_000,
    Client.KLINE_INTERVAL_12HOUR: 12 * 60 * 60_000,
    Client.KLINE_INTERVAL_1DAY: 24 * 60 * 60_000,
}


def _as_dataframe(klines: list) -> pd.DataFrame:
    """
    Convert raw binance klines to a pandas DataFrame.
    Columns (subset): open_time, open, high, low, close, volume
    """
    cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]
    df = pd.DataFrame(klines, columns=cols)
    # keep numeric types
    for c in ["open_time", "close_time", "number_of_trades"]:
        df[c] = pd.to_numeric(df[c], downcast="integer", errors="coerce")
    for c in [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "taker_buy_base",
        "taker_buy_quote",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[["open_time", "open", "high", "low", "close", "volume"]]
    return df


def _bin_pattern_to_binary(arr: np.ndarray) -> np.ndarray:
    """
    TA-Lib candlestick functions return -100, 0, +100.
    Map strictly positive to 1, else 0 (so -100 and 0 => 0).
    """
    # sign(positive) -> 1; sign(0 or negative) -> 0
    return (np.sign(arr) > 0).astype(np.int32)


class HistoryManager:
    """
    Manages historical and latest market data, indicators, and features.
    """

    def __init__(
        self,
        client: Client,
        symbol: str,
        interval: str,
        start_str: str,
        timelag: int = 20,
    ) -> None:
        self.client = client
        self.symbol = symbol
        self.interval = interval
        self.start_str = start_str
        self.timelag = timelag

        if interval not in INTERVAL_TO_MS:
            raise ValueError(f"Unsupported interval: {interval}")

        self.interval_ms = INTERVAL_TO_MS[interval]

        # core OHLCV data
        self.df_ohlcv: pd.DataFrame = pd.DataFrame()
        # full modeling table (features + label)
        self.df_features: pd.DataFrame = pd.DataFrame()

        # List of predictor column names (fixed order)
        self.predictor_cols = [
            "EMA",
            "CMO",
            "MINUSDM",
            "PLUSDM",
            "CLOSE",
            "CLOSEL1",
            "CLOSEL2",
            "CLOSEL3",
            "PATT_3OUT",  # binary pattern
            "PATT_CMB",  # binary pattern
        ]

    # ---------- Data Fetch ----------

    def load_history(self) -> None:
        """
        Loads full history from start_str to now and builds features.
        """
        klines = self.client.get_historical_klines(
            symbol=self.symbol,
            interval=self.interval,
            start_str=self.start_str,
        )
        self.df_ohlcv = _as_dataframe(klines)
        if self.df_ohlcv.empty:
            raise RuntimeError("No historical data returned.")

        self._recompute_features()

    def update_with_latest(self, limit: int = 2) -> None:
        """
        Pulls the most recent klines (default 2 for safety), appends any new CLOSED bars,
        then recomputes indicators/features.
        """
        recent = self.client.get_klines(
            symbol=self.symbol, interval=self.interval, limit=limit
        )
        df_recent = _as_dataframe(recent)

        if self.df_ohlcv.empty:
            self.df_ohlcv = df_recent.copy()
        else:
            last_open_time = int(self.df_ohlcv["open_time"].iloc[-1])
            new_rows = df_recent[df_recent["open_time"] > last_open_time]
            if not new_rows.empty:
                self.df_ohlcv = pd.concat([self.df_ohlcv, new_rows], ignore_index=True)

        # Rebuild features whenever we extend ohclv (simple & safe)
        self._recompute_features()

    # ---------- Features ----------

    def _recompute_features(self) -> None:
        """
        Compute indicators, patterns and target with a consistent alignment.
        Target UP_DOWN = 1 if next close > current close (shift(-1)), else 0.
        Drop NaNs from indicator warmups and the last unlabeled row.
        """
        df = self.df_ohlcv.copy()

        # Technical indicators
        close = df["close"].to_numpy(dtype=float)
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)

        ema = talib.EMA(close, timeperiod=self.timelag)
        cmo = talib.CMO(close, timeperiod=self.timelag)
        minusdm = talib.MINUS_DM(high, low, timeperiod=self.timelag)
        plusdm = talib.PLUS_DM(high, low, timeperiod=self.timelag)

        patt_3out_raw = talib.CDL3OUTSIDE(
            df["open"].to_numpy(dtype=float),
            high,
            low,
            close,
        )
        patt_cmb_raw = talib.CDLCLOSINGMARUBOZU(
            df["open"].to_numpy(dtype=float),
            high,
            low,
            close,
        )

        patt_3out = _bin_pattern_to_binary(patt_3out_raw)
        patt_cmb = _bin_pattern_to_binary(patt_cmb_raw)

        # shifts
        closel1 = df["close"].shift(1)
        closel2 = df["close"].shift(2)
        closel3 = df["close"].shift(3)

        # assemble feature table
        feat = pd.DataFrame(
            {
                "EMA": ema,
                "CMO": cmo,
                "MINUSDM": minusdm,
                "PLUSDM": plusdm,
                "CLOSE": df["close"],
                "CLOSEL1": closel1,
                "CLOSEL2": closel2,
                "CLOSEL3": closel3,
                "PATT_3OUT": patt_3out,
                "PATT_CMB": patt_cmb,
            },
            index=df.index,
        )

        # target for *next* bar
        up_down = (df["close"].shift(-1) > df["close"]).astype("float64")
        feat["UP_DOWN"] = up_down

        # drop warmup NaNs and unlabeled last row
        feat = feat.dropna().astype({"UP_DOWN": "int64"})
        self.df_features = feat

    # ---------- Accessors ----------

    def last_closed_open_time_ms(self) -> int:
        """
        Returns the open_time ms of the last CLOSED bar in df_features.
        (df_features has dropped the last unlabeled row, so last row is safely closed)
        """
        # map index back to base ohlcv
        last_idx = self.df_features.index[-1]
        return int(self.df_ohlcv.loc[last_idx, "open_time"])

    def next_bar_close_time_ms(self) -> int:
        """
        Compute the expected close time of the next bar after the last CLOSED bar in features.
        """
        last_open_ms = self.last_closed_open_time_ms()
        return last_open_ms + self.interval_ms

    def dataset(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Returns (X, y) for model training.
        """
        X = self.df_features[self.predictor_cols].copy()
        y = self.df_features["UP_DOWN"].copy()
        return X, y

    def latest_features_row(self) -> pd.DataFrame:
        """
        Returns the last available feature row (for predicting the next bar).
        This is the final row of df_features.
        """
        return self.df_features[self.predictor_cols].iloc[[-1]].copy()
