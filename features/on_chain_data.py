import requests
import pandas as pd
from typing import Dict, List, Optional, Literal

"""

df_4h  = get_btc_onchain_smoothed("2025-06-30", "2025-09-28", interval="4h",
                                  lookback_days=3, include_price=False)

unique_addresses_used_smooth - smoothed daily count of distinct BTC addresses active (sender or receiver). Proxy for user/activity breadth
confirmed_tx_per_day_smooth - smoothed daily number of confirmed transactions
output_value_per_day_btc_smooth - smoothed total BTC value moved on-chain that day (in BTC)Big swings can reflect exchange/whale moves
avg_tx_per_block_smooth - smoothed average transactions included per block
median_confirmation_time_min_smooth - smoothed median minutes a tx waited from broadcast to confirmation
avg_confirmation_time_min_smooth - smoothed average confirmation time (in minutes)
total_hash_rate_ths_smooth - smoothed network hashrate (in terahashes per second)Security / miner supply-side proxy
network_difficulty_smooth - smoothed mining difficultyl. Adjusts ~bi-weekly; stepwise series
miners_revenue_usd_smooth - smoothed miner revenue in USD (subsidy + fees)
total_tx_fees_btc_smooth - smoothed total fees paid by users (in BTC) per day
blockchain_size_smooth - smoothed cumulative blockchain data size (GB)Monotonic ↑ by design

unique_addresses_used_smooth	confirmed_tx_per_day_smooth	output_value_per_day_btc_smooth	avg_tx_per_block_smooth	median_confirmation_time_min_smooth	avg_confirmation_time_min_smooth	total_hash_rate_ths_smooth	network_difficulty_smooth	miners_revenue_usd_smooth	total_tx_fees_btc_smooth	blockchain_size_smooth	avg_block_size_smooth
ts_utc												
2025-07-01 00:00:00+00:00	521131.0	339268.0	629712.813392	2339.77931	6.85	16.949163	8.430357e+08	1.169585e+14	4.933359e+07	4.228973	668824.13727	1.377226
2025-07-01 04:00:00+00:00	521131.0	339268.0	629712.813392	2339.77931	6.85	16.949163	8.430357e+08	1.169585e+14	4.933359e+07	4.228973	668824.13727	1.377226
2025-07-01 08:00:00+00:00	521131.0	339268.0	629712.813392	2339.77931	6.85	16.949163	8.430357e+08	1.169585e+14	4.933359e+07	4.228973	668824.13727	1.377226
2025-07-01 12:00:00+00:00	521131.0	339268.0	629712.813392	2339.77931	6.85	16.949163	8.430357e+08	1.169585e+14	4.933359e+07	4.228973	668824.13727	1.377226

"""




BC_BASE = "https://api.blockchain.info/charts/{slug}"
UA = {"User-Agent": "onchain-mini/1.0"}

SUPPORTED_INTERVALS = {"1h", "4h", "1d"}

BTC_METRICS: Dict[str, str] = {
    "n-unique-addresses": "unique_addresses_used",
    "n-transactions": "confirmed_tx_per_day",
    "output-volume": "output_value_per_day_btc",
    "n-transactions-per-block": "avg_tx_per_block",
    "median-confirmation-time": "median_confirmation_time_min",
    "avg-confirmation-time": "avg_confirmation_time_min",
    "hash-rate": "total_hash_rate_ths",
    "difficulty": "network_difficulty",
    "miners-revenue": "miners_revenue_usd",
    "transaction-fees": "total_tx_fees_btc",
    "blocks-size": "blockchain_size",
    "avg-block-size": "avg_block_size",
}

# --- helpers ---
def _fetch_chart(slug: str, start_iso: str) -> Optional[dict]:
    try:
        r = requests.get(
            BC_BASE.format(slug=slug),
            params={"start": start_iso, "format": "json", "sampled": "false"},
            headers=UA, timeout=30,
        )
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None

def _to_df(payload: dict, col: str) -> pd.DataFrame:
    vals = payload.get("values", [])
    if not vals:
        return pd.DataFrame(columns=[col])
    df = pd.DataFrame(vals)
    df["ts"] = pd.to_datetime(df["x"], unit="s", utc=True)
    df[col] = pd.to_numeric(df["y"], errors="coerce")
    return df[["ts", col]].dropna().sort_values("ts")

def _clip(df: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    if df.empty:
        return df
    m = (df["ts"] >= start_ts) & (df["ts"] <= end_ts)
    return df.loc[m]

def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure index is tz-aware UTC without raising on already-aware indexes."""
    idx = df.index
    if getattr(idx, "tz", None) is None:
        df.index = idx.tz_localize("UTC")
    else:
        df.index = idx.tz_convert("UTC")
    return df

def _to_daily_groupby(df: pd.DataFrame, col: str, how: Literal["mean","last"]) -> pd.DataFrame:
    """Force daily by grouping on UTC date (no resample bin-edge quirks)."""
    if df.empty:
        return pd.DataFrame(columns=[col])
    day = df["ts"].dt.floor("D")  # keeps tz-awareness
    if how == "mean":
        out = df.groupby(day, as_index=True)[col].mean()
    else:
        out = df.groupby(day, as_index=True)[col].last()
    out.index.name = "ts_utc"
    out = out.to_frame()
    return _ensure_utc_index(out)

def _smooth_past_only(daily_df: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    """Per-column: shift(1).rolling(window).mean() -> leakage-safe."""
    out = pd.DataFrame(index=daily_df.index)
    for c in daily_df.columns:
        s = daily_df[c].astype(float)
        out[c + "_smooth"] = s.shift(1).rolling(lookback_days, min_periods=1).mean()
    return out

def _make_grid(start_ts: pd.Timestamp, end_ts: pd.Timestamp, interval: str) -> pd.DatetimeIndex:
    if interval == "1d":
        return pd.date_range(start=start_ts.floor("D"), end=end_ts.floor("D"), freq="1d", tz="UTC")
    if interval == "4h":
        return pd.date_range(start=start_ts.floor("4h"), end=end_ts, freq="4h", tz="UTC")
    if interval == "1h":
        return pd.date_range(start=start_ts.floor("1h"), end=end_ts, freq="1h", tz="UTC")
    raise ValueError("Unsupported interval")

def _fetch_binance(symbol: str, interval: Literal["1h","4h","1d"],
                   start_ms: int, end_ms: int) -> pd.DataFrame:
    """Use kline OPEN time so indices land exactly on the grid."""
    iv = {"1h":"1h","4h":"4h","1d":"1d"}[interval]
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": iv, "startTime": start_ms, "endTime": end_ms, "limit": 1000}
    try:
        r = requests.get(url, params=params, headers=UA, timeout=30); r.raise_for_status()
        data = r.json()
    except requests.RequestException:
        return pd.DataFrame()

    if not isinstance(data, list) or not data:
        return pd.DataFrame()

    cols = ["open_time","open","high","low","close","volume",
            "close_time","qav","num_trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(data, columns=cols)
    ts = pd.to_datetime(df["open_time"], unit="ms", utc=True)  # align on open
    out = pd.DataFrame({
        "ts_utc": ts,
        "price_open":  pd.to_numeric(df["open"], errors="coerce"),
        "price_high":  pd.to_numeric(df["high"], errors="coerce"),
        "price_low":   pd.to_numeric(df["low"], errors="coerce"),
        "price_close": pd.to_numeric(df["close"], errors="coerce"),
        "price_volume":pd.to_numeric(df["volume"], errors="coerce"),
    }).dropna(subset=["price_close"])
    return out.set_index("ts_utc").sort_index()

# --- public: minimal API ---
def get_btc_onchain_smoothed(
    start: str,
    end: str,
    interval: Literal["1h","4h","1d"] = "1d",
    metrics: Optional[List[str]] = None,
    lookback_days: int = 3,
    agg_to_daily: Literal["mean","last"] = "mean",
    include_price: bool = False,
    price_symbol: str = "BTCUSDT",
) -> pd.DataFrame:
    """
    Minimal on-chain fetch for BTC:
      1) fetch each metric
      2) force to DAILY via groupby(day)
      3) build full daily grid and forward-fill past values
      4) leakage-safe smoothing: shift(1).rolling(lookback_days).mean()
      5) expand to 1h/4h/1d via forward-fill
      6) (optional) join Binance BTCUSDT on the same grid (aligned by kline OPEN)
    """
    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(f"interval must be one of {sorted(SUPPORTED_INTERVALS)}")

    start_ts = pd.to_datetime(start, utc=True)
    end_ts   = pd.to_datetime(end,   utc=True)
    if end_ts < start_ts:
        raise ValueError("end must be >= start")

    chosen = BTC_METRICS if metrics is None else {k: BTC_METRICS[k] for k in metrics if k in BTC_METRICS}

    # 1) pull & 2) daily-ize
    daily_frames = []
    for slug, col in chosen.items():
        payload = _fetch_chart(slug, start_ts.strftime("%Y-%m-%d"))
        if not payload:
            continue
        df = _to_df(payload, col)
        df = _clip(df, start_ts, end_ts)
        if df.empty:
            continue
        daily = _to_daily_groupby(df, col, agg_to_daily)  # clean daily index at 00:00 UTC
        daily_frames.append(daily)

    if not daily_frames:
        return pd.DataFrame()

    # 3) full daily grid + ffill (uses only past values)
    daily = pd.concat(daily_frames, axis=1, join="outer").sort_index()
    full_idx = _make_grid(start_ts, end_ts, "1d")
    daily = daily.reindex(full_idx).ffill()  # propagate prior values
    daily.index.name = "ts_utc"

    # 4) leakage-safe smoothing (per column)
    daily_smooth = _smooth_past_only(daily, lookback_days)  # *_smooth columns

    # 5) expand to requested interval
    if interval == "1d":
        out = daily_smooth.copy()
    else:
        grid = _make_grid(start_ts, end_ts, interval)
        out = daily_smooth.reindex(grid, method="ffill")

    # 6) optional price merge (LEFT join to avoid extra timestamps)
    if include_price:
        price = _fetch_binance(
            price_symbol, interval,
            int(out.index[0].timestamp() * 1000),
            int(out.index[-1].timestamp() * 1000),
        )
        out = out.join(price, how="left")

    out.index.name = "ts_utc"
    return out
