import requests
import pandas as pd
from typing import Literal, Optional

"""
example output (4h interval, lookback_days=3, past_only=True):
fng_score — the raw Fear & Greed Index value (0-100) for that calendar day.
fng_label — the provider's text label for that day (Extreme Fear, Fear, Neutral, Greed, Extreme Greed).
fng_ordinal — your label mapped to a number 1-5. (Extreme Fear=1 … Extreme Greed=5).
fng_ordinal_smooth — a past-only rolling average of fng_ordinal over your lookback window (3 days in your run). It only uses prior days, so it's leakage-safe.
fng_ordinal_smooth_int — fng_ordinal_smooth rounded to the nearest integer 1-5 (handy if your model wants discrete states instead of a float).


fng_score	fng_label	fng_ordinal	fng_ordinal_smooth	fng_ordinal_smooth_int
ts_utc					
2025-06-30 00:00:00+00:00	66	Greed	4	NaN	<NA>
2025-06-30 04:00:00+00:00	66	Greed	4	NaN	<NA>
2025-06-30 08:00:00+00:00	66	Greed	4	NaN	<NA>
2025-06-30 12:00:00+00:00	66	Greed	4	NaN	<NA>
2025-06-30 16:00:00+00:00	66	Greed	4	NaN	<NA>
...	...	...	...	...	...
2025-09-27 08:00:00+00:00	33	Fear	2	2.0	2
2025-09-27 12:00:00+00:00	33	Fear	2	2.0	2
2025-09-27 16:00:00+00:00	33	Fear	2	2.0	2
2025-09-27 20:00:00+00:00	33	Fear	2	2.0	2
2025-09-28 00:00:00+00:00	37	Fear	2	2.0	2

"""


SUPPORTED_INTERVALS = {"15min", "1h", "2h", "4h", "6h", "1d"}

_FNG_TO_ORD = {
    "Extreme Fear": 1,
    "Fear": 2,
    "Neutral": 3,
    "Greed": 4,
    "Extreme Greed": 5,
}

def get_fng_features(
    start: str,
    end: str,
    interval: Literal["15min","1h","2h","4h","6h","1d"] = "1d",
    lookback_days: int = 3,
    past_only: bool = True,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(f"interval must be one of {sorted(SUPPORTED_INTERVALS)}")

    s = session or requests.Session()
    url = "https://api.alternative.me/fng/"
    resp = s.get(url, params={"limit": 0, "format": "json"}, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    records = payload.get("data", [])
    if not records:
        return pd.DataFrame(columns=[
            "fng_score","fng_label","fng_ordinal",
            "fng_ordinal_smooth","fng_ordinal_smooth_int"
        ])

    df = pd.DataFrame(records)
    ts = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
    ts_norm = ts.dt.normalize()

    df = pd.DataFrame({
        "ts_utc": ts_norm,  # day-level alignment
        "fng_score": pd.to_numeric(df["value"], errors="coerce"),
        "fng_label": df["value_classification"].astype(str),
    }).dropna(subset=["fng_score"]).sort_values("ts_utc").drop_duplicates("ts_utc")

    start_ts = pd.to_datetime(start, utc=True)
    end_ts   = pd.to_datetime(end,   utc=True)
    df = df[(df["ts_utc"] >= start_ts.normalize()) & (df["ts_utc"] <= end_ts.normalize())]
    if df.empty:
        return pd.DataFrame(columns=[
            "fng_score","fng_label","fng_ordinal",
            "fng_ordinal_smooth","fng_ordinal_smooth_int"
        ])

    df["fng_ordinal"] = df["fng_label"].map(_FNG_TO_ORD).fillna(3).astype(int)

    base = df.set_index("ts_utc").sort_index()
    series = base["fng_ordinal"].astype(float)
    if past_only:
        smoothed = series.shift(1).rolling(lookback_days, min_periods=1).mean()
    else:
        smoothed = series.rolling(lookback_days, min_periods=1).mean()

    base["fng_ordinal_smooth"] = smoothed.clip(lower=1, upper=5)
    base["fng_ordinal_smooth_int"] = base["fng_ordinal_smooth"].round().astype("Int64")

    if interval == "1d":
        out = base.copy()
        out.index.name = "ts_utc"  # keep consistent with sub-daily case
    else:
        target_index = pd.date_range(start=start_ts, end=end_ts, freq=interval, tz="UTC")
        out = base.reindex(target_index, method="ffill")
        out.index.name = "ts_utc"

    cols = ["fng_score","fng_label","fng_ordinal","fng_ordinal_smooth","fng_ordinal_smooth_int"]
    return out[cols]
