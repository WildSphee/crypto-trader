"""
news_to_features.py
-------------------
Turn timestamped news headlines into trainable features aligned to BTC price bars.

Inputs (CSV, UTF-8):
1) headlines.csv with columns:
   - timestamp   : ISO8601 string in UTC (e.g., 2025-10-03T13:45:00Z)
   - source      : news source name (e.g., AP, Reuters, People)
   - title       : headline text

2) prices.csv with columns:
   - timestamp   : ISO8601 string in UTC (bar end time)
   - close       : BTC close price for that bar

Config (edit in code or pass CLI args):
   --bar_freq        Resample frequency for features (default: 5min)
   --lookbacks       Comma-separated lookbacks in minutes to aggregate headlines (default: 30,60,180)
   --horizon_min     Forecast horizon in minutes for target label (default: 360 i.e., 6h)
   --min_latency_s   Seconds to delay headline effect (simulate feed latency; default: 120)
   --out             Output CSV path (default: training_dataset.csv)

Example:
   python news_to_features.py --headlines headlines.csv --prices prices.csv --out training_dataset.csv

Notes:
 - This script uses a tiny lexicon-based sentiment + topic flags so it runs with no external deps.
 - Replace `compute_sentiment()` with any model you prefer when available.
 - Uses strictly pre-t timestamps to avoid leakage; headline timestamps are shifted by `min_latency_s`.
"""

import argparse
import re
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def source_weight(source: str) -> float:
    s = (source or "").strip().lower()
    return SOURCE_WEIGHTS.get(s, SOURCE_WEIGHTS.get("unknown", 0.5))


@dataclass
class SentimentResult:
    score: float  # in [-1, 1]
    intensity: float  # approx intensity [0,1]


def compute_sentiment(text: str) -> SentimentResult:
    """
    Tiny lexicon sentiment: (#pos - #neg) / (#pos + #neg + 1)
    Intensity boosts with exclamations, all-caps tokens, and intensifiers/deintensifiers.
    """
    t = normalize_text(text)
    tokens = re.findall(r"[a-z]+", t)

    pos = sum(tok in POS_WORDS for tok in tokens)
    neg = sum(tok in NEG_WORDS for tok in tokens)

    base = (pos - neg) / (pos + neg + 1.0)
    exclaims = text.count("!")

    caps_tokens = sum(1 for w in re.findall(r"\b[A-Z]{3,}\b", text))
    intens = sum(tok in INTENSIFIERS for tok in tokens)
    deintens = sum(tok in DEINTENSIFIERS for tok in tokens)

    intensity = min(
        1.0,
        0.2 * exclaims
        + 0.1 * caps_tokens
        + 0.15 * intens
        - 0.1 * deintens
        + 0.3 * abs(base),
    )
    # squash base by intensity (more intense -> closer to -1/+1)
    score = float(np.tanh(1.5 * base * (1 + intensity)))
    return SentimentResult(score=score, intensity=float(intensity))


def topic_flags(text: str) -> dict:
    t = normalize_text(text)
    return {k: int(bool(re.search(pat, t))) for k, pat in TOPIC_KEYWORDS.items()}


def novelty_against_window(embeddings_window: List[np.ndarray], v: np.ndarray) -> float:
    if len(embeddings_window) == 0:
        return 1.0
    mu = np.mean(embeddings_window, axis=0)
    num = float(np.dot(v, mu))
    den = float(np.linalg.norm(v) * np.linalg.norm(mu) + 1e-9)
    cos = num / den if den != 0 else 0.0
    return float(1.0 - cos)


def tiny_embedding(text: str, dim: int = 32) -> np.ndarray:
    """Deterministic hashing into a small vector; stand-in for real embeddings."""
    rng = np.random.RandomState(abs(hash(text)) % (2**32))
    v = rng.normal(size=dim)
    v /= np.linalg.norm(v) + 1e-9
    return v.astype(np.float32)


def build_headline_frame(dfH: pd.DataFrame, min_latency_s: int) -> pd.DataFrame:
    # basic cleaning + per-headline features
    df = dfH.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp_effective"] = df["timestamp"] + pd.to_timedelta(
        min_latency_s, unit="s"
    )
    df["source_weight"] = df["source"].map(source_weight)
    sent = df["title"].apply(compute_sentiment)
    df["sentiment"] = [s.score for s in sent]
    df["sent_intensity"] = [s.intensity for s in sent]
    df["len_tokens"] = (
        df["title"].fillna("").apply(lambda s: len(re.findall(r"[A-Za-z]+", s)))
    )
    # topics
    flags = df["title"].apply(topic_flags)
    for k in TOPIC_KEYWORDS.keys():
        df[k] = flags.apply(lambda d: d[k])
    # novelty via tiny embedding over rolling 24h window (approximation)
    df = df.sort_values("timestamp_effective")
    embs = []
    nov = []
    cutoff = pd.Timedelta("24H")
    for _, row in df.iterrows():
        t = row["timestamp_effective"]
        # drop old
        while len(embs) and (t - embs[0][0]) > cutoff:
            embs.pop(0)
        v = tiny_embedding(row["title"])
        window_vecs = [v2 for (_, v2) in embs]
        nov.append(novelty_against_window(window_vecs, v))
        embs.append((t, v))
    df["novelty_24h"] = nov
    return df


def aggregate_to_bars(
    dfH: pd.DataFrame, bar_index: pd.DatetimeIndex, lookbacks_min: List[int]
) -> pd.DataFrame:
    # For each lookback window, compute aggregates using headlines with timestamp_effective <= bar time
    out = pd.DataFrame(index=bar_index)
    # headline stream indexed by effective time
    h = dfH.set_index("timestamp_effective").sort_index()
    # Precompute 1-min bins to speed up rolling ops
    h1 = (
        h.assign(count=1)
        .resample("1min")
        .agg(
            {
                "sentiment": "mean",
                "sent_intensity": "mean",
                "novelty_24h": "mean",
                "count": "sum",
                "source_weight": "sum",
                "is_trump": "sum",
                "is_geopolitical": "sum",
                "is_policy": "sum",
                "is_crime_cartels": "sum",
                "is_macro": "sum",
                "is_crypto_specific": "sum",
            }
        )
    ).fillna(0.0)

    for W in lookbacks_min:
        win = f"{W}min"
        roll = h1.rolling(win, min_periods=1).sum()
        # averages based on sums (avoid divide by zero)
        count = roll["count"].reindex(bar_index, method="pad").fillna(0.0)
        sent_sum = (
            h1["sentiment"]
            .rolling(win, min_periods=1)
            .sum()
            .reindex(bar_index, method="pad")
            .fillna(0.0)
        )
        sent_mean = (sent_sum / (count.replace(0, np.nan))).fillna(0.0)
        nov_mean = (
            h1["novelty_24h"]
            .rolling(win, min_periods=1)
            .mean()
            .reindex(bar_index, method="pad")
            .fillna(0.0)
        )
        sw_sum = roll["source_weight"].reindex(bar_index, method="pad").fillna(0.0)

        out[f"sent_mean_{W}"] = sent_mean
        out[f"novelty_mean_{W}"] = nov_mean
        out[f"volume_{W}"] = count
        out[f"volume_w_{W}"] = sw_sum
        for k in [
            "is_trump",
            "is_geopolitical",
            "is_policy",
            "is_crime_cartels",
            "is_macro",
            "is_crypto_specific",
        ]:
            out[f"{k}_count_{W}"] = roll[k].reindex(bar_index, method="pad").fillna(0.0)

        # EWMA of sentiment within W using half-life = W/2 minutes
        halflife = max(1, W // 2)
        # construct a minute series of sentiment mean, then ewma, then align
        s_min = (h1["sentiment"].resample("1min").mean()).fillna(0.0)
        ew = s_min.ewm(halflife=f"{halflife}min", times=s_min.index).mean()
        out[f"sent_ewma_hl{halflife}_{W}"] = ew.reindex(bar_index, method="pad").fillna(
            0.0
        )

    return out


def make_targets(prices: pd.DataFrame, horizon_min: int) -> pd.DataFrame:
    p = prices.copy()
    p = p.sort_index()
    # log returns
    p["logp"] = np.log(p["close"].replace(0, np.nan)).ffill()
    future = p["logp"].shift(
        -horizon_min // int(pd.Timedelta("1min").total_seconds() / 60)
    )
    p["fwd_logret"] = future - p["logp"]
    p["fwd_ret"] = np.exp(p["fwd_logret"]) - 1.0
    # alert-style label: move >= +3% or <= -3%
    thr = 0.03
    y = pd.Series(0, index=p.index, dtype="int8")
    y[p["fwd_ret"] >= thr] = 1
    y[p["fwd_ret"] <= -thr] = -1
    return pd.DataFrame(
        {
            "y_klass_3pct_{}m".format(horizon_min): y,
            "y_reg_fwdret_{}m".format(horizon_min): p["fwd_ret"],
        }
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headlines", type=str, required=True)
    ap.add_argument("--prices", type=str, required=True)
    ap.add_argument("--bar_freq", type=str, default="5min")
    ap.add_argument("--lookbacks", type=str, default="30,60,180")
    ap.add_argument("--horizon_min", type=int, default=360)
    ap.add_argument("--min_latency_s", type=int, default=120)
    ap.add_argument("--out", type=str, default="training_dataset.csv")
    args = ap.parse_args()

    lookbacks_min = [int(x) for x in args.lookbacks.split(",")]

    # Load inputs
    dfH = pd.read_csv(args.headlines)
    dfP = pd.read_csv(args.prices)

    # Prices
    dfP["timestamp"] = pd.to_datetime(dfP["timestamp"], utc=True)
    dfP = dfP.sort_values("timestamp").set_index("timestamp")
    # resample to bar_freq (bar end)
    dfP = dfP.resample(args.bar_freq).last().dropna(subset=["close"])

    # Headlines -> per-headline features
    dfH = build_headline_frame(dfH, min_latency_s=args.min_latency_s)

    # Build bar index from prices
    bar_index = dfP.index

    # Aggregate headline features into bars
    agg = aggregate_to_bars(dfH, bar_index, lookbacks_min)

    # Join with prices (controls)
    feat = agg.join(dfP[["close"]]).copy()
    # add simple market controls (lagged returns & realized vol)
    feat["ret_1"] = feat["close"].pct_change(1)
    feat["ret_5"] = feat["close"].pct_change(5)
    feat["ret_12"] = feat["close"].pct_change(12)
    feat["rv_30"] = feat["ret_1"].rolling(30, min_periods=5).std().fillna(0.0)

    # Targets
    targets = make_targets(dfP[["close"]], horizon_min=args.horizon_min)

    # Final dataset (drop last rows with NaN targets due to horizon)
    out = feat.join(targets).dropna(subset=targets.columns.tolist())

    out.to_csv(args.out, index_label="bar_end")
    print(f"Wrote {args.out} with shape {out.shape}")
    print("Columns:", list(out.columns))


if __name__ == "__main__":
    main()
