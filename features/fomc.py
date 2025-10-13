import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag

from utils.finbert_tf import FinBertTF

DEFAULT_URLS = [
    # 2025
    "https://www.federalreserve.gov/monetarypolicy/fomcminutes20250129.htm",
    "https://www.federalreserve.gov/monetarypolicy/fomcminutes20250319.htm",
    "https://www.federalreserve.gov/monetarypolicy/fomcminutes20250507.htm",
    "https://www.federalreserve.gov/monetarypolicy/fomcminutes20250618.htm",
    "https://www.federalreserve.gov/monetarypolicy/fomcminutes20250730.htm",
    # 2024
    "https://www.federalreserve.gov/monetarypolicy/fomcminutes20240131.htm",
    "https://www.federalreserve.gov/monetarypolicy/fomcminutes20240320.htm",
    "https://www.federalreserve.gov/monetarypolicy/fomcminutes20240501.htm",
    "https://www.federalreserve.gov/monetarypolicy/fomcminutes20240612.htm",
    "https://www.federalreserve.gov/monetarypolicy/fomcminutes20240731.htm",
    "https://www.federalreserve.gov/monetarypolicy/fomcminutes20240918.htm",
    "https://www.federalreserve.gov/monetarypolicy/fomcminutes20241107.htm",
    "https://www.federalreserve.gov/monetarypolicy/fomcminutes20241218.htm",
    # 2023
    "https://www.federalreserve.gov/monetarypolicy/fomcminutes20230201.htm",
    "https://www.federalreserve.gov/monetarypolicy/fomcminutes20230322.htm",
    "https://www.federalreserve.gov/monetarypolicy/fomcminutes20230503.htm",
    "https://www.federalreserve.gov/monetarypolicy/fomcminutes20230614.htm",
    "https://www.federalreserve.gov/monetarypolicy/fomcminutes20230726.htm",
    "https://www.federalreserve.gov/monetarypolicy/fomcminutes20230920.htm",
    "https://www.federalreserve.gov/monetarypolicy/fomcminutes20231101.htm",
    "https://www.federalreserve.gov/monetarypolicy/fomcminutes20231213.htm",
]

# ---------- helpers


def _sha256_text(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _clean_text(html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "form"]):
        tag.decompose()
    main = soup.find(attrs={"role": "main"}) or soup.find("main") or soup
    root: Tag = main if isinstance(main, Tag) else soup
    for t in root.find_all("aside"):
        t.decompose()
    text = root.get_text(separator=" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


def _title(soup: BeautifulSoup) -> Optional[str]:
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    return None


def _fetch_html_with_cache(url: str, cache_dir: Path, timeout: int) -> Tuple[str, str]:
    """
    Returns (html, source) where source is "cache" or "network".
    """
    key = _sha256_text(url)
    html_cache = cache_dir / "http" / f"{key}.html"
    if html_cache.exists():
        return html_cache.read_text(encoding="utf-8", errors="ignore"), "cache"
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    html = resp.text
    html_cache.write_text(html, encoding="utf-8")
    return html, "network"


def _score_with_cache(text: str, fb: FinBertTF, cache_dir: Path) -> Dict[str, float]:
    key = _sha256_text(text or "")
    sc_path = cache_dir / "sentiment" / f"{key}.json"
    if sc_path.exists():
        return json.loads(sc_path.read_text(encoding="utf-8"))
    s = (
        fb.score(text)
        if text
        else {"pos": float("nan"), "neg": float("nan"), "neutral": float("nan")}
    )
    sc_path.write_text(json.dumps(s), encoding="utf-8")
    return s


@dataclass
class PageRow:
    url: str
    status: str  # "ok", "cached", "http_error", "parse_error"
    http_status: Optional[int]
    title: Optional[str]
    word_count: Optional[int]
    html_checksum: Optional[str]
    source: Optional[str]  # "network" or "cache"
    # html_path: Optional[str]
    # text_path: Optional[str]


def scrape_fomc_minutes(
    urls: List[str] = DEFAULT_URLS,
    *,
    cache_dir: str = ".fomc_cache",
    enable_sentiment: bool = True,
    timeout: int = 30,
):
    """
    Minimal scraper with deterministic caching.
    - HTTP cache at {cache_dir}/http/<sha>.html
    - Sentiment cache at {cache_dir}/sentiment/<sha>.json
    - Writes cleaned TXT + original HTML to {out_dir}/ for convenience.
    Returns: (pages_df, sentiment_df)
    """
    cache = Path(cache_dir)
    _ensure_dir(cache / "http")
    _ensure_dir(cache / "sentiment")

    fb = FinBertTF() if enable_sentiment else None

    page_rows: List[PageRow] = []
    sentiments: List[Dict[str, float | str]] = []

    for url in urls:
        try:
            html, source = _fetch_html_with_cache(url, cache, timeout)
            soup = BeautifulSoup(html, "html.parser")
            title = _title(soup)
            text = _clean_text(html)
            wc = len(text.split()) if text else 0
            checksum = _sha256_text(html)

            page_rows.append(
                PageRow(
                    url=url,
                    status="cached" if source == "cache" else "ok",
                    http_status=200,
                    title=title,
                    word_count=wc,
                    html_checksum=checksum,
                    source=source,
                )
            )

            if enable_sentiment and fb is not None:
                s = _score_with_cache(text, fb, cache)
                sentiments.append(
                    {
                        "url": url,
                        "title": title or "",
                        "checksum": checksum,
                        "finbert_pos": s["pos"],
                        "finbert_neg": s["neg"],
                        "finbert_neutral": s["neutral"],
                        "source": "cache"
                        if _sha256_text(text or "") in (Path(p).stem for p in [])
                        else None,  # placeholder; not used
                    }
                )

        except requests.HTTPError as e:
            page_rows.append(
                PageRow(
                    url=url,
                    status="http_error",
                    http_status=e.response.status_code if e.response else None,
                    title=None,
                    word_count=None,
                    html_checksum=None,
                    source=None,
                    html_path=None,
                    text_path=None,
                )
            )
        except Exception:
            page_rows.append(
                PageRow(
                    url=url,
                    status="parse_error",
                    http_status=None,
                    title=None,
                    word_count=None,
                    html_checksum=None,
                    source=None,
                    html_path=None,
                    text_path=None,
                )
            )

    pages_df = pd.DataFrame([asdict(r) for r in page_rows]).sort_values("url")
    sent_df = pd.DataFrame(sentiments).sort_values("url") if enable_sentiment else None
    return pages_df, sent_df


_INTERVAL_MAP: Dict[str, str] = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1H",
    "2h": "2H",
    "4h": "4H",
    "6h": "6H",
    "8h": "8H",
    "12h": "12H",
    "1d": "1D",
}


def _pd_freq(interval: str) -> str:
    if interval not in _INTERVAL_MAP:
        raise ValueError(f"unsupported interval: {interval}")
    return _INTERVAL_MAP[interval]


def _meeting_ts_from_url(u: str) -> Optional[pd.Timestamp]:
    # e.g. .../fomcminutes20240731.htm -> 2024-07-31 00:00:00+00:00
    m = re.search(r"fomcminutes(\d{8})\.htm", u)
    if not m:
        return None
    return pd.to_datetime(m.group(1), format="%Y%m%d", utc=True)


def _default_metrics() -> List[str]:
    # all numeric; composite explained below
    return [
        "finbert_pos",
        "finbert_neg",
        "finbert_neutral",
        "fomc_sentiment_composite",  # pos - neg
        "fomc_sentiment_polarity",  # (pos - neg) / (pos + neg)
        "word_count",
    ]


def get_fomc_sentiment(
    start: str,
    end: str,
    interval: Literal[
        "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"
    ] = "1d",
    metrics: Optional[List[str]] = None,
    lookback_days: int = 3,
    *,
    smooth_kind: Literal["ema", "sma", "median"] = "ema",
    smooth_span: int = 5,  # EMA span (≈ smoothing strength)
    sma_window: int = 5,  # for smooth_kind="sma"
    median_window: int = 5,  # for smooth_kind="median"
    cache_dir: str = ".fomc_cache",
    enable_sentiment: bool = True,
) -> pd.DataFrame:
    """
    Returns a numeric-only DataFrame with index name 'ts_utc' and columns ending in `_smooth`.
    - Each FOMC meeting creates an 'event' value (FinBERT scores and composites).
    - Those values are carried forward for `lookback_days`, then set to NaN until next event.
    - The full series is resampled to `interval` and smoothed.

    Example columns:
        finbert_pos_smooth, finbert_neg_smooth, finbert_neutral_smooth,
        fomc_sentiment_composite_smooth, fomc_sentiment_polarity_smooth, word_count_smooth
    """
    freq = _pd_freq(interval)

    pages_df, sent_df = scrape_fomc_minutes(
        cache_dir=cache_dir,
        enable_sentiment=enable_sentiment,
    )
    if sent_df is None or sent_df.empty:
        raise RuntimeError(
            "Sentiment is disabled or empty. Enable it in scrape_fomc_minutes()."
        )

    # 2) derive event timestamps and numeric payloads
    ev = sent_df.copy()
    ev["ts_utc"] = ev["url"].map(_meeting_ts_from_url)
    ev = ev.dropna(subset=["ts_utc"]).sort_values("ts_utc")

    # attach word_count from pages_df for an extra numeric feature
    wc = pages_df[["url", "word_count"]].copy()
    ev = ev.merge(wc, on="url", how="left")

    # clean numeric types
    for c in ["finbert_pos", "finbert_neg", "finbert_neutral", "word_count"]:
        if c in ev.columns:
            ev[c] = pd.to_numeric(ev[c], errors="coerce")

    # composites (all numeric)
    ev["fomc_sentiment_composite"] = ev["finbert_pos"] - ev["finbert_neg"]
    denom = (ev["finbert_pos"] + ev["finbert_neg"]).replace(0, np.nan)
    ev["fomc_sentiment_polarity"] = ev["fomc_sentiment_composite"] / denom

    # select metrics
    keep = metrics if metrics is not None else _default_metrics()
    missing = [m for m in keep if m not in ev.columns]
    if missing:
        raise ValueError(f"Requested metrics not available: {missing}")

    ev = ev[["ts_utc"] + keep].set_index("ts_utc").sort_index()

    # 3) build uniform timeline
    idx = pd.date_range(
        start=pd.to_datetime(start, utc=True),
        end=pd.to_datetime(end, utc=True),
        freq=freq,
    )
    idx.name = "ts_utc"

    # we’ll forward-fill event values but only within lookback_days of the last event
    # compute "last event time at each idx"
    event_time_series = pd.Series(ev.index, index=ev.index)

    # 2) forward-fill the last event timestamp onto the uniform timeline
    last_event_time = event_time_series.reindex(
        idx, method="ffill"
    )  # <-- Series, not tuple

    # 3) forward-fill the metric values onto the same timeline
    ff = ev.reindex(idx, method="ffill")

    # 4) compute age (in days) since the last event time
    idx_series = pd.Series(pd.DatetimeIndex(idx), index=idx)
    age_days = (idx_series - last_event_time).dt.total_seconds() / 86400.0  # NaT -> NaN

    # 5) mask out anything older than lookback_days
    mask_old = age_days > float(lookback_days)
    ff.loc[mask_old.values, :] = np.nan

    # 4) smoothing
    if smooth_kind == "ema":
        smoothed = ff.ewm(span=smooth_span, min_periods=1, adjust=False).mean()
    elif smooth_kind == "sma":
        smoothed = ff.rolling(window=sma_window, min_periods=1).mean()
    elif smooth_kind == "median":
        smoothed = ff.rolling(window=median_window, min_periods=1).median()
    else:
        raise ValueError("smooth_kind must be one of: 'ema', 'sma', 'median'")

    # 5) numeric-only, proper names
    smoothed = smoothed.apply(pd.to_numeric, errors="coerce")
    smoothed.index.name = "ts_utc"
    smoothed.columns = [f"{c}_smooth" for c in smoothed.columns]

    return smoothed
