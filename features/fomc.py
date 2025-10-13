import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


# ---------- cache-aware fetch/sentiment


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


# ---------- datamodel


@dataclass
class PageRow:
    url: str
    status: str  # "ok", "cached", "http_error", "parse_error"
    http_status: Optional[int]
    title: Optional[str]
    word_count: Optional[int]
    html_checksum: Optional[str]
    source: Optional[str]  # "network" or "cache"
    html_path: Optional[str]
    text_path: Optional[str]


# ---------- main API (bare bones)


def scrape_fomc_minutes(
    urls: List[str] = DEFAULT_URLS,
    *,
    out_dir: str = "fomc_data",
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
    out = Path(out_dir)
    cache = Path(cache_dir)
    _ensure_dir(out)
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

            # write artifacts (overwrite; idempotent)
            base = os.path.splitext(os.path.basename(url))[0]
            html_path = out / f"{base}.html"
            txt_path = out / f"{base}.txt"
            html_path.write_text(html, encoding="utf-8")
            txt_path.write_text(text, encoding="utf-8")

            page_rows.append(
                PageRow(
                    url=url,
                    status="cached" if source == "cache" else "ok",
                    http_status=200,
                    title=title,
                    word_count=wc,
                    html_checksum=checksum,
                    source=source,
                    html_path=str(html_path),
                    text_path=str(txt_path),
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
