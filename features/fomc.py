import os
import re
import time
import json
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from urllib.parse import urlparse
import urllib.robotparser as robotparser

from utils.bert_sentiment import FinBert
import requests
from bs4 import BeautifulSoup, Tag
from dateutil import parser as dateparser
import pandas as pd
import tldextract
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

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

def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _write_jsonl(path: Path, objs: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for o in objs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")


def _extract_title(soup: BeautifulSoup) -> Optional[str]:
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    return None


def _extract_last_update(soup: BeautifulSoup) -> Optional[str]:
    txt = soup.get_text(" ", strip=True)
    m = re.search(r"Last Update:\s*([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})", txt)
    return m.group(1) if m else None


def _infer_meeting_dates_from_title(
    title: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    if not title:
        return None, None
    t = title.replace("–", "-")
    m = re.search(r"([A-Za-z/]+)\s+(\d{1,2})-(\d{1,2}),\s*(\d{4})", t)
    if not m:
        return None, None
    month, d1, d2, year = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
    try:
        start = dateparser.parse(f"{month} {d1}, {year}").date().isoformat()
        end = dateparser.parse(f"{month} {d2}, {year}").date().isoformat()
        return start, end
    except Exception:
        return None, None


def _clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "form"]):
        tag.decompose()
    main = (
        soup.find(attrs={"role": "main"})
        or soup.find("main")
        or soup.find(id="article")
    )
    root: Tag = main if isinstance(main, Tag) else soup
    for t in root.find_all("aside"):
        t.decompose()
    text = root.get_text(separator=" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class PageRecord:
    url: str
    status: str
    http_status: Optional[int] = None
    error: Optional[str] = None
    title: Optional[str] = None
    meeting_start: Optional[str] = None
    meeting_end: Optional[str] = None
    released: Optional[str] = None  # keep field for future override
    last_update: Optional[str] = None
    word_count: Optional[int] = None
    checksum: Optional[str] = None
    html_path: Optional[str] = None
    text_path: Optional[str] = None


class Robots:
    def __init__(self, session: requests.Session, ua: str, timeout: int = 15):
        self.session = session
        self.ua = ua
        self.timeout = timeout
        self.cache: Dict[str, robotparser.RobotFileParser] = {}

    def _load(self, root: str) -> robotparser.RobotFileParser:
        # Always prefer HTTPS if present in the URL you’re scraping.
        url = root.rstrip("/") + "/robots.txt"
        rp = robotparser.RobotFileParser()
        try:
            resp = self.session.get(url, timeout=self.timeout, headers={"User-Agent": self.ua})
            if resp.status_code >= 400:
                # Treat missing/forbidden robots as unavailable -> allow
                rp.parse([])  # empty rules = allow everything
            else:
                # robotparser wants an iterable of lines (no trailing newlines required)
                rp.parse(resp.text.splitlines())
        except Exception:
            rp.parse([])  # allow on fetch errors
        return rp

    def allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        root = f"{parsed.scheme}://{parsed.netloc}"
        if root not in self.cache:
            self.cache[root] = self._load(root)
        return self.cache[root].can_fetch(self.ua, url)



def _host(url: str) -> str:
    return tldextract.extract(url).registered_domain


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((requests.RequestException,)),
)
def _fetch(
    session: requests.Session, url: str, timeout: int, ua: str
) -> requests.Response:
    return session.get(url, timeout=timeout, headers={"User-Agent": ua})


DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36 "
    "FOMC-Minutes-Scraper/2.0 (+https://example.org/research)"
)

DEFAULT_TIMEOUT = 30
DEFAULT_DELAY = 0.4


def scrape_fomc_minutes(
    urls: List[str] = DEFAULT_URLS,
    out_dir: str = "fomc_data",
    write_manifests: bool = True,  # write JSONL/CSV + raw HTML/TXT
    manifest_format: str = "both",  # "jsonl" | "csv" | "both"
    respect_robots: bool = True,
    user_agent: str = DEFAULT_USER_AGENT,
    timeout: int = DEFAULT_TIMEOUT,
    per_host_delay: float = DEFAULT_DELAY,
    cache_dir: str = ".fomc_cache",  # disk cache for html + sentiment
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Scrape FOMC minutes pages and (optionally) run FinBERT sentiment on the cleaned text.
    Returns (pages_df, sentiment_df). Writes outputs if write_manifests=True.
    """

    # --- setup
    session = requests.Session()
    robots = Robots(session, user_agent)
    out_path, cache_path = Path(out_dir), Path(cache_dir)
    _ensure_dir(out_path)
    _ensure_dir(cache_path / "http")
    _ensure_dir(cache_path / "sentiment")

    # in-memory caches (this run)
    http_mem: Dict[str, str] = {}
    sent_mem: Dict[str, Dict[str, float]] = {}

    last_host = None
    records: List[PageRecord] = []
    finbert = FinBert()
    sentiments: List[Dict[str, str | float]] = []

    for url in urls:
        # robots
        if respect_robots and not robots.allowed(url):
            records.append(PageRecord(url=url, status="blocked_by_robots"))
            continue

        # polite delay per host
        host = _host(url)
        if last_host == host and per_host_delay > 0:
            time.sleep(per_host_delay)
        last_host = host

        # --- fetch with cache (disk first, then mem fallback)
        url_key = _sha256_text(url)
        http_cache_file = cache_path / "http" / f"{url_key}.html"

        try:
            if url in http_mem:
                html = http_mem[url]
                http_status = 200
            elif http_cache_file.exists():
                html = http_cache_file.read_text(encoding="utf-8", errors="ignore")
                http_status = 200
                http_mem[url] = html
            else:
                resp = _fetch(session, url, timeout, user_agent)
                http_status = resp.status_code
                if resp.status_code != 200:
                    records.append(
                        PageRecord(
                            url=url,
                            status="http_error",
                            http_status=resp.status_code,
                            error=f"HTTP {resp.status_code}",
                        )
                    )
                    continue
                html = resp.text
                _write_text(http_cache_file, html)
                http_mem[url] = html

            # --- parse + save minimal artifacts
            checksum = _sha256_bytes(html.encode("utf-8"))
            soup = BeautifulSoup(html, "html.parser")
            title = _extract_title(soup)
            last_update = _extract_last_update(soup)
            text = _clean_text(html)
            wc = len(text.split())
            start, end = _infer_meeting_dates_from_title(title)

            base = os.path.splitext(os.path.basename(url))[0]
            html_path = out_path / f"{base}.html"
            txt_path = out_path / f"{base}.txt"
            _write_text(html_path, html)
            _write_text(txt_path, text)

            rec = PageRecord(
                url=url,
                status="ok",
                http_status=http_status,
                title=title,
                meeting_start=start,
                meeting_end=end,
                last_update=last_update,
                word_count=wc,
                checksum=checksum,
                html_path=str(html_path),
                text_path=str(txt_path),
            )
            records.append(rec)

            # --- sentiment with cache (by text checksum)
            if enable_sentiment:
                text_key = _sha256_text(text or "")
                sent_cache_file = cache_path / "sentiment" / f"{text_key}.json"
                if text_key in sent_mem:
                    s = sent_mem[text_key]
                elif sent_cache_file.exists():
                    s = json.loads(sent_cache_file.read_text(encoding="utf-8"))
                    sent_mem[text_key] = s
                else:
                    s = (
                        finbert.score(text)
                        if text
                        else {
                            "pos": float("nan"),
                            "neg": float("nan"),
                            "neutral": float("nan"),
                        }
                    )
                    sent_mem[text_key] = s
                    sent_cache_file.write_text(json.dumps(s), encoding="utf-8")
                sentiments.append(
                    {
                        "url": url,
                        "title": title or "",
                        "checksum": checksum,
                        "finbert_pos": s["pos"],
                        "finbert_neg": s["neg"],
                        "finbert_neutral": s["neutral"],
                    }
                )

        except Exception as e:
            records.append(PageRecord(url=url, status="error", error=str(e)))

    # --- results
    pages_df = pd.DataFrame([asdict(r) for r in records])
    if not pages_df.empty and "url" in pages_df.columns:
        pages_df = pages_df.sort_values("url")

    if enable_sentiment:
        sent_df = pd.DataFrame(sentiments)
        if not sent_df.empty and "url" in sent_df.columns:
            sent_df = sent_df.sort_values("url")
    else:
        sent_df = None

    if write_manifests:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        if sent_df is not None and not sent_df.empty:
            sent_df.to_csv(out_path / f"fomc_sentiment_summary_{ts}.csv", index=False)
        if manifest_format in ("jsonl", "both"):
            _write_jsonl(out_path / f"manifest_{ts}.jsonl", [asdict(r) for r in records])
        if manifest_format in ("csv", "both") and not pages_df.empty:
            pages_df.to_csv(out_path / f"manifest_{ts}.csv", index=False)


    return pages_df, sent_df
