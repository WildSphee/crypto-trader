from dataclasses import dataclass
from typing import List, Dict, Iterable, Optional, Tuple

import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


LABEL_MAP = {
    "positive": "pos",
    "negative": "neg",
    "neutral": "neutral",
    "LABEL_0": "neg",
    "LABEL_1": "neutral",
    "LABEL_2": "pos",
}


def _normalize_scores(raw: Dict[str, float]) -> Dict[str, float]:
    pos = float(raw.get("pos", 0.0))
    neg = float(raw.get("neg", 0.0))
    neu = float(raw.get("neutral", 0.0))
    total = pos + neg + neu
    if total == 0:
        return {"pos": 0.0, "neg": 0.0, "neutral": 1.0}
    return {"pos": pos / total, "neg": neg / total, "neutral": neu / total}


def _split_paragraphs(text: str) -> List[str]:
    """Split into reasonably sized paragraphs for FinBERT scoring."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paras = re.split(r"\n{2,}", text)
    return [re.sub(r"\s+", " ", p).strip() for p in paras if p.strip()]


@dataclass
class FinBertResult:
    pos: float
    neg: float
    neutral: float

    def to_dict(self) -> Dict[str, float]:
        return {"pos": self.pos, "neg": self.neg, "neutral": self.neutral}


class FinBert:
    """
    Wrapper for the FinBERT model ('yiyanghkust/finbert-tone') using Hugging Face Transformers.
    """

    def __init__(
        self,
        model_name: str = "yiyanghkust/finbert-tone",
        cache_dir: Optional[str] = None,
        device: Optional[int] = None,
        max_length: int = 256,
        batch_size: int = 16,
    ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir)
        self.pipe = pipeline(
            task="text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            top_k=None,
            truncation=True,
            max_length=max_length,
            return_all_scores=True,
            device=device if device is not None else None,
        )
        self.batch_size = batch_size

    def score(self, text: str) -> Dict[str, float]:
        """Return overall FinBERT sentiment for a text."""
        if not text.strip():
            return {"pos": float("nan"), "neg": float("nan"), "neutral": float("nan")}
        result = self.pipe(text[:4000])
        return _normalize_scores(self._convert_pipe_scores(result))

    def batch_score(self, texts: Iterable[str]) -> List[Dict[str, float]]:
        """Batch score multiple texts."""
        texts = list(texts)
        results: List[Dict[str, float]] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = [t[:4000] for t in texts[i : i + self.batch_size]]
            out_batch = self.pipe(chunk)
            # HF may return list of dicts or list[list[dict]]
            if isinstance(out_batch[0], dict):
                out_batch = [[x] for x in out_batch]
            for out in out_batch:
                results.append(_normalize_scores(self._convert_pipe_scores(out)))
        return results

    def score_paragraphs(
        self,
        text: str,
        min_words: int = 20,
        merge_short: bool = True,
    ) -> Tuple[List[str], List[Dict[str, float]]]:
        """Split a text into paragraphs and compute FinBERT sentiment for each."""
        paras = _split_paragraphs(text)
        if merge_short:
            merged, buf, count = [], "", 0
            for p in paras:
                wc = len(p.split())
                if wc >= min_words:
                    if buf:
                        merged.append(buf.strip())
                        buf, count = "", 0
                    merged.append(p)
                else:
                    buf = (buf + " " + p).strip()
                    count += wc
                    if count >= min_words:
                        merged.append(buf)
                        buf, count = "", 0
            if buf:
                merged.append(buf)
            paras = merged

        scores = self.batch_score(paras)
        return paras, scores

    @staticmethod
    def _convert_pipe_scores(pipe_out) -> Dict[str, float]:
        """Convert HF pipeline output into a {'pos','neg','neutral'} dict."""
        scores = {"pos": 0.0, "neg": 0.0, "neutral": 0.0}
        if isinstance(pipe_out, list):
            for item in pipe_out:
                label = str(item.get("label"))
                key = LABEL_MAP.get(label)
                if key:
                    scores[key] = float(item.get("score", 0.0))
        elif isinstance(pipe_out, dict) and "label" in pipe_out:
            key = LABEL_MAP.get(str(pipe_out["label"]))
            if key:
                scores[key] = float(pipe_out.get("score", 0.0))
        return scores
