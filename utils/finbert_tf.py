import re
from typing import Dict, List

import numpy as np
from transformers import AutoTokenizer, TFBertForSequenceClassification

_FINBERT_REPO = "yiyanghkust/finbert-tone"


def _split_into_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?;])\s+(?=[A-Z(])", text)
    return [p.strip() for p in parts if len(p.strip()) > 3]


def _batch(xs: List[str], n: int) -> List[List[str]]:
    return [xs[i : i + n] for i in range(0, len(xs), n)]


class FinBertTF:
    """FinBERT sentiment scorer (neg/neu/pos) — TensorFlow only."""

    def __init__(
        self,
        model_name: str = _FINBERT_REPO,
        max_length: int = 256,
        batch_size: int = 16,
    ):
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # load native TF weights (no from_pt, no torch)
        self.model = TFBertForSequenceClassification.from_pretrained(model_name)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x, axis=-1, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=-1, keepdims=True)

    def _forward_logits(self, texts: List[str]) -> np.ndarray:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="tf",
        )
        out = self.model(enc)
        return out.logits.numpy()  # shape [B, 3]

    def score(self, text: str) -> Dict[str, float]:
        sents = _split_into_sentences(text)
        if not sents:
            return {"pos": float("nan"), "neg": float("nan"), "neutral": float("nan")}
        lens = np.array([max(1, len(s.split())) for s in sents], dtype=float)
        probs_all = []
        for chunk in _batch(sents, self.batch_size):
            logits = self._forward_logits(chunk)
            probs_all.append(self._softmax(logits))
        probs = np.concatenate(probs_all, axis=0)  # [N,3] -> (neg,neu,pos)
        w = lens / lens.sum()
        agg = (probs * w[:, None]).sum(axis=0)
        return {"pos": float(agg[2]), "neg": float(agg[0]), "neutral": float(agg[1])}
