"""Sentence splitting and sliding-window passage construction."""
import re

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str, min_words: int = 5) -> list[str]:
    """Split text into sentences, dropping fragments shorter than min_words."""
    sentences = [s.strip() for s in _SENT_SPLIT.split(text.strip())]
    return [s for s in sentences if len(s.split()) >= min_words]


def make_windows(sentences: list[str], size: int = 2, stride: int = 1) -> list[str]:
    """Group consecutive sentences into overlapping passages for matching."""
    if not sentences:
        return []
    if len(sentences) <= size:
        return [" ".join(sentences)]
    return [
        " ".join(sentences[i : i + size])
        for i in range(0, len(sentences) - size + 1, stride)
    ]
