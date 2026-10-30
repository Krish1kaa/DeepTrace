"""Flag passages of a submitted text that closely match reference passages."""
from .chunk import make_windows, split_sentences


def check_plagiarism(text: str, index, threshold: float = 0.85, k: int = 3) -> list[dict]:
    windows = make_windows(split_sentences(text))
    if not windows:
        return []
    results = index.search(windows, k=k)
    flags = []
    for window, hits in zip(windows, results):
        if hits and hits[0][0] >= threshold:
            score, matched, source = hits[0]
            flags.append(
                {"text": window, "score": score, "matched": matched, "source": source}
            )
    return flags
