"""RoBERTa-based classifier that estimates whether text is machine-generated."""
import os

from transformers import pipeline

# Default checkpoint: a public RoBERTa output detector. Override with your own
# fine-tuned checkpoint via the DEEPTRACE_DETECTOR_MODEL environment variable.
DEFAULT_MODEL = os.getenv(
    "DEEPTRACE_DETECTOR_MODEL", "openai-community/roberta-base-openai-detector"
)


class AITextDetector:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: int = -1):
        self.pipe = pipeline(
            "text-classification", model=model_name, device=device
        )

    def score(self, text: str) -> float:
        """Return the probability (0 to 1) that the text is AI-generated.

        Inputs longer than the model's 512-token limit are truncated.
        """
        result = self.pipe(text, truncation=True, max_length=512)[0]
        label = result["label"].lower()
        confidence = float(result["score"])
        return confidence if label == "fake" else 1.0 - confidence
