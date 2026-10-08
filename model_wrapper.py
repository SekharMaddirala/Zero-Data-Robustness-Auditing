"""Model abstraction: Hugging Face sequence classifiers and an offline mock model."""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from metamorphic_nlp_auditor.config import MOCK_PROFILES, MockProfile, ModelConfig


@dataclass(frozen=True)
class PredictionBatch:
    """Batched model output.

    Attributes:
        logits: Unnormalised scores, shape ``(n, k)``.
        probabilities: Softmax of ``logits``, shape ``(n, k)``.
        labels: Predicted class indices (argmax), shape ``(n,)``.
        confidences: Softmax probability of the predicted class, shape ``(n,)``.
        label_names: Names of the ``k`` classes.
    """

    logits: np.ndarray
    probabilities: np.ndarray
    labels: np.ndarray
    confidences: np.ndarray
    label_names: tuple[str, ...]

    def __len__(self) -> int:
        return int(self.logits.shape[0])

    @classmethod
    def from_logits(cls, logits: np.ndarray, label_names: Sequence[str]) -> "PredictionBatch":
        """Build a batch from raw logits using a numerically stable softmax."""
        arr = np.asarray(logits, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"logits must be 2-D, got shape {arr.shape}.")
        shifted = arr - arr.max(axis=1, keepdims=True) if arr.shape[0] else arr
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True) if arr.shape[0] else exp
        labels = probs.argmax(axis=1) if arr.shape[0] else np.zeros(0, dtype=np.int64)
        conf = probs.max(axis=1) if arr.shape[0] else np.zeros(0)
        return cls(arr, probs, labels.astype(np.int64), conf, tuple(label_names))


class BaseNLPModel(ABC):
    """Interface every audited classifier must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique model name."""

    @property
    @abstractmethod
    def backend(self) -> str:
        """Backend identifier (``"huggingface"`` or ``"mock"``)."""

    @property
    @abstractmethod
    def label_names(self) -> tuple[str, ...]:
        """Class names, indexed by label id."""

    @abstractmethod
    def predict(self, texts: Sequence[str]) -> PredictionBatch:
        """Classify ``texts`` and return logits, probabilities, labels and confidences."""


class HuggingFaceNLPModel(BaseNLPModel):
    """Wrapper around ``AutoModelForSequenceClassification`` / ``AutoTokenizer``.

    ``torch`` and ``transformers`` are imported lazily so that the rest of the package
    (and the mock backend) works without them.
    """

    def __init__(self, config: ModelConfig) -> None:
        if config.hf_model_id is None:
            raise ValueError("hf_model_id is required for HuggingFaceNLPModel.")
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "The 'huggingface' backend needs `torch` and `transformers` "
                "(pip install torch transformers), or set ExperimentConfig.offline_mode=True."
            ) from exc
        self._torch = torch
        self._config = config
        self._device = self._resolve_device(config.device)
        self._tokenizer = AutoTokenizer.from_pretrained(
            config.hf_model_id, local_files_only=config.local_files_only)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            config.hf_model_id, local_files_only=config.local_files_only)
        self._model.to(self._device)
        self._model.eval()
        id2label = self._model.config.id2label
        self._label_names = tuple(str(id2label[i]) for i in range(self._model.config.num_labels))

    def _resolve_device(self, requested: str) -> str:
        if requested != "auto":
            return requested
        if self._torch.cuda.is_available():
            return "cuda"
        mps = getattr(self._torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def backend(self) -> str:
        return "huggingface"

    @property
    def label_names(self) -> tuple[str, ...]:
        return self._label_names

    def predict(self, texts: Sequence[str]) -> PredictionBatch:
        if len(texts) == 0:
            return PredictionBatch.from_logits(np.zeros((0, len(self._label_names))), self._label_names)
        chunks: list[np.ndarray] = []
        size = self._config.batch_size
        with self._torch.no_grad():
            for start in range(0, len(texts), size):
                encoded = self._tokenizer(
                    list(texts[start:start + size]), padding=True, truncation=True,
                    max_length=self._config.max_length, return_tensors="pt",
                )
                encoded = {k: v.to(self._device) for k, v in encoded.items()}
                logits = self._model(**encoded).logits
                chunks.append(logits.detach().cpu().numpy().astype(np.float64))
        return PredictionBatch.from_logits(np.concatenate(chunks, axis=0), self._label_names)


# --------------------------------------------------------------------------- #
# Mock model
# --------------------------------------------------------------------------- #

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z']*")
_NEGATORS = frozenset({"not", "never", "no", "cannot"})
_INTENSIFIERS: dict[str, float] = {
    "very": 1.4, "really": 1.4, "extremely": 1.7, "incredibly": 1.7, "exceptionally": 1.7,
    "truly": 1.4, "genuinely": 1.3, "quite": 1.2, "rather": 1.15, "fairly": 1.05,
}
_CORE_LEXICON: dict[str, float] = {
    "good": 1.2, "great": 1.8, "wonderful": 2.0, "fantastic": 2.0, "amazing": 2.0, "beautiful": 1.6,
    "enjoyable": 1.4, "impressive": 1.5, "nice": 1.2, "happy": 1.4, "helpful": 1.3, "friendly": 1.3,
    "clean": 0.8, "pleasant": 1.3, "acceptable": 0.7, "adequate": 0.6, "polite": 1.0, "courteous": 1.0,
    "agreeable": 1.0, "lively": 1.0, "quick": 0.9, "fast": 0.9, "attractive": 1.2, "exciting": 1.4,
    "interesting": 1.2, "satisfying": 1.3, "loved": 1.8, "love": 1.8, "praised": 1.1,
    "bad": -1.4, "terrible": -2.0, "awful": -2.0, "horrible": -2.0, "boring": -1.5,
    "disappointing": -1.6, "poor": -1.3, "ugly": -1.5, "slow": -1.0, "dull": -1.3, "annoying": -1.4,
    "rude": -1.5, "dirty": -1.2, "unpleasant": -1.4, "unhelpful": -1.3, "unfriendly": -1.3,
    "unhappy": -1.4, "unenjoyable": -1.4, "unimpressive": -1.4, "unremarkable": -1.0,
    "unattractive": -1.2, "uninteresting": -1.2, "disagreeable": -1.2, "complained": -1.0, "criticized": -1.1, "mediocre": -0.9,
}
_EXTENDED_LEXICON: dict[str, float] = {
    "fine": 0.9, "decent": 0.8, "excellent": 2.0, "superb": 2.0, "marvelous": 2.0, "delightful": 1.8,
    "terrific": 2.0, "fabulous": 2.0, "astonishing": 1.8, "remarkable": 1.6, "gorgeous": 1.8,
    "stunning": 1.8, "pleasurable": 1.4, "entertaining": 1.4, "striking": 1.3, "admirable": 1.4,
    "useful": 1.2, "supportive": 1.3, "amiable": 1.2, "welcoming": 1.3, "spotless": 1.2, "tidy": 0.9,
    "lovely": 1.6, "glad": 1.2, "cheerful": 1.3, "charming": 1.5, "thrilling": 1.7, "gripping": 1.5,
    "engaging": 1.3, "fascinating": 1.5, "gratifying": 1.3, "rewarding": 1.3, "civil": 0.8,
    "lousy": -1.6, "dreadful": -2.0, "atrocious": -2.0, "tedious": -1.5, "underwhelming": -1.3,
    "unsatisfying": -1.4, "unsightly": -1.4, "hideous": -2.0, "sluggish": -1.1, "plodding": -1.1,
    "lifeless": -1.3, "irritating": -1.4, "vexing": -1.3, "impolite": -1.3, "discourteous": -1.3,
    "inferior": -1.2, "substandard": -1.3, "lackluster": -1.2, "objectionable": -1.3, "filthy": -1.6, "grubby": -1.2,
}
_FUNCTION_WORDS = frozenset({
    "the", "a", "an", "i", "we", "our", "my", "what", "there", "in", "on", "at", "last",
    "according", "it", "this", "that", "and", "but",
})


class MockNLPModel(BaseNLPModel):
    """Deterministic, offline stand-in for a binary sentiment classifier.

    Logits come from a lexicon-based sentiment margin plus profile-controlled,
    hash-derived perturbations, giving *predictable* failure modes:

    * **Lexical brittleness** - extended (synonym) words are known only with probability
      ``lexicon_coverage``; unknown words contribute nothing.
    * **Negation blindness / weakness** - negators are ignored (``negation_aware=False``)
      or only partially flip polarity (``negation_strength < 1``).
    * **Entity bias** - capitalised mid-sentence tokens (names, places) shift the logits by
      a stable pseudo-random amount.
    * **Format sensitivity** - any change to the raw string (whitespace, punctuation)
      shifts the margin slightly.

    All randomness is derived from SHA-256 hashes, so results are identical across runs,
    processes and platforms.
    """

    _LABELS: tuple[str, ...] = ("NEGATIVE", "POSITIVE")

    def __init__(self, name: str, profile: MockProfile | str = "balanced", seed: int = 1234) -> None:
        """
        Args:
            name: Model name used in reports.
            profile: A :class:`MockProfile` or a key of ``MOCK_PROFILES``.
            seed: Salt for all hash-derived quantities.
        """
        self._name = name
        self._profile = MOCK_PROFILES[profile] if isinstance(profile, str) else profile
        self._seed = seed
        extended = {w: s for w, s in _EXTENDED_LEXICON.items()
                    if self._unit01(f"lex:{w}") < self._profile.lexicon_coverage}
        self._lexicon: dict[str, float] = {**_CORE_LEXICON, **extended}

    @classmethod
    def from_config(cls, config: ModelConfig, seed: int = 1234) -> "MockNLPModel":
        """Build a mock model from a :class:`ModelConfig`."""
        return cls(config.name, config.mock_profile, seed)

    @property
    def name(self) -> str:
        return self._name

    @property
    def backend(self) -> str:
        return "mock"

    @property
    def label_names(self) -> tuple[str, ...]:
        return self._LABELS

    @property
    def profile(self) -> MockProfile:
        """The behavioural profile in use."""
        return self._profile

    # -- deterministic pseudo-randomness ------------------------------------ #
    def _unit01(self, key: str) -> float:
        digest = hashlib.sha256(f"{self._seed}|{key}".encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") / 2.0 ** 64

    def _unit(self, key: str) -> float:
        """Deterministic value in [-1, 1)."""
        return 2.0 * self._unit01(key) - 1.0

    # -- scoring ------------------------------------------------------------ #
    @staticmethod
    def _is_negated(tokens: list[str], index: int) -> bool:
        for tok in tokens[max(0, index - 3):index]:
            if tok in _NEGATORS or tok.endswith("n't"):
                return True
        return False

    def _sentiment_score(self, text: str) -> float:
        tokens = [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]
        score = 0.0
        for i, tok in enumerate(tokens):
            weight = self._lexicon.get(tok)
            if weight is None:
                continue
            if i > 0 and tokens[i - 1] in _INTENSIFIERS:
                weight *= _INTENSIFIERS[tokens[i - 1]]
            if self._profile.negation_aware and self._is_negated(tokens, i):
                weight *= -self._profile.negation_strength
            score += weight
        return score

    def _entity_bias(self, text: str) -> float:
        bias = 0.0
        for match in _TOKEN_RE.finditer(text):
            token = match.group(0)
            if not token[0].isupper():
                continue
            preceding = text[:match.start()].rstrip()
            if not preceding or preceding[-1] in ".!?":
                continue  # sentence-initial capitalisation is not an entity cue
            entity = token.split("'")[0]
            if entity.lower() in _FUNCTION_WORDS or entity.lower() in self._lexicon:
                continue
            bias += self._profile.entity_bias_strength * self._unit(f"ent:{entity}")
        return bias

    def _margin(self, text: str) -> float:
        p = self._profile
        raw = (self._sentiment_score(text) + p.base_bias + self._entity_bias(text)
               + p.format_jitter * self._unit(f"fmt:{text}"))
        return p.sharpness * raw

    def predict(self, texts: Sequence[str]) -> PredictionBatch:
        logits = np.zeros((len(texts), 2), dtype=np.float64)
        for i, text in enumerate(texts):
            margin = self._margin(text)
            offset = 2.0 * self._unit(f"off:{text}")  # unnormalised logit offset (softmax-invariant)
            logits[i] = (offset - margin / 2.0, offset + margin / 2.0)
        return PredictionBatch.from_logits(logits, self._LABELS)


def build_model(config: ModelConfig, seed: int = 1234) -> BaseNLPModel:
    """Instantiate the backend named in ``config``.

    Args:
        config: Model configuration (already offline-resolved if desired).
        seed: Seed for the mock backend.
    """
    if config.backend == "mock":
        return MockNLPModel.from_config(config, seed)
    if config.backend == "huggingface":
        return HuggingFaceNLPModel(config)
    raise ValueError(f"Unsupported backend {config.backend!r}.")


__all__: list[Any] = [
    "PredictionBatch", "BaseNLPModel", "HuggingFaceNLPModel", "MockNLPModel", "build_model",
]
