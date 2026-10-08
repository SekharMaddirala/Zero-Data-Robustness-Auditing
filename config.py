"""Typed configuration objects for the Metamorphic Robustness Auditor (Phase 1).

All configuration is expressed as plain dataclasses with eager validation so that
mis-specified experiments fail fast, before any model is loaded.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

SUPPORTED_BACKENDS: tuple[str, ...] = ("mock", "huggingface")
SUPPORTED_DIVERGENCES: tuple[str, ...] = ("jsd", "l1")
DEFAULT_HF_MODEL_ID: str = "distilbert-base-uncased-finetuned-sst-2-english"


class RelationType(str, Enum):
    """The four metamorphic relations audited by the framework."""

    LEXICAL_EQUIVALENCE = "MR-1"
    SYNTACTIC_INVARIANCE = "MR-2"
    ENTITY_SWAP = "MR-3"
    NEGATION_INVERSION = "MR-4"

    @property
    def short_id(self) -> str:
        """Identifier without the hyphen (e.g. ``MR1``), used in test ids."""
        return self.value.replace("-", "")


@dataclass(frozen=True)
class MockProfile:
    """Behavioural profile of the offline :class:`MockNLPModel`.

    Each field switches on a *predictable failure mode* so that the auditing
    pipeline can be validated end-to-end without any network or GPU access.

    Attributes:
        name: Profile identifier.
        lexicon_coverage: Fraction (0-1) of the *extended* (synonym) lexicon the
            mock "knows". Unknown words contribute nothing -> lexical brittleness.
        negation_aware: If False the model ignores negators (shallow heuristic),
            so "not bad" is scored like "bad".
        negation_strength: Fraction of the polarity that is flipped by a negator
            when ``negation_aware`` is True (<1 yields weaker confidence on litotes).
        entity_bias_strength: Magnitude of the deterministic per-entity logit bias.
        format_jitter: Magnitude of deterministic surface-form noise (whitespace,
            punctuation, any string change).
        sharpness: Scale from the internal sentiment margin to logit margin.
        base_bias: Constant bias towards the positive (+) or negative (-) class.
    """

    name: str
    lexicon_coverage: float = 1.0
    negation_aware: bool = True
    negation_strength: float = 0.9
    entity_bias_strength: float = 0.1
    format_jitter: float = 0.1
    sharpness: float = 1.6
    base_bias: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.lexicon_coverage <= 1.0:
            raise ValueError("lexicon_coverage must lie in [0, 1].")
        if not 0.0 <= self.negation_strength <= 1.5:
            raise ValueError("negation_strength must lie in [0, 1.5].")
        if self.entity_bias_strength < 0 or self.format_jitter < 0:
            raise ValueError("entity_bias_strength and format_jitter must be >= 0.")
        if self.sharpness <= 0:
            raise ValueError("sharpness must be positive.")


MOCK_PROFILES: dict[str, MockProfile] = {
    "robust": MockProfile(
        name="robust", lexicon_coverage=1.0, negation_aware=True, negation_strength=0.95,
        entity_bias_strength=0.05, format_jitter=0.05, sharpness=1.8, base_bias=0.05,
    ),
    "balanced": MockProfile(
        name="balanced", lexicon_coverage=0.8, negation_aware=True, negation_strength=0.7,
        entity_bias_strength=0.30, format_jitter=0.12, sharpness=1.6, base_bias=0.10,
    ),
    "brittle": MockProfile(
        name="brittle", lexicon_coverage=0.5, negation_aware=True, negation_strength=0.5,
        entity_bias_strength=0.60, format_jitter=0.25, sharpness=1.5, base_bias=-0.10,
    ),
    "negation_blind": MockProfile(
        name="negation_blind", lexicon_coverage=0.9, negation_aware=False, negation_strength=0.0,
        entity_bias_strength=0.10, format_jitter=0.08, sharpness=1.8, base_bias=0.0,
    ),
}


@dataclass
class ModelConfig:
    """Configuration of a single classifier under audit.

    Attributes:
        name: Unique display name used as the key in reports.
        backend: ``"huggingface"`` or ``"mock"``.
        hf_model_id: Hugging Face hub id / local path (required for ``huggingface``).
        mock_profile: Key into :data:`MOCK_PROFILES` (used by the mock backend, and
            by the offline substitute when ``ExperimentConfig.offline_mode`` is True).
        device: ``"auto"``, ``"cpu"``, ``"cuda"`` or ``"mps"``.
        batch_size: Inference batch size.
        max_length: Maximum token length (truncation).
        local_files_only: Forbid network access when loading from the HF cache.
    """

    name: str
    backend: str = "mock"
    hf_model_id: str | None = None
    mock_profile: str = "balanced"
    device: str = "auto"
    batch_size: int = 32
    max_length: int = 128
    local_files_only: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ModelConfig.name must be non-empty.")
        if self.backend not in SUPPORTED_BACKENDS:
            raise ValueError(f"backend must be one of {SUPPORTED_BACKENDS}, got {self.backend!r}.")
        if self.backend == "huggingface" and not self.hf_model_id:
            raise ValueError("hf_model_id is required for the 'huggingface' backend.")
        if self.mock_profile not in MOCK_PROFILES:
            raise ValueError(f"Unknown mock_profile {self.mock_profile!r}; options: {sorted(MOCK_PROFILES)}.")
        if self.batch_size < 1 or self.max_length < 1:
            raise ValueError("batch_size and max_length must be >= 1.")


@dataclass
class PerturbationConfig:
    """Parameters controlling seed generation and metamorphic variant creation.

    Attributes:
        num_seeds: Number of synthetic seed sentences (default 600).
        enabled_relations: Which MRs to instantiate.
        lexical_variants: Synonym-substitution variants per seed (MR-1).
        include_voice_conversion: Add active<->passive conversions to MR-2.
        include_structural: Add adverbial repositioning and complementizer insertion to MR-2.
        padding_variants: Padding variants for each padded seed (surface-form control inside MR-2).
        padding_fraction: Fraction of seeds that receive padding (keeps trivial edits a minority).
        entity_variants: Entity-swap variants per seed (MR-3).
        negation_variants: Double-negation variants per seed (MR-4).
    """

    num_seeds: int = 600
    enabled_relations: tuple[RelationType, ...] = tuple(RelationType)
    lexical_variants: int = 2
    include_voice_conversion: bool = True
    include_structural: bool = True
    padding_variants: int = 1
    padding_fraction: float = 0.15
    entity_variants: int = 2
    negation_variants: int = 2

    def __post_init__(self) -> None:
        self.enabled_relations = tuple(RelationType(r) for r in self.enabled_relations)
        if self.num_seeds < 4:
            raise ValueError("num_seeds must be >= 4 (one per linguistic category).")
        if not self.enabled_relations:
            raise ValueError("At least one metamorphic relation must be enabled.")
        if not 0.0 <= self.padding_fraction <= 1.0:
            raise ValueError("padding_fraction must lie in [0, 1].")
        for attr in ("lexical_variants", "padding_variants", "entity_variants", "negation_variants"):
            if getattr(self, attr) < 0:
                raise ValueError(f"{attr} must be >= 0.")


@dataclass
class AuditConfig:
    """Parameters of the audit/metric computation.

    Attributes:
        divergence_metric: Primary MPD metric, ``"jsd"`` (base-2 Jensen-Shannon) or ``"l1"``.
        confidence_drop_threshold: Minimum confidence drop (on non-flipped pairs) counted
            as a "degradation event".
        max_violation_examples: Worst violating pairs retained per relation in the report.
        include_pair_records: Whether to embed per-pair records in the report.
        wilson_z: z-score for the Wilson confidence interval on MVR (1.96 -> 95 %).
    """

    divergence_metric: str = "jsd"
    confidence_drop_threshold: float = 0.05
    max_violation_examples: int = 5
    include_pair_records: bool = True
    wilson_z: float = 1.96

    def __post_init__(self) -> None:
        if self.divergence_metric not in SUPPORTED_DIVERGENCES:
            raise ValueError(f"divergence_metric must be one of {SUPPORTED_DIVERGENCES}.")
        if not 0.0 <= self.confidence_drop_threshold <= 1.0:
            raise ValueError("confidence_drop_threshold must lie in [0, 1].")
        if self.max_violation_examples < 0:
            raise ValueError("max_violation_examples must be >= 0.")
        if self.wilson_z <= 0:
            raise ValueError("wilson_z must be positive.")


def _default_mock_models() -> tuple[ModelConfig, ...]:
    return (
        ModelConfig(name="mock-robust", backend="mock", mock_profile="robust"),
        ModelConfig(name="mock-balanced", backend="mock", mock_profile="balanced"),
        ModelConfig(name="mock-brittle", backend="mock", mock_profile="brittle"),
        ModelConfig(name="mock-negation-blind", backend="mock", mock_profile="negation_blind"),
    )


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration.

    Attributes:
        random_seed: Master seed (seed generation + mock-model determinism).
        offline_mode: If True, every ``huggingface`` model is transparently replaced by
            a :class:`MockNLPModel` so nothing is downloaded and no GPU is required.
        models: Models under audit.
        perturbation: Seed/variant generation parameters.
        audit: Metric parameters.
        output_dir: Default directory for JSON reports.
    """

    random_seed: int = 1234
    offline_mode: bool = True
    models: tuple[ModelConfig, ...] = field(default_factory=_default_mock_models)
    perturbation: PerturbationConfig = field(default_factory=PerturbationConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    output_dir: str = "audit_outputs"

    def __post_init__(self) -> None:
        self.models = tuple(self.models)
        if not self.models:
            raise ValueError("At least one model must be configured.")
        names = [m.name for m in self.models]
        if len(set(names)) != len(names):
            raise ValueError(f"Model names must be unique, got {names}.")

    def effective_model_configs(self) -> tuple[ModelConfig, ...]:
        """Return model configs after applying the offline-mode substitution."""
        resolved: list[ModelConfig] = []
        for cfg in self.models:
            if self.offline_mode and cfg.backend == "huggingface":
                resolved.append(dataclasses.replace(cfg, backend="mock"))
            else:
                resolved.append(cfg)
        return tuple(resolved)

    @classmethod
    def offline_default(cls, random_seed: int = 1234) -> "ExperimentConfig":
        """Fully offline experiment over four mock models with distinct failure modes."""
        return cls(random_seed=random_seed, offline_mode=True)

    @classmethod
    def huggingface_default(cls, random_seed: int = 1234) -> "ExperimentConfig":
        """Experiment over real pre-trained Hugging Face classifiers (needs torch + transformers)."""
        models = (
            ModelConfig(name="distilbert-sst2", backend="huggingface", hf_model_id=DEFAULT_HF_MODEL_ID,
                        mock_profile="balanced"),
            ModelConfig(name="bert-sst2", backend="huggingface",
                        hf_model_id="textattack/bert-base-uncased-SST-2", mock_profile="brittle"),
            ModelConfig(name="roberta-twitter-sentiment", backend="huggingface",
                        hf_model_id="cardiffnlp/twitter-roberta-base-sentiment-latest", mock_profile="brittle"),
        )
        return cls(random_seed=random_seed, offline_mode=False, models=models)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable snapshot of the configuration."""
        return to_jsonable(self)


def to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses, enums, tuples and numpy values to JSON-safe types."""
    if isinstance(obj, Enum):
        return obj.value
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {(k.value if isinstance(k, Enum) else str(k)): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "tolist") and callable(obj.tolist):
        return obj.tolist()
    return obj
