"""Sensitivity sweeps and validity checks for the metamorphic auditor.

Four analyses are provided:

* **Seed-count sweep** - does MVR converge as the number of seed sentences grows?
* **Random-seed stability** - variability of MVR across master seeds.
* **Confidence-threshold sweep** - how the "degradation event" rate depends on its threshold.
* **Dose-response (mock) sweep** - the auditor's *construct validity* check: a defect injected
  into the mock model at increasing strength must produce a monotone MVR response on the
  relation designed to expose it (reported via Spearman's rho).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from metamorphic_nlp_auditor.config import MOCK_PROFILES, ExperimentConfig
from metamorphic_nlp_auditor.core.perturbation_generator import PerturbationGenerator
from metamorphic_nlp_auditor.models.model_wrapper import BaseNLPModel, MockNLPModel, build_model
from metamorphic_nlp_auditor.runner.audit_engine import AuditEngine, AuditReport

#: parameter -> (relation expected to expose it, expected direction of MVR as the parameter grows)
DOSE_RESPONSE_TARGETS: dict[str, tuple[str, str]] = {
    "lexicon_coverage": ("MR-1", "decreasing"),
    "entity_bias_strength": ("MR-3", "increasing"),
    "negation_strength": ("MR-4", "decreasing"),
    "format_jitter": ("MR-2", "increasing"),
}
DEFAULT_DOSE_VALUES: dict[str, tuple[float, ...]] = {
    "lexicon_coverage": (0.0, 0.25, 0.5, 0.75, 1.0),
    "entity_bias_strength": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    "negation_strength": (0.0, 0.25, 0.5, 0.75, 1.0),
    "format_jitter": (0.0, 0.1, 0.2, 0.4, 0.6, 0.8),
}


@dataclass
class SweepResult:
    """Rows of one sweep plus descriptive notes.

    Attributes:
        name: Sweep identifier (e.g. ``"seed_count"``).
        x_label: Human-readable name of the swept variable.
        rows: Flat dictionaries, each containing ``x``, ``model``, ``relation`` and metric fields.
        notes: Extra info (``metric``, ``display_relation``, ``group_by``, ``spearman_rho`` ...).
    """

    name: str
    x_label: str
    rows: list[dict[str, Any]]
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation."""
        return {"name": self.name, "x_label": self.x_label, "rows": self.rows, "notes": self.notes}


def _ranks(values: Sequence[float]) -> np.ndarray:
    """Average ranks (ties share the mean rank)."""
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=np.float64)
    i = 0
    while i < len(arr):
        j = i
        while j + 1 < len(arr) and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Spearman rank correlation; ``None`` if undefined (constant input or fewer than 2 points)."""
    if len(x) != len(y) or len(x) < 2:
        return None
    rx, ry = _ranks(x), _ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _flatten(report: AuditReport, x: float, **extra: Any) -> list[dict[str, Any]]:
    return [{**row, "x": x, **extra} for row in report.to_rows()]


class SensitivityAnalyzer:
    """Runs the sweeps against variations of a base :class:`ExperimentConfig`."""

    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config
        self._real_models: list[BaseNLPModel] | None = None

    def _engine(self, cfg: ExperimentConfig) -> AuditEngine:
        """Build an engine; real (Hugging Face) models are loaded once and reused across runs."""
        resolved = cfg.effective_model_configs()
        if self._real_models is None and any(c.backend == "huggingface" for c in resolved):
            self._real_models = [build_model(c, cfg.random_seed) for c in resolved]
        if self._real_models is not None:
            return AuditEngine(cfg, models=self._real_models)
        return AuditEngine(cfg)

    def _lean(self, random_seed: int | None = None, **perturbation: Any) -> ExperimentConfig:
        cfg = self._config
        return dataclasses.replace(
            cfg,
            random_seed=cfg.random_seed if random_seed is None else random_seed,
            perturbation=dataclasses.replace(cfg.perturbation, **perturbation),
            audit=dataclasses.replace(cfg.audit, include_pair_records=False),
        )

    # -- 1. seed-count sweep ------------------------------------------------ #
    def sweep_seed_count(self, counts: Sequence[int] = (40, 80, 120, 200, 300)) -> SweepResult:
        """MVR as a function of the number of seed sentences."""
        rows: list[dict[str, Any]] = []
        for n in counts:
            report = self._engine(self._lean(num_seeds=n)).run()
            rows += _flatten(report, float(n))
        return SweepResult("seed_count", "Number of seed sentences", rows,
                           {"metric": "mvr", "display_relation": "ALL", "group_by": "model"})

    # -- 2. random-seed stability ------------------------------------------- #
    def sweep_random_seeds(self, seeds: Sequence[int] = tuple(range(10))) -> SweepResult:
        """Repeat the audit under different master seeds and summarise the spread of MVR."""
        rows: list[dict[str, Any]] = []
        for s in seeds:
            report = self._engine(self._lean(random_seed=int(s))).run()
            rows += _flatten(report, float(s))
        buckets: dict[tuple[str, str], list[float]] = {}
        for r in rows:
            buckets.setdefault((r["model"], r["relation"]), []).append(r["mvr"])
        stability = [
            {"model": m, "relation": rel, "n_runs": len(v), "mean_mvr": float(np.mean(v)),
             "std_mvr": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
             "min_mvr": float(np.min(v)), "max_mvr": float(np.max(v))}
            for (m, rel), v in sorted(buckets.items())
        ]
        return SweepResult("random_seed", "Master random seed", rows,
                           {"metric": "mvr", "display_relation": "ALL", "group_by": "model", "stability": stability})

    # -- 3. confidence-threshold sweep -------------------------------------- #
    @staticmethod
    def sweep_confidence_threshold(report: AuditReport,
                                   thresholds: Sequence[float] = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20)) -> SweepResult:
        """Share of non-flipped pairs whose confidence drops by more than each threshold.

        Raises:
            ValueError: If the report has no per-pair records.
        """
        rows: list[dict[str, Any]] = []
        for name, audit in report.models.items():
            if not audit.pair_records:
                raise ValueError("Per-pair records are required (AuditConfig.include_pair_records=True).")
            groups: dict[str, list[float]] = {"ALL": []}
            for rec in audit.pair_records:
                if rec["violated"]:
                    continue
                groups["ALL"].append(rec["confidence_drop"])
                groups.setdefault(rec["relation"], []).append(rec["confidence_drop"])
            for relation, drops in sorted(groups.items()):
                arr = np.asarray(drops, dtype=np.float64)
                for t in thresholds:
                    rows.append({"model": name, "relation": relation, "x": float(t),
                                 "degradation_rate": float((arr > t).mean()) if arr.size else 0.0,
                                 "n_non_flipped": int(arr.size)})
        return SweepResult("confidence_threshold", "Confidence-drop threshold", rows,
                           {"metric": "degradation_rate", "display_relation": "ALL", "group_by": "model"})

    # -- 4. dose-response on the mock model --------------------------------- #
    def sweep_dose_response(self, parameter: str, values: Sequence[float] | None = None,
                            base_profile: str = "balanced") -> SweepResult:
        """Vary one mock-model defect parameter and measure the MVR response.

        Args:
            parameter: A key of :data:`DOSE_RESPONSE_TARGETS`.
            values: Parameter values (defaults to :data:`DEFAULT_DOSE_VALUES`).
            base_profile: Mock profile providing all other parameters.

        Returns:
            ``SweepResult`` whose notes hold the target relation, expected direction, Spearman
            rho of MVR vs. parameter on that relation, and ``supported`` (rho matches the
            expected direction with ``|rho| >= 0.8``).
        """
        if parameter not in DOSE_RESPONSE_TARGETS:
            raise ValueError(f"parameter must be one of {sorted(DOSE_RESPONSE_TARGETS)}.")
        target, direction = DOSE_RESPONSE_TARGETS[parameter]
        grid = tuple(values) if values is not None else DEFAULT_DOSE_VALUES[parameter]
        cfg = self._lean()
        cases = PerturbationGenerator(cfg.perturbation, cfg.random_seed).generate_test_cases()
        rows: list[dict[str, Any]] = []
        for v in grid:
            profile = dataclasses.replace(MOCK_PROFILES[base_profile], name=f"{base_profile}[{parameter}={v}]",
                                          **{parameter: float(v)})
            model = MockNLPModel(f"mock-{base_profile}", profile, cfg.random_seed)
            rows += _flatten(AuditEngine(cfg, models=[model]).run(cases), float(v))
        target_rows = sorted((r for r in rows if r["relation"] == target), key=lambda r: r["x"])
        rho = spearman_rho([r["x"] for r in target_rows], [r["mvr"] for r in target_rows])
        sign = 1.0 if direction == "increasing" else -1.0
        return SweepResult(
            f"dose_{parameter}", parameter, rows,
            {"metric": "mvr", "display_relation": None, "group_by": "relation", "target_relation": target,
             "expected_direction": direction, "spearman_rho": rho,
             "supported": bool(rho is not None and sign * rho >= 0.8), "base_profile": base_profile},
        )

    def run_all_dose_responses(self, base_profile: str = "balanced") -> list[SweepResult]:
        """Dose-response sweep for every parameter in :data:`DOSE_RESPONSE_TARGETS`."""
        return [self.sweep_dose_response(p, None, base_profile) for p in DOSE_RESPONSE_TARGETS]
