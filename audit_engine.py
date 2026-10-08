"""Audit engine: executes metamorphic pairs and computes MVR, MPD and confidence degradation."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Sequence

import numpy as np

from metamorphic_nlp_auditor.config import ExperimentConfig, RelationType, to_jsonable
from metamorphic_nlp_auditor.core.perturbation_generator import (
    MetamorphicTestCase,
    PerturbationGenerator,
)
from metamorphic_nlp_auditor.models.model_wrapper import BaseNLPModel, build_model

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Metric primitives
# --------------------------------------------------------------------------- #

def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Base-2 Jensen-Shannon divergence between two distributions (range [0, 1])."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > _EPS
        return float(np.sum(a[mask] * np.log2(a[mask] / np.maximum(b[mask], _EPS))))

    return max(0.0, 0.5 * _kl(p, m) + 0.5 * _kl(q, m))


def l1_distance(p: np.ndarray, q: np.ndarray) -> float:
    """L1 distance between two distributions (range [0, 2])."""
    return float(np.abs(np.asarray(p, dtype=np.float64) - np.asarray(q, dtype=np.float64)).sum())


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion; ``(0, 0)`` when ``n == 0``."""
    if n == 0:
        return 0.0, 0.0
    phat = successes / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PairResult:
    """Outcome of one metamorphic pair for one model."""

    test_id: str
    seed_id: str
    relation: str
    operator: str
    category: str
    seed_text: str
    transformed_text: str
    seed_label: int
    transformed_label: int
    seed_confidence: float
    transformed_confidence: float
    violated: bool
    jsd: float
    l1: float

    @property
    def confidence_drop(self) -> float:
        """``conf(x) - conf(x')``; positive means confidence was lost."""
        return self.seed_confidence - self.transformed_confidence


@dataclass(frozen=True)
class MetricSummary:
    """Aggregated metrics over a set of pairs.

    Attributes:
        n_pairs: Number of pairs.
        n_violations: Pairs with ``f(x) != f(x')``.
        mvr: Metamorphic Violation Rate.
        mvr_ci_low / mvr_ci_high: Wilson interval bounds on MVR.
        mpd: Mean Prediction Divergence under the configured primary metric.
        mpd_jsd / mpd_l1: MPD under Jensen-Shannon (base 2) and L1 respectively.
        n_non_flipped: Pairs whose label did not change.
        mean_confidence_drop: Signed mean of ``conf(x) - conf(x')`` over non-flipped pairs.
        mean_confidence_degradation: Mean of ``max(0, drop)`` over non-flipped pairs.
        degradation_rate: Fraction of non-flipped pairs whose drop exceeds the threshold.
    """

    n_pairs: int
    n_violations: int
    mvr: float
    mvr_ci_low: float
    mvr_ci_high: float
    mpd: float
    mpd_jsd: float
    mpd_l1: float
    n_non_flipped: int
    mean_confidence_drop: float
    mean_confidence_degradation: float
    degradation_rate: float


@dataclass
class ModelAudit:
    """All audit results for a single model."""

    model_name: str
    backend: str
    label_names: tuple[str, ...]
    overall: MetricSummary
    by_relation: dict[str, MetricSummary]
    by_operator: dict[str, MetricSummary]
    by_category: dict[str, MetricSummary]
    violation_examples: dict[str, list[dict[str, Any]]]
    pair_records: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AuditReport:
    """Structured report across all audited models."""

    metadata: dict[str, Any]
    models: dict[str, ModelAudit]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dictionary."""
        return {"metadata": to_jsonable(self.metadata),
                "models": {name: to_jsonable(audit) for name, audit in self.models.items()}}

    def to_json(self, indent: int = 2) -> str:
        """Serialise the report to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save(self, path: str) -> str:
        """Write the JSON report to ``path`` (parent directories are created); returns ``path``."""
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self.to_json())
        return path

    def to_rows(self) -> list[dict[str, Any]]:
        """Flat ``(model, relation)`` rows (plus an ``ALL`` row per model) for tables/plots."""
        rows: list[dict[str, Any]] = []
        for name, audit in self.models.items():
            for relation, summary in [("ALL", audit.overall), *audit.by_relation.items()]:
                rows.append({"model": name, "relation": relation, **to_jsonable(summary)})
        return rows


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #

class AuditEngine:
    """Runs paired metamorphic tests ``(x, x')`` against one or more models."""

    def __init__(self, config: ExperimentConfig, models: Sequence[BaseNLPModel] | None = None) -> None:
        """
        Args:
            config: Experiment configuration.
            models: Pre-built models; if omitted they are built from
                ``config.effective_model_configs()`` (honouring offline mode).
        """
        self._config = config
        if models is None:
            models = [build_model(cfg, config.random_seed) for cfg in config.effective_model_configs()]
        self._models = list(models)

    def run(self, test_cases: Sequence[MetamorphicTestCase] | None = None) -> AuditReport:
        """Audit every model and return an :class:`AuditReport`.

        Args:
            test_cases: Optional pre-generated test cases; generated from the config if omitted.
        """
        generator = PerturbationGenerator(self._config.perturbation, self._config.random_seed)
        if test_cases is None:
            seeds = generator.generate_seeds()
            cases = generator.generate_test_cases(seeds)
            generation_summary = generator.summarize(seeds, cases)
        else:
            cases = list(test_cases)
            seeds = []
            generation_summary = {"n_seeds": len({c.seed_id for c in cases}), "n_test_cases": len(cases)}
        if not cases:
            raise ValueError("No test cases to run.")
        audits = {m.name: self.evaluate_model(m, cases) for m in self._models}
        metadata = {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "random_seed": self._config.random_seed,
            "offline_mode": self._config.offline_mode,
            "generation": generation_summary,
            "config": to_jsonable(self._config),
        }
        return AuditReport(metadata=metadata, models=audits)

    def evaluate_model(self, model: BaseNLPModel, cases: Sequence[MetamorphicTestCase]) -> ModelAudit:
        """Evaluate one model on ``cases`` (each unique text is classified exactly once)."""
        cfg = self._config.audit
        unique: dict[str, int] = {}
        for case in cases:
            for text in (case.seed_text, case.transformed_text):
                unique.setdefault(text, len(unique))
        batch = model.predict(list(unique))

        pairs: list[PairResult] = []
        for case in cases:
            i, j = unique[case.seed_text], unique[case.transformed_text]
            p, q = batch.probabilities[i], batch.probabilities[j]
            seed_label, new_label = int(batch.labels[i]), int(batch.labels[j])
            pairs.append(PairResult(
                test_id=case.test_id, seed_id=case.seed_id, relation=case.relation_type.value,
                operator=case.operator, category=case.category, seed_text=case.seed_text,
                transformed_text=case.transformed_text, seed_label=seed_label,
                transformed_label=new_label, seed_confidence=float(batch.confidences[i]),
                transformed_confidence=float(batch.confidences[j]),
                violated=case.expected_invariance and seed_label != new_label,
                jsd=jensen_shannon_divergence(p, q), l1=l1_distance(p, q),
            ))

        def summarise(group: Sequence[PairResult]) -> MetricSummary:
            return self._summarise(group)

        examples: dict[str, list[dict[str, Any]]] = {}
        for relation in RelationType:
            worst = sorted((p for p in pairs if p.relation == relation.value and p.violated),
                           key=lambda p: p.jsd, reverse=True)[:cfg.max_violation_examples]
            if worst:
                examples[relation.value] = [self._record(p) for p in worst]

        return ModelAudit(
            model_name=model.name, backend=model.backend, label_names=model.label_names,
            overall=summarise(pairs),
            by_relation=self._group(pairs, lambda p: p.relation),
            by_operator=self._group(pairs, lambda p: p.operator),
            by_category=self._group(pairs, lambda p: p.category),
            violation_examples=examples,
            pair_records=[self._record(p) for p in pairs] if cfg.include_pair_records else [],
        )

    # -- helpers ------------------------------------------------------------ #
    def _group(self, pairs: Sequence[PairResult], key: Callable[[PairResult], str]) -> dict[str, MetricSummary]:
        buckets: dict[str, list[PairResult]] = defaultdict(list)
        for pair in pairs:
            buckets[key(pair)].append(pair)
        return {k: self._summarise(v) for k, v in sorted(buckets.items())}

    def _summarise(self, group: Sequence[PairResult]) -> MetricSummary:
        cfg = self._config.audit
        n = len(group)
        if n == 0:
            return MetricSummary(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0)
        violations = sum(p.violated for p in group)
        low, high = wilson_interval(violations, n, cfg.wilson_z)
        mpd_jsd = float(np.mean([p.jsd for p in group]))
        mpd_l1 = float(np.mean([p.l1 for p in group]))
        stable = [p for p in group if not p.violated]
        if stable:
            drops = np.array([p.confidence_drop for p in stable], dtype=np.float64)
            mean_drop = float(drops.mean())
            degradation = float(np.maximum(drops, 0.0).mean())
            rate = float((drops > cfg.confidence_drop_threshold).mean())
        else:
            mean_drop = degradation = rate = 0.0
        return MetricSummary(
            n_pairs=n, n_violations=violations, mvr=violations / n, mvr_ci_low=low, mvr_ci_high=high,
            mpd=mpd_jsd if cfg.divergence_metric == "jsd" else mpd_l1, mpd_jsd=mpd_jsd, mpd_l1=mpd_l1,
            n_non_flipped=len(stable), mean_confidence_drop=mean_drop,
            mean_confidence_degradation=degradation, degradation_rate=rate,
        )

    @staticmethod
    def _record(pair: PairResult) -> dict[str, Any]:
        record = to_jsonable(pair)
        record["confidence_drop"] = pair.confidence_drop
        return record


# --------------------------------------------------------------------------- #
# Command line entry point
# --------------------------------------------------------------------------- #

def main(argv: Sequence[str] | None = None) -> AuditReport:
    """CLI: run an audit and write ``<output_dir>/audit_report.json``."""
    parser = argparse.ArgumentParser(description="Zero-data metamorphic robustness audit.")
    parser.add_argument("--backend", choices=("mock", "huggingface"), default="mock",
                        help="'mock' runs fully offline; 'huggingface' downloads/loads real models.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-seeds", type=int, default=600)
    parser.add_argument("--output-dir", default="audit_outputs")
    args = parser.parse_args(argv)

    config = (ExperimentConfig.offline_default(args.seed) if args.backend == "mock"
              else ExperimentConfig.huggingface_default(args.seed))
    config.perturbation.num_seeds = args.num_seeds
    config.output_dir = args.output_dir

    report = AuditEngine(config).run()
    path = report.save(os.path.join(config.output_dir, "audit_report.json"))

    print(f"{'model':<24}{'relation':<9}{'n':>5}{'MVR':>8}{'MPD(JSD)':>10}{'conf.drop':>11}")
    for row in report.to_rows():
        print(f"{row['model']:<24}{row['relation']:<9}{row['n_pairs']:>5}{row['mvr']:>8.3f}"
              f"{row['mpd_jsd']:>10.4f}{row['mean_confidence_degradation']:>11.4f}")
    print(f"\nReport written to {path}")
    return report


if __name__ == "__main__":  # pragma: no cover
    main()
