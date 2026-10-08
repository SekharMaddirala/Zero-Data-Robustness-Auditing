"""Tests for config, generator, models and the audit engine."""

import numpy as np
import pytest

from metamorphic_nlp_auditor.config import ExperimentConfig, ModelConfig, PerturbationConfig, RelationType
from metamorphic_nlp_auditor.core.perturbation_generator import PerturbationGenerator
from metamorphic_nlp_auditor.models.model_wrapper import MockNLPModel
from metamorphic_nlp_auditor.runner.audit_engine import (
    AuditEngine,
    jensen_shannon_divergence,
    l1_distance,
    wilson_interval,
)


def test_config_validation():
    with pytest.raises(ValueError):
        ModelConfig(name="x", backend="huggingface")  # missing hf_model_id
    with pytest.raises(ValueError):
        PerturbationConfig(num_seeds=2)
    with pytest.raises(ValueError):
        ExperimentConfig(models=(ModelConfig(name="a"), ModelConfig(name="a")))


def test_offline_mode_substitutes_mock():
    cfg = ExperimentConfig(models=(ModelConfig(name="m", backend="huggingface", hf_model_id="x/y"),))
    assert cfg.effective_model_configs()[0].backend == "mock"


def test_generator_builds_100_plus_unique_seeds_and_all_relations():
    gen = PerturbationGenerator(PerturbationConfig(), seed=1)
    seeds = gen.generate_seeds()
    cases = gen.generate_test_cases(seeds)
    assert len(seeds) >= 100 and len({s.text for s in seeds}) == len(seeds)
    assert {c.relation_type for c in cases} == set(RelationType)
    assert all(c.expected_invariance and c.seed_text != c.transformed_text for c in cases)
    assert len({c.test_id for c in cases}) == len(cases)


def test_generator_is_deterministic():
    a = PerturbationGenerator(PerturbationConfig(), seed=5).generate_test_cases()
    b = PerturbationGenerator(PerturbationConfig(), seed=5).generate_test_cases()
    assert [c.as_tuple() for c in a] == [c.as_tuple() for c in b]


def test_mock_model_is_deterministic_and_valid():
    m1, m2 = MockNLPModel("a", "balanced", 7), MockNLPModel("a", "balanced", 7)
    texts = ["The film was very good.", "The film was very bad.", "The table has three cups."]
    b1, b2 = m1.predict(texts), m2.predict(texts)
    assert np.array_equal(b1.logits, b2.logits)
    assert np.allclose(b1.probabilities.sum(axis=1), 1.0)
    assert b1.labels[0] == 1 and b1.labels[1] == 0


def test_negation_blind_mock_fails_mr4_but_robust_does_not():
    cfg = ExperimentConfig()
    cases = [c for c in PerturbationGenerator(cfg.perturbation, 1).generate_test_cases()
             if c.relation_type == RelationType.NEGATION_INVERSION]
    robust = AuditEngine(cfg, [MockNLPModel("r", "robust", 1)]).run(cases).models["r"].overall.mvr
    blind = AuditEngine(cfg, [MockNLPModel("b", "negation_blind", 1)]).run(cases).models["b"].overall.mvr
    assert blind > 0.5 > robust


def test_metric_primitives():
    p, q = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    assert jensen_shannon_divergence(p, p) == pytest.approx(0.0)
    assert jensen_shannon_divergence(p, q) == pytest.approx(1.0)
    assert l1_distance(p, q) == pytest.approx(2.0)
    low, high = wilson_interval(5, 10)
    assert 0.0 < low < 0.5 < high < 1.0
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_full_offline_run_report_is_json_serialisable():
    report = AuditEngine(ExperimentConfig()).run()
    assert set(report.models) == {m.name for m in ExperimentConfig().models}
    assert len(report.to_json()) > 1000
    for audit in report.models.values():
        assert 0.0 <= audit.overall.mvr <= 1.0


def test_default_scale_and_structural_majority_in_mr2():
    gen = PerturbationGenerator(PerturbationConfig(), seed=1234)
    seeds = gen.generate_seeds()
    cases = gen.generate_test_cases(seeds)
    mr2 = [c for c in cases if c.relation_type == RelationType.SYNTACTIC_INVARIANCE]
    padding = [c for c in mr2 if c.operator == "punctuation_padding"]
    assert len(seeds) == 600 and len(cases) >= 2000
    assert len(padding) < 0.5 * len(mr2)
    assert sum(c.relation_type == RelationType.NEGATION_INVERSION for c in cases) >= 400


def test_human_validation_roundtrip(tmp_path):
    import csv
    from metamorphic_nlp_auditor.analysis.human_validation import (
        agreement_report, cohen_kappa, export_annotation_sheets, fleiss_kappa)
    assert cohen_kappa(list("yyyn"), list("yyyn")) == pytest.approx(1.0)
    assert cohen_kappa(list("yyny"), list("nnyn")) < 0
    assert fleiss_kappa([["yes", "yes"], ["no", "no"], ["yes", "no"]]) == pytest.approx(0.3333333, rel=1e-4)
    cases = PerturbationGenerator(PerturbationConfig(num_seeds=200), 1).generate_test_cases()
    sheet, key = export_annotation_sheets(cases, 40, str(tmp_path), seed=1)
    rows = list(csv.DictReader(open(sheet, encoding="utf-8-sig")))
    assert len(rows) == 40
    for name, flip in (("a.csv", 0), ("b.csv", 5)):
        with open(tmp_path / name, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader()
            for i, r in enumerate(rows):
                r = dict(r); r["same_polarity"] = "no" if i < flip else "yes"; r["both_natural"] = "yes"; w.writerow(r)
    report = agreement_report([str(tmp_path / "a.csv"), str(tmp_path / "b.csv")], key)
    assert report["n_pairs"] == 40 and report["percent_agreement"] == pytest.approx(35 / 40)
    assert 0.0 < report["majority_yes_rate"]["ALL"]["rate"] <= 1.0
