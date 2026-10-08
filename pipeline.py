"""End-to-end orchestration: audit -> figures -> sensitivity sweeps -> paper artefacts."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from typing import Any, Sequence

from metamorphic_nlp_auditor.analysis import plotting
from metamorphic_nlp_auditor.analysis.report_generator import PaperReportGenerator
from metamorphic_nlp_auditor.analysis.sensitivity import (
    DEFAULT_DOSE_VALUES,
    DOSE_RESPONSE_TARGETS,
    SensitivityAnalyzer,
    SweepResult,
)
from metamorphic_nlp_auditor.config import ExperimentConfig, ModelConfig, to_jsonable
from metamorphic_nlp_auditor.runner.audit_engine import AuditEngine


def run_full_pipeline(config: ExperimentConfig, output_dir: str, run_sweeps: bool = True,
                      quick: bool = False) -> dict[str, Any]:
    """Run the complete study and write every artefact under ``output_dir``.

    Args:
        config: Experiment configuration (per-pair records are forced on).
        output_dir: Destination directory (created if needed).
        run_sweeps: Whether to run the sensitivity analyses.
        quick: Use coarser sweep grids (faster, for smoke tests).

    Returns:
        Manifest mapping artefact names to file paths (also written to ``manifest.json``).
    """
    config = dataclasses.replace(config, audit=dataclasses.replace(config.audit, include_pair_records=True))
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    manifest: dict[str, Any] = {}

    report = AuditEngine(config).run()
    manifest["audit_report_json"] = report.save(os.path.join(output_dir, "audit_report.json"))

    figures: dict[str, str] = {}

    def add_figure(key: str, caption: str, paths: list[str]) -> None:
        manifest[key] = paths
        figures[caption] = os.path.relpath(paths[0], output_dir).replace(os.sep, "/")

    add_figure("fig_mvr_heatmap", "MVR per model and metamorphic relation",
               plotting.plot_mvr_heatmap(report, os.path.join(fig_dir, "fig_mvr_heatmap")))
    add_figure("fig_mvr_bars", "MVR per relation with 95% Wilson intervals",
               plotting.plot_mvr_bars(report, os.path.join(fig_dir, "fig_mvr_bars")))
    add_figure("fig_confidence_drop", "Confidence drop on non-flipped pairs",
               plotting.plot_confidence_drop(report, os.path.join(fig_dir, "fig_confidence_drop")))
    add_figure("fig_mpd_vs_mvr", "Prediction divergence vs. violation rate",
               plotting.plot_mpd_vs_mvr(report, os.path.join(fig_dir, "fig_mpd_vs_mvr")))

    sweeps: list[SweepResult] = []
    if run_sweeps:
        analyzer = SensitivityAnalyzer(config)
        counts = (100, 300, 600) if quick else (100, 200, 400, 600, 800)
        rseeds = tuple(range(3 if quick else 10))
        sweeps.append(analyzer.sweep_seed_count(counts))
        sweeps.append(analyzer.sweep_random_seeds(rseeds))
        sweeps.append(analyzer.sweep_confidence_threshold(report))
        for param in DOSE_RESPONSE_TARGETS:
            values = DEFAULT_DOSE_VALUES[param][::2] if quick else DEFAULT_DOSE_VALUES[param]
            sweeps.append(analyzer.sweep_dose_response(param, values))

        for sweep in sweeps:
            notes = sweep.notes
            if sweep.name == "random_seed":
                continue
            key = f"fig_sweep_{sweep.name}"
            paths = plotting.plot_sweep(
                sweep.rows, os.path.join(fig_dir, key), sweep.x_label, metric=notes.get("metric", "mvr"),
                group_by=notes.get("group_by", "model"), relation=notes.get("display_relation", "ALL"),
                title=f"Target relation: {notes['target_relation']}" if "target_relation" in notes else "")
            add_figure(key, f"Sensitivity sweep: {sweep.name}", paths)
        sweeps_path = os.path.join(output_dir, "sensitivity_sweeps.json")
        with open(sweeps_path, "w", encoding="utf-8") as handle:
            json.dump([to_jsonable(s.to_dict()) for s in sweeps], handle, indent=2)
        manifest["sensitivity_json"] = sweeps_path

    manifest.update(PaperReportGenerator(report, sweeps, figures).write_all(output_dir))
    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """CLI for the Phase 2 pipeline."""
    parser = argparse.ArgumentParser(description="Full metamorphic audit study with figures, sweeps and report.")
    parser.add_argument("--backend", choices=("mock", "huggingface"), default="mock")
    parser.add_argument("--model-name", action="append", default=None, metavar="HF_MODEL_ID",
                        help="Hugging Face model id to audit (repeatable). Implies --backend huggingface.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or mps (Hugging Face only).")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-seeds", type=int, default=600)
    parser.add_argument("--output-dir", default="audit_outputs")
    parser.add_argument("--no-sweeps", action="store_true", help="Skip sensitivity analyses.")
    parser.add_argument("--quick", action="store_true", help="Coarser sweep grids for a fast smoke test.")
    args = parser.parse_args(argv)

    if args.model_name:
        models = tuple(ModelConfig(name=mid.split("/")[-1], backend="huggingface", hf_model_id=mid,
                                   device=args.device) for mid in args.model_name)
        config = ExperimentConfig(random_seed=args.seed, offline_mode=False, models=models)
    elif args.backend == "mock":
        config = ExperimentConfig.offline_default(args.seed)
    else:
        config = ExperimentConfig.huggingface_default(args.seed)
    config.perturbation.num_seeds = args.num_seeds
    manifest = run_full_pipeline(config, args.output_dir, run_sweeps=not args.no_sweeps, quick=args.quick)
    print(f"Artefacts written to {args.output_dir}:")
    for key, value in manifest.items():
        print(f"  {key}: {value if isinstance(value, str) else value[0]}")
    return manifest


if __name__ == "__main__":  # pragma: no cover
    main()
