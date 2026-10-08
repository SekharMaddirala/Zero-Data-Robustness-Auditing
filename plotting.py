"""Publication-style figures (IEEE column widths, serif fonts, PNG + PDF output)."""

from __future__ import annotations

import os
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from metamorphic_nlp_auditor.runner.audit_engine import AuditReport  # noqa: E402

RELATION_ORDER: tuple[str, ...] = ("MR-1", "MR-2", "MR-3", "MR-4")
_COLORS: tuple[str, ...] = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000")
_MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v", "P")
_RC: dict[str, Any] = {
    "font.family": "serif", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
}
SINGLE_COLUMN_IN = 3.5
DOUBLE_COLUMN_IN = 7.16


def _save(fig: "plt.Figure", base_path: str, formats: Sequence[str]) -> list[str]:
    os.makedirs(os.path.dirname(os.path.abspath(base_path)), exist_ok=True)
    paths = []
    for fmt in formats:
        path = f"{base_path}.{fmt}"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        paths.append(path)
    plt.close(fig)
    return paths


def _relations_present(report: AuditReport) -> list[str]:
    present = {r for audit in report.models.values() for r in audit.by_relation}
    return [r for r in RELATION_ORDER if r in present] + sorted(present - set(RELATION_ORDER))


def plot_mvr_heatmap(report: AuditReport, base_path: str, formats: Sequence[str] = ("png", "pdf")) -> list[str]:
    """Heatmap of MVR (rows: models, columns: relations + overall)."""
    relations = _relations_present(report)
    columns = relations + ["ALL"]
    models = list(report.models)
    matrix = np.array([
        [audit.by_relation[r].mvr if r in audit.by_relation else np.nan for r in relations] + [audit.overall.mvr]
        for audit in report.models.values()
    ])
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_IN + 0.6, 0.45 * len(models) + 1.0), constrained_layout=True)
        vmax = max(float(np.nanmax(matrix)), 1e-9)
        image = ax.imshow(matrix, cmap="YlOrRd", vmin=0.0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(columns)), labels=columns)
        ax.set_yticks(range(len(models)), labels=models)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if not np.isnan(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=7,
                            color="white" if matrix[i, j] > 0.6 * vmax else "black")
        ax.set_title("Metamorphic Violation Rate")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="MVR")
    return _save(fig, base_path, formats)


def plot_mvr_bars(report: AuditReport, base_path: str, formats: Sequence[str] = ("png", "pdf")) -> list[str]:
    """Grouped bars of MVR per relation with Wilson 95 % interval error bars."""
    relations = _relations_present(report)
    columns = relations + ["ALL"]
    models = list(report.models.items())
    width = 0.8 / max(len(models), 1)
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_IN, 2.6), constrained_layout=True)
        for k, (name, audit) in enumerate(models):
            summaries = [audit.by_relation.get(r, audit.overall) if r != "ALL" else audit.overall for r in columns]
            mvr = np.array([s.mvr for s in summaries])
            lower = np.maximum(mvr - np.array([s.mvr_ci_low for s in summaries]), 0.0)
            upper = np.maximum(np.array([s.mvr_ci_high for s in summaries]) - mvr, 0.0)
            positions = np.arange(len(columns)) + (k - (len(models) - 1) / 2) * width
            ax.bar(positions, mvr, width * 0.92, yerr=[lower, upper], capsize=2, label=name,
                   color=_COLORS[k % len(_COLORS)], error_kw={"linewidth": 0.7})
        ax.set_xticks(range(len(columns)), labels=columns)
        ax.set_ylabel("MVR (95% Wilson CI)")
        ax.set_ylim(0, 1.0)
        ax.legend(ncol=min(len(models), 4), frameon=False)
    return _save(fig, base_path, formats)


def plot_confidence_drop(report: AuditReport, base_path: str, formats: Sequence[str] = ("png", "pdf")) -> list[str]:
    """Box plots of ``conf(x) - conf(x')`` over non-flipped pairs, per relation and model.

    Raises:
        ValueError: If the report was produced without per-pair records.
    """
    relations = _relations_present(report)
    models = list(report.models.items())
    if not all(audit.pair_records for _, audit in models):
        raise ValueError("Per-pair records are required (AuditConfig.include_pair_records=True).")
    width = 0.8 / max(len(models), 1)
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_IN, 2.6), constrained_layout=True)
        for k, (name, audit) in enumerate(models):
            data, positions = [], []
            for i, rel in enumerate(relations):
                drops = [r["confidence_drop"] for r in audit.pair_records if r["relation"] == rel and not r["violated"]]
                if drops:
                    data.append(drops)
                    positions.append(i + (k - (len(models) - 1) / 2) * width)
            if not data:
                continue
            color = _COLORS[k % len(_COLORS)]
            box = ax.boxplot(data, positions=positions, widths=width * 0.85, patch_artist=True, showfliers=False,
                             medianprops={"color": "black", "linewidth": 0.8})
            for patch in box["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.75)
            ax.plot([], [], color=color, linewidth=6, label=name)
        ax.axhline(0.0, color="gray", linewidth=0.6, linestyle="--")
        ax.set_xticks(range(len(relations)), labels=relations)
        ax.set_ylabel("Confidence drop (non-flipped pairs)")
        ax.legend(ncol=min(len(models), 4), frameon=False)
    return _save(fig, base_path, formats)


def plot_mpd_vs_mvr(report: AuditReport, base_path: str, formats: Sequence[str] = ("png", "pdf")) -> list[str]:
    """Scatter of MPD (JSD) against MVR, one point per (model, relation)."""
    relations = _relations_present(report)
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_IN, 2.8), constrained_layout=True)
        for k, (name, audit) in enumerate(report.models.items()):
            for m, rel in enumerate(relations):
                if rel not in audit.by_relation:
                    continue
                s = audit.by_relation[rel]
                ax.scatter(s.mvr, s.mpd_jsd, color=_COLORS[k % len(_COLORS)], marker=_MARKERS[m % len(_MARKERS)],
                           s=28, edgecolor="black", linewidth=0.4)
        for k, name in enumerate(report.models):
            ax.scatter([], [], color=_COLORS[k % len(_COLORS)], marker="o", label=name)
        for m, rel in enumerate(relations):
            ax.scatter([], [], color="white", edgecolor="black", marker=_MARKERS[m % len(_MARKERS)], label=rel)
        ax.set_xlabel("MVR")
        ax.set_ylabel("MPD (JSD, bits)")
        ax.legend(frameon=False, fontsize=6, ncol=2)
    return _save(fig, base_path, formats)


def plot_sweep(rows: Sequence[dict[str, Any]], base_path: str, x_label: str, metric: str = "mvr",
               group_by: str = "model", relation: str | None = "ALL", title: str = "",
               formats: Sequence[str] = ("png", "pdf")) -> list[str]:
    """Line plot of ``metric`` against the swept variable ``x``.

    Args:
        rows: Flat sweep rows (each has ``x``, ``model``, ``relation`` and metric keys).
        base_path: Output path without extension.
        x_label: Axis label for ``x``.
        metric: Row key to plot (``mvr``, ``mpd_jsd``, ``degradation_rate`` ...).
        group_by: ``"model"`` or ``"relation"`` - one line per group.
        relation: Keep only this relation; ``None`` keeps every relation except ``ALL``.
        title: Optional figure title.
    """
    selected = [r for r in rows if (r["relation"] == relation if relation is not None else r["relation"] != "ALL")]
    if not selected:
        raise ValueError("No rows left to plot after filtering.")
    groups = sorted({str(r[group_by]) for r in selected})
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_IN, 2.6), constrained_layout=True)
        for k, group in enumerate(groups):
            series = sorted((r for r in selected if str(r[group_by]) == group), key=lambda r: r["x"])
            xs = np.array([r["x"] for r in series], dtype=float)
            ys = np.array([r[metric] for r in series], dtype=float)
            color = _COLORS[k % len(_COLORS)]
            ax.plot(xs, ys, marker=_MARKERS[k % len(_MARKERS)], markersize=3.5, linewidth=1.0, color=color, label=group)
            if metric == "mvr" and all("mvr_ci_low" in r for r in series):
                ax.fill_between(xs, [r["mvr_ci_low"] for r in series], [r["mvr_ci_high"] for r in series],
                                color=color, alpha=0.15, linewidth=0)
        ax.set_xlabel(x_label)
        ax.set_ylabel(metric.upper().replace("_", " "))
        if title:
            ax.set_title(title)
        ax.legend(frameon=False)
    return _save(fig, base_path, formats)
