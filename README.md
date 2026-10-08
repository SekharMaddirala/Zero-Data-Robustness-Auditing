# Zero-Data Robustness Auditing in Pre-Trained NLP Classifiers

Official artifact repository for the paper:  
**"Zero-Data Robustness Auditing: Automated Metamorphic Testing and Invariant Violation Discovery in Pre-Trained NLP Classifiers"**  
Author: *Maddirala SriRama Sekhar* (Aditya University )

---

## 📌 Repository Structure & File Mapping

Because files are hosted in a flat root layout on this release branch, the table below maps each file to its logical package module and responsibility within the framework:

| File Name | Logical Module | Description |
| :--- | :--- | :--- |
| `config.py` | `root / config` | Global experimental parameters, random seeds (1234), and relation specifications. |
| `perturbation_generator.py` | `core/` | Synthesizes 600 seed sentences across 64 templates and derives 2,110 metamorphic test pairs. |
| `metamorphic_operators.py` | `core/` | Implements deterministic operators for MR-1 (Lexical), MR-2 (Syntax), MR-3 (Entity), and MR-4 (Litotes). |
| `model_wrapper.py` | `models/` | Hugging Face Transformers pipeline wrapper for DistilBERT, BERT, and RoBERTa architectures. |
| `audit_engine.py` | `runner/` | Orchestrates zero-data batch inference and metric accumulation. |
| `pipeline.py` | `analysis/` | Computes MVR (Equation 1), MPD with Jensen–Shannon divergence (Equation 2), and confidence degradation (Equation 3). |
| `sensitivity.py` | `analysis/` | Sweeps across sample sizes (40–200 seeds), random master seeds, and confidence thresholds (τ). |
| `plotting.py` | `analysis/` | Generates publication-grade distribution heatmaps and bar figures. |
| `annotation_key.csv` | `annotation/` | Ground-truth key of 150 stratified test pairs sampled across all four relations. |
| `annotation_sheet.csv` | `annotation/` | Blank blinded annotation form for independent validation studies. |
| `rater1.csv`, `rater2.csv` | `annotation/` | Independent blinded ratings (150 pairs) recording semantic polarity preservation and naturalness. |
| `results_report.md` | `results/` | Full Markdown log detailing empirical violation rates, confidence drops, and divergence values. |
| `fig_mvr_heatmap.png` | `figures/` | Visual heatmap of Metamorphic Violation Rates across models and relations (Fig. 1). |
| `fig_confidence_drop.png` | `figures/` | Confidence degradation rate (Δ > 0.05) on non-flipped invariant pairs (Fig. 2). |
| `test_operators.py` | `tests/` | Unit test suite verifying deterministic operator behavior and boundary preservation. |
| `test_pipeline.py` | `tests/` | End-to-end regression tests verifying scoring integrity against synthetic baselines. |

---

## 🛠️ Environment Setup

Install required Python packages:

```bash
pip install torch transformers pandas numpy scipy scikit-learn matplotlib seaborn
