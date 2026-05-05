# DeAnchor — Reproducible Audit Benchmark for Score-Conditioned Dependence in Hybrid ML+LLM Pipelines

[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/data-CC--BY--4.0-orange.svg)](LICENSE-DATA)

Reference implementation and released benchmark for the paper
_"The DeAnchor Audit: A Reproducible Benchmark for Score-Conditioned Dependence in Hybrid ML+LLM Pipelines"_ (NeurIPS 2026 **Evaluations & Datasets (E&D) Track** submission — formerly Datasets & Benchmarks).

**Evaluative role of this contribution.** DeAnchor is an _executable audit toolkit and paired-response evaluation dataset_ whose primary claim is that hybrid ML+LLM pipelines deserve a pre-deployment audit because their LLM layer can be score-conditioned mirroring rather than independent judgment. The contribution supports evaluative claims of the form "does adding this LLM layer add measurable, independent discriminative value?" under controlled-ablation, nested-LR, and TOST assumptions. The audit toolkit is the central reusable artifact; the released paired-response dataset documents the eight-model × four-regime evaluative landscape behind the toolkit.

## What this repository provides

1. **Audit toolkit** — `audit.py`, a single command that produces a
   deploy/skip/mitigate verdict from any `(transaction, label, ML score, LLM
output)` paired JSONL.
2. **Released paired LLM-response dataset** — approximately 3,500 LLM responses
   (≈650 strict score-aware/score-blind paired matches, plus condition variants
   and within-subject paired-by-design records) across 8 LLM families, 4 feature
   regimes, plus a synthetic medical stress test.
3. **Decision framework** — three-gate pre-deployment decision tree implemented
   in `evaluation/decision_framework.py`.
4. **Public replication protocol draft** — locked H1–H3 in
   `paper/main.tex` Appendix Z (public draft; OSF posting before camera-ready).

## Single-command audit

```bash
python audit.py --inputs experiments/results/audit_fixture_ieee_50.jsonl
```

Output: paired controlled-ablation shift, AUROCs (raw ML, score-blind LLM, score-aware LLM, fusion), nested LR test, TOST equivalence, three-gate decision-framework verdict.

## Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Exact pin file: `requirements_exact.txt` (regenerated from a working install).

CPU-only Docker (no GPU required for the audit toolkit):

```bash
docker build -t deanchor .
docker run --rm deanchor python scripts/validate_paper_numbers.py
```

## Required environment variables (re-running collected experiments only)

| Variable            | Purpose                                                            |
| ------------------- | ------------------------------------------------------------------ |
| `OPENAI_API_KEY`    | GPT-4o, GPT-4o-mini, o3-mini                                       |
| `ANTHROPIC_API_KEY` | Claude-Haiku-4.5                                                   |
| `GEMINI_API_KEY`    | Gemini-2.5-flash, Gemini-2.5-pro thinking                          |
| `QWEN_MODEL_DIR`    | Optional: local Qwen2.5-7B-Instruct path (for offline replication) |

The audit toolkit itself does not call any LLM API — it only consumes already-collected paired LLM outputs.

## End-to-end pipeline

```bash
# 1. Data + ML baseline
python scripts/download_data.py --dataset ieee_cis
python ml_baseline/feature_engineering.py --input ieee_cis
python ml_baseline/train_xgboost.py --dataset ieee_cis

# 2. LLM input subsets (canonical stratified sampling with manifests)
python scripts/generate_llm_inputs.py --dataset ieee_cis --n_fp 100 --n_tp 100

# 3. Core experiments (re-collect paired LLM outputs)
python experiments/run_controlled_ablation.py \
    --input experiments/results/llm_input_ieee_cis_200.jsonl \
    --model gpt-4o-mini
python experiments/run_dose_response.py \
    --input experiments/results/llm_input_ieee_cis_200.jsonl
python experiments/run_counterfactual.py \
    --input experiments/results/llm_input_ieee_cis_200.jsonl
python experiments/run_medical_pilot.py
python experiments/run_extension_battery.py    # 8-model + mitigation + paraphrase + format

# 4. Audit + analysis
python audit.py \
    --inputs experiments/results/audit_fixture_ieee_50.jsonl \
    --output experiments/results/audit_ieee.json
python evaluation/raw_ml_baseline.py \
    --llm_results experiments/results/scoreaware_ieee_cis_200.jsonl \
    --output experiments/results/raw_ml_baseline_comparison.json
python evaluation/regression_analysis.py \
    --output experiments/results/regression_analysis.json
python evaluation/platt_scaling.py \
    --input experiments/results/scoreaware_ieee_cis_200.jsonl \
    --output experiments/results/platt_scaled_results.json

# 5. Validation
pytest tests/ -q
python scripts/validate_paper_numbers.py
```

## Released benchmark composition

| Setting                  | n        | Models                                                                                                          |
| ------------------------ | -------- | --------------------------------------------------------------------------------------------------------------- |
| IEEE-CIS (anonymized)    | 200      | GPT-4o-mini (primary), GPT-4o, Qwen2.5-7B, Claude-Haiku-4.5, Gemini-2.5-flash, o3-mini, Gemini-2.5-pro thinking |
| PaySim                   | 200      | mixed (GPT-4o/4o-mini + Qwen-7B; see paper limitations)                                                         |
| Enriched production-like | 100      | Qwen-7B, Gemma-9B, GPT-4o-mini                                                                                  |
| UCI-Adult-style profiles | 100      | GPT-4o-mini (synthetic profiles, real GradientBoosting baseline)                                                |
| Medical (synthetic)      | 30 + 100 | GPT-4o-mini                                                                                                     |

## Claim → code → artifact map

| Paper claim                                  | Script                                        | Artifact                                                            |
| -------------------------------------------- | --------------------------------------------- | ------------------------------------------------------------------- |
| Controlled ablation (α=0.53, GPT-4o-mini)    | `experiments/run_controlled_ablation.py`      | `controlled_ablation_ieee_{with,without}_score.jsonl`               |
| 8-model cross-family ablation (Section 5.3)  | `experiments/run_extension_battery.py`        | `extension_crossmodel_audit.json`, `extension_reasoning_audit.json` |
| Mitigation Zoo (5 strategies)                | `experiments/run_extension_battery.py`        | `mitigation_zoo_audit.json`                                         |
| Bayesian prior dose-response                 | `experiments/run_extension_battery.py`        | `prior_dose_response_audit.json`                                    |
| Prompt paraphrase + score format robustness  | `experiments/run_extension_battery.py`        | `prompt_paraphrase_audit.json`, `score_format_audit.json`           |
| Three-gate decision framework                | `evaluation/decision_framework.py`            | (returned by `audit.py`)                                            |
| Fusion LR test (no incremental)              | `evaluation/fusion_baseline.py`               | `fusion_baseline.json`                                              |
| Optimal combiner normative baseline          | `evaluation/normative_baseline.py`            | `normative_baseline.json`                                           |
| TOST equivalence + decision curve            | `evaluation/equivalence_and_decisioncurve.py` | `equivalence_decisioncurve.json`                                    |
| Counterfactual asymmetry                     | `experiments/run_counterfactual.py`           | `causal_anchoring.json`                                             |
| Medical cross-domain pilot                   | `experiments/run_medical_pilot.py`            | `medical_domain.json`, `medical_n100_audit.json`                    |
| Mechanism-distinguishing (base-rate priming) | `experiments/run_final_acceptance.py`         | `mechanism_audit.json`                                              |
| Multi-seed ablation stability                | `experiments/run_final_acceptance.py`         | `multiseed_ablation_audit.json`                                     |
| Platt scaling                                | `evaluation/platt_scaling.py`                 | `platt_scaled_results.json`                                         |

## Reproducibility

- Code: MIT License (see `LICENSE`).
- Data: CC BY 4.0 (see `LICENSE-DATA`).
- Random seed `42` is locked across XGBoost, LLM seeded calls (where supported), bootstrap, and 5-fold CV.
- Estimated cost to fully replicate the released benchmark: ≈ \$15 in API charges + 8 hours wall-clock on a single GPU.
- All paper-cited numbers are auto-validated via `scripts/validate_paper_numbers.py` (62 checks).

## Known limitations (also discussed in paper §7 / `CHECK.md`)

1. **PaySim n=200 is mixed-model**: 75 GPT-4o/4o-mini + 125 Qwen-7B. We provide it as cost-honest disclosure but it should not be read as a clean cross-dataset replication.
2. **UCI-Adult-style profiles are synthetic** (deterministic `random.Random(42)`); the GradientBoosting baseline is real, but the profiles are not from the UCI Adult Census dataset.
3. **Reasoning models (o3-mini, Gemini-2.5-pro thinking) are evaluated with heterogeneous n**; CIs are wide and reported in Appendix V.
4. **No human expert evaluation**; LLM-as-Judge has +1.4-point self-preference inflation (Appendix I).
5. **Mitigation Zoo's quantization gate** was tested on a uniformly high-score audit sample; behavior on mid-low score regimes is untested.
6. **Replication protocol** is documented in Appendix Z as a public draft; an OSF posting will be added before camera-ready.

## Citation

(Anonymous submission)
