"""DeAnchor audit toolkit — single user-facing entry point.

Runs the four-step DeAnchor audit on a paired score-aware/score-blind LLM-output JSONL
file and emits a deploy/skip/mitigate verdict from the three-gate decision framework.

Usage
-----
    python audit.py --inputs <paired_jsonl> [--ml-score-key fraud_score]

Input JSONL: one line per transaction, with both a score-aware and score-blind LLM
output, and the ML score and ground-truth label. Two accepted layouts:

  Layout A (already paired in one record):
      {"text": ..., "label": 0/1, "ml_score": 0.0-1.0,
       "fl_aware": 0.0-1.0, "fl_blind": 0.0-1.0}

  Layout B (per-record, two complementary files merged on transaction_id):
      Pass two files with --aware and --blind; we'll merge by id.

Outputs (stdout + optional JSON):
  - Score-blind LLM AUROC (B), score-aware AUROC (A), raw ML AUROC (M),
    fusion AUROC (F)
  - Paired controlled-ablation shift (Δ) with paired t-test and Pearson r
  - Three-gate decision verdict: deploy | mitigate | skip | narrate-only
  - JSON dump if --output is given

This entry point is documented in the paper as the canonical reproduction command.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation.decision_framework import evaluate_gates  # noqa: E402


def _load_paired(path: Path, ml_key: str = "ml_score") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            label = r.get("label", r.get("y", r.get("is_fraud")))
            ml = r.get(ml_key, r.get("fraud_score", r.get("score")))
            fla = r.get("fl_aware", r.get("aware_fl"))
            flb = r.get("fl_blind", r.get("blind_fl"))
            if None in (label, ml, fla, flb):
                continue
            rows.append({"label": int(label), "ml": float(ml),
                         "aware": float(fla), "blind": float(flb)})
    return rows


def _cv_auroc(X: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 42) -> float:
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    preds = np.zeros(len(y))
    for tr, va in skf.split(X, y):
        lr = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
        preds[va] = lr.predict_proba(X[va])[:, 1]
    return float(roc_auc_score(y, preds))


def run_audit(rows: list[dict[str, Any]], prevalence: float | None = None) -> dict[str, Any]:
    if len(rows) < 20:
        raise ValueError(f"Need ≥20 paired rows, got {len(rows)}.")
    y = np.array([r["label"] for r in rows], dtype=int)
    ml = np.array([r["ml"] for r in rows], dtype=float)
    aware = np.array([r["aware"] for r in rows], dtype=float)
    blind = np.array([r["blind"] for r in rows], dtype=float)

    # --- Step 1: paired controlled-ablation shift ---
    diff = aware - blind
    t_stat, t_p = stats.ttest_rel(aware, blind)
    r_score, _ = stats.pearsonr(ml, aware) if ml.std() > 0 else (float("nan"), 1.0)
    alpha_num = float(diff.mean())
    alpha_den = float(ml.mean() - blind.mean())
    alpha = alpha_num / alpha_den if abs(alpha_den) > 1e-6 else None

    # --- AUROCs (1-feature CV probes) ---
    auroc_M = _cv_auroc(ml.reshape(-1, 1), y)
    auroc_A = _cv_auroc(aware.reshape(-1, 1), y)
    auroc_B = _cv_auroc(blind.reshape(-1, 1), y)
    auroc_F = _cv_auroc(np.column_stack([ml, aware]), y)

    # --- Step 2: nested LR test (does aware add over raw ML?) ---
    lr_full = LogisticRegression(max_iter=1000, C=1e6).fit(np.column_stack([ml, aware]), y)
    lr_red = LogisticRegression(max_iter=1000, C=1e6).fit(ml.reshape(-1, 1), y)
    pf = np.clip(lr_full.predict_proba(np.column_stack([ml, aware]))[:, 1], 1e-12, 1 - 1e-12)
    pr = np.clip(lr_red.predict_proba(ml.reshape(-1, 1))[:, 1], 1e-12, 1 - 1e-12)
    ll_f = float(np.sum(y * np.log(pf) + (1 - y) * np.log(1 - pf)))
    ll_r = float(np.sum(y * np.log(pr) + (1 - y) * np.log(1 - pr)))
    chi2 = max(0.0, 2 * (ll_f - ll_r))
    lr_p = float(1 - stats.chi2.cdf(chi2, 1))

    # --- Step 3: TOST equivalence on AUROC difference (margin ±0.03) ---
    margin = 0.03
    diff_AUROC = auroc_F - auroc_M
    se = float(np.sqrt((auroc_F * (1 - auroc_F) + auroc_M * (1 - auroc_M)) / len(y)))
    z_lower = (diff_AUROC - (-margin)) / se if se > 0 else 0.0
    z_upper = ((margin) - diff_AUROC) / se if se > 0 else 0.0
    p_lower = float(1 - stats.norm.cdf(z_lower))
    p_upper = float(1 - stats.norm.cdf(z_upper))
    tost_equivalent = (p_lower < 0.05) and (p_upper < 0.05)

    # --- Step 4: gate verdict ---
    gates = evaluate_gates(B=auroc_B, A=auroc_A, M=auroc_M, F=auroc_F,
                           prevalence=prevalence)

    return {
        "n": int(len(rows)),
        "controlled_ablation": {
            "paired_shift_mean": round(alpha_num, 4),
            "paired_t": round(float(t_stat), 4),
            "paired_p": float(t_p),
            "alpha": round(alpha, 4) if alpha is not None else None,
            "pearson_r_score_vs_aware": round(float(r_score), 4),
        },
        "auroc": {
            "raw_ml": round(auroc_M, 4),
            "score_aware_llm": round(auroc_A, 4),
            "score_blind_llm": round(auroc_B, 4),
            "fusion": round(auroc_F, 4),
        },
        "incremental_value_test": {
            "chi2": round(chi2, 4),
            "p": lr_p,
            "interpretation": "LLM adds incremental value" if lr_p < 0.05 else "no incremental value",
        },
        "tost_equivalence": {
            "margin": margin,
            "auroc_diff_F_minus_M": round(diff_AUROC, 4),
            "p_lower": p_lower,
            "p_upper": p_upper,
            "equivalent_within_margin": bool(tost_equivalent),
        },
        "decision_framework": gates,
    }


def _print_verdict(result: dict[str, Any]) -> None:
    ca = result["controlled_ablation"]
    au = result["auroc"]
    iv = result["incremental_value_test"]
    eq = result["tost_equivalence"]
    g = result["decision_framework"]
    print("=" * 64)
    print(f"DeAnchor Audit Result   (n={result['n']})")
    print("=" * 64)
    print(f"  Paired shift (aware-blind):  {ca['paired_shift_mean']:+.4f}  "
          f"(t={ca['paired_t']:+.2f}, p={ca['paired_p']:.2e})")
    print(f"  Anchoring α:                  {ca['alpha']}")
    print(f"  Pearson r (score vs aware):   {ca['pearson_r_score_vs_aware']}")
    print()
    print(f"  AUROC raw ML:                 {au['raw_ml']:.4f}")
    print(f"  AUROC score-blind LLM:        {au['score_blind_llm']:.4f}")
    print(f"  AUROC score-aware LLM:        {au['score_aware_llm']:.4f}")
    print(f"  AUROC fusion:                 {au['fusion']:.4f}")
    print()
    print(f"  LR test (LLM over raw ML):    chi2={iv['chi2']:.2f}, p={iv['p']:.4f}  "
          f"=> {iv['interpretation']}")
    print(f"  TOST equivalence (±{eq['margin']}):    "
          f"{'YES' if eq['equivalent_within_margin'] else 'NO'}  "
          f"(diff={eq['auroc_diff_F_minus_M']:+.4f})")
    print()
    print(f"  Decision-framework gates:")
    for g_name, g_val in g.get("gates", {}).items():
        print(f"    • {g_name:30s} {g_val}")
    print()
    print(f"  >>> VERDICT: {g.get('verdict', 'UNKNOWN').upper()} <<<")
    if g.get("rationale"):
        print(f"      Rationale: {g['rationale']}")
    print("=" * 64)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--inputs", required=True, type=Path,
                   help="Paired-record JSONL (Layout A).")
    p.add_argument("--ml-score-key", default="ml_score",
                   help="Field name for the ML score (default: ml_score).")
    p.add_argument("--prevalence", type=float, default=None,
                   help="Operational prevalence; if absent, uses sample base rate.")
    p.add_argument("--output", type=Path, default=None,
                   help="Optional output JSON path.")
    args = p.parse_args()

    if not args.inputs.exists():
        print(f"ERROR: input file does not exist: {args.inputs}", file=sys.stderr)
        return 2
    rows = _load_paired(args.inputs, ml_key=args.ml_score_key)
    if not rows:
        print(f"ERROR: no usable paired records loaded from {args.inputs}.\n"
              "       Required fields: label, ml_score (or {ml-score-key}), "
              "fl_aware, fl_blind.", file=sys.stderr)
        return 2

    result = run_audit(rows, prevalence=args.prevalence)
    _print_verdict(result)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        print(f"  (written to {args.output})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
