"""Normative baseline: is the score-aware LLM as good as an optimal combiner
of raw ML + score-BLIND LLM output?

Addresses Reviewer A Q2 and the AC's "normative baseline" requirement.

The key idea: if the LLM were using the ML score rationally, its score-aware output
should be at least as discriminative as a theoretically-reasonable combiner that
uses raw ML score + the LLM's own score-blind judgment (which represents the LLM's
independent evidence about the transaction).

Comparison:
  (A) raw ML only                         - baseline
  (B) score-aware LLM only                - what the LLM actually produces
  (C) raw ML + score-blind LLM (logistic) - optimal combiner, normative ceiling
  (D) raw ML + score-aware LLM (logistic) - fusion

If (C) > (B): the LLM fails to integrate its own independent evidence with the
  ML score; a simple logistic combiner using the LLM's score-blind output plus
  raw ML beats the LLM's own score-aware reasoning.
If (C) ≈ (A): the score-blind LLM has no independent signal to contribute,
  making score reliance rational rather than irrational.

Output: experiments/results/normative_baseline.json
"""

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

BASE = Path(__file__).resolve().parent.parent
AWARE = BASE / "experiments" / "results" / "scoreaware_ieee_cis_200.jsonl"
BLIND = BASE / "experiments" / "results" / "scoreblind_ieee_cis_200_final.jsonl"
OUTPUT = BASE / "experiments" / "results" / "normative_baseline.json"


def load_paired():
    """Load score-aware and score-blind records and pair them by transaction_id."""
    aware_by_tid = {}
    with open(AWARE) as f:
        for ln in f:
            r = json.loads(ln)
            orig = r.get("original", {})
            tid = orig.get("transaction_id")
            fp = r.get("fp_explanation") or {}
            fl = fp.get("fraud_likelihood") if isinstance(fp, dict) else None
            ml = orig.get("fraud_score")
            y = orig.get("is_fraud")
            if tid is None or not isinstance(fl, (int, float)) or ml is None or y is None:
                continue
            aware_by_tid[tid] = {"fl_aware": float(fl), "ml": float(ml), "y": int(y)}

    rows = []
    with open(BLIND) as f:
        for ln in f:
            r = json.loads(ln)
            orig = r.get("original", {})
            tid = orig.get("transaction_id")
            ba = r.get("blind_assessment", {})
            fl_blind = ba.get("fraud_likelihood") if isinstance(ba, dict) else None
            if tid is None or not isinstance(fl_blind, (int, float)):
                continue
            if tid not in aware_by_tid:
                continue
            a = aware_by_tid[tid]
            rows.append(
                {
                    "tid": tid,
                    "ml": a["ml"],
                    "fl_aware": a["fl_aware"],
                    "fl_blind": float(fl_blind),
                    "y": a["y"],
                }
            )
    return rows


def cv_auc(X, y, seed=42):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    preds = np.zeros(len(y))
    for tr, va in skf.split(X, y):
        lr = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
        preds[va] = lr.predict_proba(X[va])[:, 1]
    return float(roc_auc_score(y, preds)), preds


def group_stats(scores, y):
    tp = scores[y == 1]
    fp = scores[y == 0]
    delta = float(tp.mean() - fp.mean())
    pooled = np.sqrt((tp.var(ddof=1) + fp.var(ddof=1)) / 2)
    d = float(delta / pooled) if pooled > 0 else 0.0
    return {"delta": round(delta, 4), "cohens_d": round(d, 3)}


def bootstrap_ci(X, y, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            auc, _ = cv_auc(X[idx], y[idx])
            aucs.append(auc)
        except Exception:
            pass
    return np.percentile(aucs, [2.5, 97.5]).tolist() if aucs else [None, None]


def main():
    rows = load_paired()
    print(f"Paired records: {len(rows)}")
    if len(rows) < 50:
        raise SystemExit("too few paired records")

    ml = np.array([r["ml"] for r in rows]).reshape(-1, 1)
    fl_aware = np.array([r["fl_aware"] for r in rows]).reshape(-1, 1)
    fl_blind = np.array([r["fl_blind"] for r in rows]).reshape(-1, 1)
    y = np.array([r["y"] for r in rows])

    fp_count = int((y == 0).sum())
    tp_count = int((y == 1).sum())
    print(f"FP={fp_count}, TP={tp_count}")

    # (A) raw ML alone
    auc_A, _ = cv_auc(ml, y)
    ci_A = bootstrap_ci(ml, y)

    # (B) score-aware LLM alone
    auc_B, _ = cv_auc(fl_aware, y)
    ci_B = bootstrap_ci(fl_aware, y)

    # (C) optimal combiner: raw ML + score-blind LLM
    ml_blind = np.hstack([ml, fl_blind])
    auc_C, _ = cv_auc(ml_blind, y)
    ci_C = bootstrap_ci(ml_blind, y)

    # (D) fusion: raw ML + score-aware LLM
    ml_aware = np.hstack([ml, fl_aware])
    auc_D, _ = cv_auc(ml_aware, y)
    ci_D = bootstrap_ci(ml_aware, y)

    # (E) score-blind LLM alone
    auc_E, _ = cv_auc(fl_blind, y)
    ci_E = bootstrap_ci(fl_blind, y)

    # Effect sizes on raw scores
    ga = group_stats(ml[:, 0], y)
    gb = group_stats(fl_aware[:, 0], y)
    gc_scores = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr, va in skf.split(ml_blind, y):
        lr = LogisticRegression(max_iter=1000).fit(ml_blind[tr], y[tr])
        gc_scores[va] = lr.predict_proba(ml_blind[va])[:, 1]
    gc = group_stats(gc_scores, y)

    ge = group_stats(fl_blind[:, 0], y)

    summary = {
        "n": len(rows),
        "n_fp": fp_count,
        "n_tp": tp_count,
        "scores": {
            "A_raw_ml_only": {
                "auroc": round(auc_A, 4),
                "ci_95": [round(x, 4) for x in ci_A],
                "effect": ga,
            },
            "B_scoreaware_llm_only": {
                "auroc": round(auc_B, 4),
                "ci_95": [round(x, 4) for x in ci_B],
                "effect": gb,
                "note": "what the LLM actually outputs when shown the score",
            },
            "C_optimal_combiner_ml_plus_blind_llm": {
                "auroc": round(auc_C, 4),
                "ci_95": [round(x, 4) for x in ci_C],
                "effect": gc,
                "note": "normative ceiling: logistic combiner of raw ML + score-BLIND LLM output",
            },
            "D_fusion_ml_plus_aware_llm": {
                "auroc": round(auc_D, 4),
                "ci_95": [round(x, 4) for x in ci_D],
                "note": "empirical fusion with score-aware LLM (from prior analysis)",
            },
            "E_scoreblind_llm_only": {
                "auroc": round(auc_E, 4),
                "ci_95": [round(x, 4) for x in ci_E],
                "effect": ge,
                "note": "the LLM's independent signal before seeing the score",
            },
        },
        "comparisons": {
            "B_vs_A": {
                "auroc_diff": round(auc_B - auc_A, 4),
                "interpretation": "score-aware LLM vs raw ML alone",
            },
            "C_vs_A": {
                "auroc_diff": round(auc_C - auc_A, 4),
                "interpretation": "optimal combiner vs raw ML alone --- does score-blind LLM contribute?",
            },
            "C_vs_B": {
                "auroc_diff": round(auc_C - auc_B, 4),
                "interpretation": "optimal combiner vs score-aware LLM --- does LLM integrate its own evidence?",
            },
        },
        "normative_conclusion": {
            "raw_ml_auroc": round(auc_A, 4),
            "optimal_ceiling_auroc": round(auc_C, 4),
            "scoreaware_llm_auroc": round(auc_B, 4),
            "ceiling_minus_scoreaware": round(auc_C - auc_B, 4),
            "verdict": (
                "optimal combiner EXCEEDS score-aware LLM"
                if auc_C - auc_B > 0.005
                else "optimal combiner MATCHES score-aware LLM within noise"
            ),
            "interpretation": (
                "The LLM fails to integrate its own independent evidence: a simple "
                "logistic combiner using raw ML + the LLM's own score-blind output "
                "matches or exceeds the LLM's score-aware output."
                if auc_C >= auc_B - 0.005
                else "The LLM's score-aware output captures information beyond the "
                "simple combiner, suggesting effective non-linear integration."
            ),
        },
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
