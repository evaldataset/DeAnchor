"""Canonical raw ML-score baseline comparison.

Compares raw XGBoost output vs LLM score-aware fraud_likelihood on the same
sample of ML-flagged transactions. This produces the headline negative result:
LLM explanation layer adds no incremental discrimination over raw ML score.

Usage:
    python evaluation/raw_ml_baseline.py \
        --llm_results experiments/results/scoreaware_ieee_cis_200.jsonl \
        --output experiments/results/raw_ml_baseline_comparison.json

Outputs JSON with:
  - raw_ml: {fp_mean, tp_mean, delta, cohens_d, p_value}
  - llm_score_aware: {fp_mean, tp_mean, delta, cohens_d, p_value}
  - correlation: Pearson r between raw ML score and LLM fl on same items
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats


def load_paired(path: str):
    fp_ml, tp_ml, fp_llm, tp_llm = [], [], [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            tx = r.get("original", {})
            cat = tx.get("category", "")
            ml = r.get("ml_score") or tx.get("fraud_score")
            exp = r.get("fp_explanation") or r.get("assessment") or {}
            if not isinstance(exp, dict) or exp.get("parse_error"):
                continue
            fl = exp.get("fraud_likelihood")
            if not isinstance(ml, (int, float)) or not isinstance(fl, (int, float)):
                continue
            if cat == "false_positive":
                fp_ml.append(ml); fp_llm.append(fl)
            elif cat == "true_positive":
                tp_ml.append(ml); tp_llm.append(fl)
    return (
        np.array(fp_ml), np.array(tp_ml),
        np.array(fp_llm), np.array(tp_llm),
    )


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def analyze(path: str) -> dict:
    fp_ml, tp_ml, fp_llm, tp_llm = load_paired(path)
    print(f"Loaded: FP n={len(fp_ml)}, TP n={len(tp_ml)}")

    # Raw ML baseline
    d_raw = cohens_d(tp_ml, fp_ml)
    t_raw, p_raw = sp_stats.ttest_ind(tp_ml, fp_ml)
    raw = {
        "fp_mean": round(float(fp_ml.mean()), 4),
        "tp_mean": round(float(tp_ml.mean()), 4),
        "delta": round(float(tp_ml.mean() - fp_ml.mean()), 4),
        "cohens_d": round(d_raw, 3),
        "p_value": round(float(p_raw), 6),
    }

    # LLM score-aware
    d_llm = cohens_d(tp_llm, fp_llm)
    t_llm, p_llm = sp_stats.ttest_ind(tp_llm, fp_llm)
    llm = {
        "fp_mean": round(float(fp_llm.mean()), 4),
        "tp_mean": round(float(tp_llm.mean()), 4),
        "delta": round(float(tp_llm.mean() - fp_llm.mean()), 4),
        "cohens_d": round(d_llm, 3),
        "p_value": round(float(p_llm), 6),
    }

    # Correlation on all paired items
    all_ml = np.concatenate([fp_ml, tp_ml])
    all_llm = np.concatenate([fp_llm, tp_llm])
    r, p_corr = sp_stats.pearsonr(all_ml, all_llm)

    results = {
        "n_fp": len(fp_ml),
        "n_tp": len(tp_ml),
        "raw_ml": raw,
        "llm_score_aware": llm,
        "correlation": {
            "pearson_r": round(float(r), 3),
            "p_value": round(float(p_corr), 6),
        },
        "incremental_value": {
            "llm_delta_minus_raw_delta": round(llm["delta"] - raw["delta"], 4),
            "llm_d_minus_raw_d": round(d_llm - d_raw, 3),
            "llm_adds_discrimination": abs(llm["delta"]) > abs(raw["delta"]),
        },
    }

    print("\n=== Raw ML Score ===")
    for k, v in raw.items():
        print(f"  {k}: {v}")
    print("\n=== LLM Score-aware ===")
    for k, v in llm.items():
        print(f"  {k}: {v}")
    print(f"\nPearson r(ml_score, llm_fl) = {r:.3f}")
    print(f"LLM incremental discrimination: "
          f"{'YES' if results['incremental_value']['llm_adds_discrimination'] else 'NO'}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_results", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results = analyze(args.llm_results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
