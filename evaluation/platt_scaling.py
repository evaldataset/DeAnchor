"""Platt scaling (supervised logistic remapping) for LLM fraud_likelihood.

Fits a logistic regression on LLM output logits to improve calibration (ECE)
and FP/TP separation. Uses stratified 5-fold CV with no sample leakage
between folds.

Note: This is supervised post-hoc remapping, NOT improved reasoning.
Any improvement in Δ reflects the logistic transform amplifying small
differences in the narrow raw output range (0.85--0.96).

Usage:
    python evaluation/platt_scaling.py \
        --input experiments/results/scoreaware_ieee_cis_200.jsonl \
        --output experiments/results/platt_scaled_results.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from evaluation.calibration_analysis import compute_ece


def load_fls_labels(path: str):
    fls, labels = [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            exp = r.get("fp_explanation") or r.get("assessment") or {}
            if not isinstance(exp, dict) or exp.get("parse_error"):
                continue
            fl = exp.get("fraud_likelihood")
            is_fraud = r.get("original", {}).get("is_fraud")
            if isinstance(fl, (int, float)) and is_fraud in (0, 1):
                fls.append(float(fl))
                labels.append(int(is_fraud))
    return np.array(fls), np.array(labels)


def platt_scale(scores: np.ndarray, labels: np.ndarray, n_folds: int = 5,
                seed: int = 42) -> dict:
    """Cross-validated Platt scaling on LOG-ODDS of raw scores.

    Platt's original formulation fits a logistic regression on the log-odds transform of the raw model output; this is
    the variant we use in the paper. Fits per-fold to avoid leakage; out-of-fold predictions form calibrated[idx].
    """
    eps = 1e-6
    scores_clip = np.clip(scores, eps, 1 - eps)
    logits = np.log(scores_clip / (1 - scores_clip))

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    calibrated = np.zeros_like(scores)
    fold_eces_raw, fold_eces_cal = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(scores, labels)):
        train_logits, train_l = logits[train_idx], labels[train_idx]
        val_logits, val_l = logits[val_idx], labels[val_idx]
        val_raw = scores[val_idx]

        lr = LogisticRegression()
        lr.fit(train_logits.reshape(-1, 1), train_l)
        val_cal = lr.predict_proba(val_logits.reshape(-1, 1))[:, 1]
        calibrated[val_idx] = val_cal

        raw_ece = compute_ece(val_raw.tolist(), val_l.tolist())["ece"]
        cal_ece = compute_ece(val_cal.tolist(), val_l.tolist())["ece"]
        fold_eces_raw.append(raw_ece)
        fold_eces_cal.append(cal_ece)

    return {
        "calibrated_scores": calibrated,
        "fold_eces_raw": fold_eces_raw,
        "fold_eces_cal": fold_eces_cal,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--n_folds", type=int, default=5)
    args = parser.parse_args()

    scores, labels = load_fls_labels(args.input)
    print(f"Loaded n={len(scores)}, positive rate={labels.mean():.3f}")

    result = platt_scale(scores, labels, n_folds=args.n_folds)
    cal = result["calibrated_scores"]

    raw_ece = compute_ece(scores.tolist(), labels.tolist())
    cal_ece = compute_ece(cal.tolist(), labels.tolist())

    # FP/TP separation
    fp_mask = labels == 0
    tp_mask = labels == 1
    raw_delta = float(scores[tp_mask].mean() - scores[fp_mask].mean())
    cal_delta = float(cal[tp_mask].mean() - cal[fp_mask].mean())

    summary = {
        "n": len(scores),
        "raw": {
            "ece": raw_ece["ece"],
            "fp_mean": round(float(scores[fp_mask].mean()), 4),
            "tp_mean": round(float(scores[tp_mask].mean()), 4),
            "delta": round(raw_delta, 4),
        },
        "calibrated": {
            "ece": cal_ece["ece"],
            "fp_mean": round(float(cal[fp_mask].mean()), 4),
            "tp_mean": round(float(cal[tp_mask].mean()), 4),
            "delta": round(cal_delta, 4),
        },
        "per_fold_ece_raw": [round(x, 4) for x in result["fold_eces_raw"]],
        "per_fold_ece_calibrated": [round(x, 4) for x in result["fold_eces_cal"]],
        "per_fold_ece_calibrated_mean": round(float(np.mean(result["fold_eces_cal"])), 4),
        "per_fold_ece_calibrated_std": round(float(np.std(result["fold_eces_cal"])), 4),
    }

    print(f"\nRaw ECE: {summary['raw']['ece']:.3f}, Δ: {raw_delta:+.4f}")
    print(f"Calibrated ECE: {summary['calibrated']['ece']:.3f}, Δ: {cal_delta:+.4f}")
    print(f"Per-fold calibrated ECE: {summary['per_fold_ece_calibrated_mean']:.3f} ± "
          f"{summary['per_fold_ece_calibrated_std']:.3f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
