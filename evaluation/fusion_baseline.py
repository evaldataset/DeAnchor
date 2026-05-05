"""Fusion baseline: raw ML score vs LLM fl vs [raw ML, LLM fl] logistic probe.

Directly tests the paper's "no incremental value over raw ML score" claim
by fitting three logistic regressions on the same balanced FP+TP test set
and comparing AUROC and FP/TP separation. A proper conditional test for
incremental signal is a likelihood-ratio test of the nested models.

Usage:
    python evaluation/fusion_baseline.py \
        --input experiments/results/scoreaware_ieee_cis_200.jsonl \
        --output experiments/results/fusion_baseline.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def load(path):
    rows = []
    with open(path) as f:
        for ln in f:
            r = json.loads(ln)
            fp = r.get("fp_explanation") or {}
            fl = fp.get("fraud_likelihood") if isinstance(fp, dict) else None
            if not isinstance(fl, (int, float)):
                continue
            orig = r.get("original") or {}
            ml = orig.get("fraud_score") or r.get("ml_score")
            y = orig.get("is_fraud")
            if ml is None or y is None:
                continue
            rows.append((float(ml), float(fl), int(y)))
    return np.array(rows)


def cv_auc(X, y, seed=42, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = np.zeros(len(y))
    for tr, va in skf.split(X, y):
        lr = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
        preds[va] = lr.predict_proba(X[va])[:, 1]
    return float(roc_auc_score(y, preds)), preds


def group_stats(p, y):
    tp = p[y == 1]
    fp = p[y == 0]
    delta = float(tp.mean() - fp.mean())
    pooled = np.sqrt((tp.var(ddof=1) + fp.var(ddof=1)) / 2)
    d = float(delta / pooled) if pooled > 0 else 0.0
    return {
        "fp_mean": round(float(fp.mean()), 4),
        "tp_mean": round(float(tp.mean()), 4),
        "delta": round(delta, 4),
        "cohens_d": round(d, 3),
        "n_fp": int((y == 0).sum()),
        "n_tp": int((y == 1).sum()),
    }


def lr_test(X_full, X_reduced, y):
    """Likelihood-ratio test: does adding features in X_full improve over X_reduced?"""
    lr_full = LogisticRegression(max_iter=1000, C=1e6).fit(X_full, y)
    lr_red = LogisticRegression(max_iter=1000, C=1e6).fit(X_reduced, y)
    p_full = np.clip(lr_full.predict_proba(X_full)[:, 1], 1e-12, 1 - 1e-12)
    p_red = np.clip(lr_red.predict_proba(X_reduced)[:, 1], 1e-12, 1 - 1e-12)
    ll_full = float(np.sum(y * np.log(p_full) + (1 - y) * np.log(1 - p_full)))
    ll_red = float(np.sum(y * np.log(p_red) + (1 - y) * np.log(1 - p_red)))
    df = X_full.shape[1] - X_reduced.shape[1]
    lr_stat = 2 * (ll_full - ll_red)
    p = float(1 - stats.chi2.cdf(lr_stat, df)) if df > 0 else None
    return {
        "ll_full": round(ll_full, 4),
        "ll_reduced": round(ll_red, 4),
        "lr_stat": round(lr_stat, 4),
        "df": df,
        "p_value": p,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    data = load(args.input)
    if len(data) == 0:
        raise SystemExit("no valid rows")
    ml = data[:, 0:1]
    fl = data[:, 1:2]
    both = np.hstack([ml, fl])
    y = data[:, 2].astype(int)
    print(f"Loaded n={len(y)} (FP={int((y==0).sum())}, TP={int((y==1).sum())})")

    auc_ml, p_ml = cv_auc(ml, y)
    auc_fl, p_fl = cv_auc(fl, y)
    auc_both, p_both = cv_auc(both, y)

    print(f"AUROC: raw ML={auc_ml:.3f}  LLM fl={auc_fl:.3f}  raw+LLM={auc_both:.3f}")

    lrt_over_ml = lr_test(both, ml, y)
    lrt_over_fl = lr_test(both, fl, y)

    # Pearson between raw ML score and LLM fl
    r_pearson = float(np.corrcoef(ml[:, 0], fl[:, 0])[0, 1])

    summary = {
        "n": int(len(y)),
        "n_fp": int((y == 0).sum()),
        "n_tp": int((y == 1).sum()),
        "auroc": {
            "raw_ml_only": round(auc_ml, 4),
            "llm_fl_only": round(auc_fl, 4),
            "raw_ml_plus_llm_fl": round(auc_both, 4),
        },
        "group_stats": {
            "raw_ml_only": group_stats(p_ml, y),
            "llm_fl_only": group_stats(p_fl, y),
            "raw_ml_plus_llm_fl": group_stats(p_both, y),
        },
        "likelihood_ratio_test": {
            "does_llm_fl_add_over_raw_ml": lrt_over_ml,
            "does_raw_ml_add_over_llm_fl": lrt_over_fl,
        },
        "pearson_raw_vs_llm": round(r_pearson, 4),
        "interpretation": {
            "incremental_llm_over_raw": (
                "NO — LR test p > 0.05"
                if lrt_over_ml["p_value"] is not None and lrt_over_ml["p_value"] > 0.05
                else "possibly yes"
            ),
            "incremental_raw_over_llm": (
                "YES — LR test p <= 0.05"
                if lrt_over_fl["p_value"] is not None and lrt_over_fl["p_value"] <= 0.05
                else "no"
            ),
        },
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
