"""All new analyses required by REVISION.md.

Outputs: experiments/results/revision_analyses.json

Sections:
1. Fusion baseline with bootstrap AUROC CI
2. Disagreement-region analysis (score decile breakdown)
3. Prevalence-aware evaluation (simulated post-flag distribution)
4. Mixed-effects / within-item paired regression
5. Partial correlation (ML score → LLM output, controlling for label)
6. Equivalence-style minimum detectable effect
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

BASE = Path(__file__).resolve().parent.parent
SCOREAWARE = BASE / "experiments" / "results" / "scoreaware_ieee_cis_200.jsonl"
CONTROLLED_WITH = BASE / "experiments" / "results" / "controlled_ablation_ieee_with_score.jsonl"
CONTROLLED_WITHOUT = BASE / "experiments" / "results" / "controlled_ablation_ieee_without_score.jsonl"
OUTPUT = BASE / "experiments" / "results" / "revision_analyses.json"


def load_scoreaware():
    rows = []
    with open(SCOREAWARE) as f:
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


def load_controlled():
    """Load paired controlled ablation data."""
    def _load(path):
        out = {}
        with open(path) as f:
            for ln in f:
                r = json.loads(ln)
                # Try multiple format variants
                fl = None
                for key in ["assessment", "fp_explanation"]:
                    sub = r.get(key) or {}
                    if isinstance(sub, dict) and isinstance(sub.get("fraud_likelihood"), (int, float)):
                        fl = float(sub["fraud_likelihood"])
                        break
                if fl is None:
                    continue
                orig = r.get("original", {})
                tid = orig.get("transaction_id") or r.get("transaction_id")
                ml = orig.get("fraud_score") or r.get("ml_score")
                y = orig.get("is_fraud")
                if tid is not None:
                    out[tid] = {"fl": fl, "ml": float(ml) if ml else 0, "y": int(y) if y is not None else 0}
        return out
    w = _load(CONTROLLED_WITH)
    wo = _load(CONTROLLED_WITHOUT)
    common = sorted(set(w) & set(wo))
    data = []
    for tid in common:
        data.append({
            "tid": tid,
            "fl_with": w[tid]["fl"],
            "fl_without": wo[tid]["fl"],
            "ml": w[tid]["ml"],
            "y": w[tid]["y"],
        })
    return data


# === 1. Fusion baseline with bootstrap AUROC CI ===
def fusion_auroc_ci(data, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    ml = data[:, 0:1]
    fl = data[:, 1:2]
    both = np.hstack([ml, fl])
    y = data[:, 2].astype(int)

    def cv_auc(X, y_):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        preds = np.zeros(len(y_))
        for tr, va in skf.split(X, y_):
            lr = LogisticRegression(max_iter=1000).fit(X[tr], y_[tr])
            preds[va] = lr.predict_proba(X[va])[:, 1]
        return roc_auc_score(y_, preds)

    def boot_ci(X, y_):
        aucs = []
        for _ in range(n_boot):
            idx = rng.choice(len(y_), len(y_), replace=True)
            if len(np.unique(y_[idx])) < 2:
                continue
            try:
                aucs.append(cv_auc(X[idx], y_[idx]))
            except:
                pass
        return np.percentile(aucs, [2.5, 97.5]).tolist() if aucs else [None, None]

    auc_ml = cv_auc(ml, y)
    auc_fl = cv_auc(fl, y)
    auc_both = cv_auc(both, y)

    ci_ml = boot_ci(ml, y)
    ci_fl = boot_ci(fl, y)
    ci_both = boot_ci(both, y)

    # AUROC difference CI
    diffs = []
    for _ in range(n_boot):
        idx = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            a_ml = cv_auc(ml[idx], y[idx])
            a_both = cv_auc(both[idx], y[idx])
            diffs.append(a_both - a_ml)
        except:
            pass
    diff_ci = np.percentile(diffs, [2.5, 97.5]).tolist() if diffs else [None, None]

    return {
        "raw_ml": {"auroc": round(auc_ml, 4), "ci_95": [round(x, 4) for x in ci_ml]},
        "llm_fl": {"auroc": round(auc_fl, 4), "ci_95": [round(x, 4) for x in ci_fl]},
        "fusion": {"auroc": round(auc_both, 4), "ci_95": [round(x, 4) for x in ci_both]},
        "auroc_diff_fusion_minus_raw": {"mean": round(float(np.mean(diffs)), 4), "ci_95": [round(x, 4) for x in diff_ci]},
    }


# === 2. Disagreement-region analysis ===
def disagreement_region(data):
    ml = data[:, 0]
    fl = data[:, 1]
    y = data[:, 2].astype(int)

    bins = [(0.5, 0.7, "near-threshold"), (0.7, 0.85, "moderate"), (0.85, 1.01, "high-score")]
    results = {}
    for lo, hi, name in bins:
        mask = (ml >= lo) & (ml < hi)
        if mask.sum() < 5:
            continue
        sub_ml, sub_fl, sub_y = ml[mask], fl[mask], y[mask]
        n_fp = int((sub_y == 0).sum())
        n_tp = int((sub_y == 1).sum())

        # Raw ML separation
        if n_fp > 0 and n_tp > 0:
            ml_fp = sub_ml[sub_y == 0].mean()
            ml_tp = sub_ml[sub_y == 1].mean()
            fl_fp = sub_fl[sub_y == 0].mean()
            fl_tp = sub_fl[sub_y == 1].mean()
            ml_delta = float(ml_tp - ml_fp)
            fl_delta = float(fl_tp - fl_fp)

            # AUROC in region
            try:
                auc_ml = float(roc_auc_score(sub_y, sub_ml))
            except:
                auc_ml = None
            try:
                auc_fl = float(roc_auc_score(sub_y, sub_fl))
            except:
                auc_fl = None

            results[name] = {
                "n": int(mask.sum()),
                "n_fp": n_fp,
                "n_tp": n_tp,
                "ml_score_range": f"[{lo:.2f}, {hi:.2f})",
                "raw_ml": {"delta": round(ml_delta, 4), "auroc": round(auc_ml, 4) if auc_ml else None},
                "llm_fl": {"delta": round(fl_delta, 4), "auroc": round(auc_fl, 4) if auc_fl else None},
                "pearson_ml_fl": round(float(np.corrcoef(sub_ml, sub_fl)[0, 1]), 4) if len(sub_ml) > 2 else None,
            }
    return results


# === 3. Prevalence-aware evaluation ===
def prevalence_aware(data):
    ml = data[:, 0]
    fl = data[:, 1]
    y = data[:, 2].astype(int)

    results = {}
    # Simulate different prevalence by reweighting
    for prev_name, fp_weight, tp_weight in [
        ("balanced_50_50", 1.0, 1.0),
        ("realistic_10pct_tp", 9.0, 1.0),  # 90% FP, 10% TP among flagged
        ("realistic_30pct_tp", 7.0/3, 1.0),
    ]:
        weights = np.where(y == 0, fp_weight, tp_weight)
        weights = weights / weights.sum() * len(weights)

        # Weighted AUROC approximation via bootstrap resampling
        rng = np.random.RandomState(42)
        aucs_ml, aucs_fl = [], []
        for _ in range(1000):
            idx = rng.choice(len(y), len(y), replace=True, p=weights / weights.sum())
            if len(np.unique(y[idx])) < 2:
                continue
            try:
                aucs_ml.append(roc_auc_score(y[idx], ml[idx]))
                aucs_fl.append(roc_auc_score(y[idx], fl[idx]))
            except:
                pass
        results[prev_name] = {
            "raw_ml_auroc": round(float(np.mean(aucs_ml)), 4) if aucs_ml else None,
            "llm_fl_auroc": round(float(np.mean(aucs_fl)), 4) if aucs_fl else None,
            "raw_ml_auroc_ci": [round(float(np.percentile(aucs_ml, 2.5)), 4), round(float(np.percentile(aucs_ml, 97.5)), 4)] if aucs_ml else None,
        }
    return results


# === 4. Mixed-effects style paired analysis ===
def paired_analysis(controlled):
    """Within-item analysis of score effect from controlled ablation."""
    if not controlled:
        return {"error": "no controlled data"}

    shifts = []
    for item in controlled:
        shifts.append(item["fl_with"] - item["fl_without"])
    shifts = np.array(shifts)

    mean_shift = float(shifts.mean())
    se = float(shifts.std(ddof=1) / np.sqrt(len(shifts)))
    t_stat = mean_shift / se if se > 0 else 0
    p_val = float(2 * (1 - stats.t.cdf(abs(t_stat), df=len(shifts) - 1)))

    # By label
    ys = np.array([c["y"] for c in controlled])
    fp_shifts = shifts[ys == 0]
    tp_shifts = shifts[ys == 1]

    return {
        "n_paired": len(shifts),
        "mean_shift": round(mean_shift, 4),
        "se": round(se, 4),
        "t_stat": round(t_stat, 3),
        "p_value": p_val,
        "ci_95": [round(mean_shift - 1.96 * se, 4), round(mean_shift + 1.96 * se, 4)],
        "fp_mean_shift": round(float(fp_shifts.mean()), 4) if len(fp_shifts) > 0 else None,
        "tp_mean_shift": round(float(tp_shifts.mean()), 4) if len(tp_shifts) > 0 else None,
        "label_interaction": {
            "fp_n": int(len(fp_shifts)),
            "tp_n": int(len(tp_shifts)),
            "diff": round(float(tp_shifts.mean() - fp_shifts.mean()), 4) if len(fp_shifts) > 0 and len(tp_shifts) > 0 else None,
        },
    }


# === 5. Partial correlation ===
def partial_correlation(data):
    ml = data[:, 0]
    fl = data[:, 1]
    y = data[:, 2]

    # Partial correlation: ML → FL controlling for label
    from numpy.linalg import lstsq
    # Residualize ML on y
    X_y = np.column_stack([np.ones(len(y)), y])
    coef_ml, _, _, _ = lstsq(X_y, ml, rcond=None)
    ml_resid = ml - X_y @ coef_ml
    coef_fl, _, _, _ = lstsq(X_y, fl, rcond=None)
    fl_resid = fl - X_y @ coef_fl

    r_partial = float(np.corrcoef(ml_resid, fl_resid)[0, 1])

    # Also: variance in LLM fl explained by ML score vs label
    from sklearn.linear_model import LinearRegression
    r2_ml_only = LinearRegression().fit(ml.reshape(-1, 1), fl).score(ml.reshape(-1, 1), fl)
    r2_label_only = LinearRegression().fit(y.reshape(-1, 1), fl).score(y.reshape(-1, 1), fl)
    r2_both = LinearRegression().fit(np.column_stack([ml, y]), fl).score(np.column_stack([ml, y]), fl)

    return {
        "partial_r_ml_fl_controlling_label": round(r_partial, 4),
        "r2_ml_only": round(float(r2_ml_only), 4),
        "r2_label_only": round(float(r2_label_only), 4),
        "r2_both": round(float(r2_both), 4),
        "incremental_r2_label_over_ml": round(float(r2_both - r2_ml_only), 4),
        "incremental_r2_ml_over_label": round(float(r2_both - r2_label_only), 4),
    }


# === 6. Minimum detectable effect ===
def min_detectable_effect(n=200, alpha=0.05, power=0.80):
    """Cohen's d for minimum detectable effect at given n, alpha, power."""
    from scipy.stats import norm
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    d_min = (z_alpha + z_beta) * np.sqrt(2 / (n / 2))
    return {
        "n": n,
        "alpha": alpha,
        "power": power,
        "min_detectable_d": round(float(d_min), 3),
        "note": "fusion AUROC diff of 0.002 is far below detectable threshold; absence of significance does not imply equivalence",
    }


def main():
    print("Loading data...")
    data = load_scoreaware()
    controlled = load_controlled()
    print(f"Scoreaware: n={len(data)}, Controlled pairs: n={len(controlled)}")

    print("1. Fusion AUROC CI (bootstrap, may take ~1min)...")
    fusion = fusion_auroc_ci(data)
    print(f"   Raw ML: {fusion['raw_ml']['auroc']} {fusion['raw_ml']['ci_95']}")
    print(f"   Fusion: {fusion['fusion']['auroc']} {fusion['fusion']['ci_95']}")

    print("2. Disagreement-region analysis...")
    disagreement = disagreement_region(data)
    for k, v in disagreement.items():
        print(f"   {k}: n={v['n']}, ml_delta={v['raw_ml']['delta']}, fl_delta={v['llm_fl']['delta']}")

    print("3. Prevalence-aware evaluation...")
    prevalence = prevalence_aware(data)
    for k, v in prevalence.items():
        print(f"   {k}: ml={v['raw_ml_auroc']}, fl={v['llm_fl_auroc']}")

    print("4. Paired analysis (controlled ablation)...")
    paired = paired_analysis(controlled)
    print(f"   n={paired['n_paired']}, shift={paired['mean_shift']}, p={paired['p_value']}")

    print("5. Partial correlation...")
    partial = partial_correlation(data)
    print(f"   r_partial(ML→FL|label)={partial['partial_r_ml_fl_controlling_label']}")
    print(f"   R²: ML={partial['r2_ml_only']}, label={partial['r2_label_only']}, both={partial['r2_both']}")

    print("6. Minimum detectable effect...")
    mde = min_detectable_effect()
    print(f"   min d={mde['min_detectable_d']}")

    results = {
        "fusion_auroc_ci": fusion,
        "disagreement_region": disagreement,
        "prevalence_aware": prevalence,
        "paired_controlled_ablation": paired,
        "partial_correlation": partial,
        "minimum_detectable_effect": mde,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
