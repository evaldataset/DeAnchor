"""Formal equivalence test (TOST) and decision-curve analysis.

Addresses Reviewer E (non-inferiority/equivalence) and Reviewer B (cost-sensitive
evaluation at realistic prevalence).

Outputs: experiments/results/equivalence_decisioncurve.json
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

BASE = Path(__file__).resolve().parent.parent
SCOREAWARE = BASE / "experiments" / "results" / "scoreaware_ieee_cis_200.jsonl"
OUTPUT = BASE / "experiments" / "results" / "equivalence_decisioncurve.json"


def load():
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


def cv_auc(X, y, seed=42):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    preds = np.zeros(len(y))
    for tr, va in skf.split(X, y):
        lr = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
        preds[va] = lr.predict_proba(X[va])[:, 1]
    return float(roc_auc_score(y, preds))


def tost_auroc(data, margin=0.03, n_boot=2000, seed=42):
    """TOST: two one-sided tests for AUROC equivalence.

    H0_lower: AUROC_fusion - AUROC_raw < -margin H0_upper: AUROC_fusion - AUROC_raw > +margin Reject both -> conclude
    equivalence within ±margin.

    We use bootstrap to estimate the null distributions: if the 90% CI of the observed difference lies entirely within
    (-margin, +margin), both one-sided nulls are rejected at alpha=0.05.
    """
    rng = np.random.RandomState(seed)
    ml = data[:, 0:1]
    fl = data[:, 1:2]
    both = np.hstack([ml, fl])
    y = data[:, 2].astype(int)

    obs_diff = cv_auc(both, y) - cv_auc(ml, y)

    boot_diffs = []
    for _ in range(n_boot):
        idx = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            d = cv_auc(both[idx], y[idx]) - cv_auc(ml[idx], y[idx])
            boot_diffs.append(d)
        except Exception:
            pass
    boot_diffs = np.array(boot_diffs)

    ci_90 = np.percentile(boot_diffs, [5, 95]).tolist()
    ci_95 = np.percentile(boot_diffs, [2.5, 97.5]).tolist()

    # TOST: equivalence if 90% CI entirely within (-margin, +margin)
    equivalence_established = (ci_90[0] > -margin) and (ci_90[1] < margin)

    # Also report proportion of bootstrap samples with |diff| >= margin
    prop_nonequiv = float(np.mean(np.abs(boot_diffs) >= margin))

    return {
        "margin": margin,
        "observed_diff": round(float(obs_diff), 5),
        "bootstrap_mean_diff": round(float(boot_diffs.mean()), 5),
        "ci_90": [round(x, 5) for x in ci_90],
        "ci_95": [round(x, 5) for x in ci_95],
        "equivalence_within_margin": bool(equivalence_established),
        "prop_bootstrap_nonequiv": round(prop_nonequiv, 4),
        "note": (
            "Equivalence established at margin={:.3f} if 90% CI is entirely within "
            "(-margin, +margin). Current CI: [{:.4f}, {:.4f}].".format(
                margin, ci_90[0], ci_90[1]
            )
        ),
    }


def decision_curve(data, prevalences=(0.035, 0.10, 0.30, 0.50)):
    """Decision-curve analysis with all three scores Platt-calibrated on the same scale.

    Raw scores live on different scales (raw_ml is an XGBoost posterior, llm_fl is an LLM fraud_likelihood, fusion is
    already a logistic probability). Comparing absolute thresholds across scales is unfair. We therefore map each score
    to a calibrated probability via 5-fold CV logistic regression on the true label, then apply a common probability
    threshold pt to all three.

    Net benefit = (TP/n) - (FP/n) * (pt / (1 - pt)) (Vickers & Elkin 2006). Prevalence reweights the sample to simulate
    realistic post-flag distributions.
    """
    ml = data[:, 0]
    fl = data[:, 1]
    y = data[:, 2].astype(int)

    def cv_calibrate(X):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        preds = np.zeros(len(y))
        for tr, va in skf.split(X, y):
            lr = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
            preds[va] = lr.predict_proba(X[va])[:, 1]
        return preds

    ml_cal = cv_calibrate(ml.reshape(-1, 1))
    fl_cal = cv_calibrate(fl.reshape(-1, 1))
    fusion_cal = cv_calibrate(np.column_stack([ml, fl]))

    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    result = {}

    for prev in prevalences:
        fp_weight = (1 - prev) / 0.5
        tp_weight = prev / 0.5
        weights = np.where(y == 0, fp_weight, tp_weight)
        n = float(np.sum(weights))

        by_pt = {}
        for pt in thresholds:
            row = {}
            for name, scores in [
                ("raw_ml", ml_cal),
                ("llm_fl", fl_cal),
                ("fusion", fusion_cal),
            ]:
                pred_pos = scores >= pt
                tp = float(np.sum(weights[(y == 1) & pred_pos]))
                fp = float(np.sum(weights[(y == 0) & pred_pos]))
                nb = (tp / n) - (fp / n) * (pt / (1 - pt))
                row[name] = round(nb, 5)
            tp_all = float(np.sum(weights[y == 1]))
            fp_all = float(np.sum(weights[y == 0]))
            row["treat_all"] = round(
                (tp_all / n) - (fp_all / n) * (pt / (1 - pt)), 5
            )
            row["treat_none"] = 0.0
            by_pt[f"pt_{pt:.2f}"] = row
        result[f"prevalence_{prev:.3f}"] = by_pt

    return result


def max_net_benefit_advantage(dc):
    """Across all (prevalence, pt) combinations, compute max LLM/fusion advantage over raw ML."""
    max_llm_advantage = -1.0
    max_fusion_advantage = -1.0
    best_llm = None
    best_fusion = None
    for prev, pts in dc.items():
        for pt, vals in pts.items():
            d_llm = vals["llm_fl"] - vals["raw_ml"]
            d_fus = vals["fusion"] - vals["raw_ml"]
            if d_llm > max_llm_advantage:
                max_llm_advantage = d_llm
                best_llm = (prev, pt, vals)
            if d_fus > max_fusion_advantage:
                max_fusion_advantage = d_fus
                best_fusion = (prev, pt, vals)
    return {
        "max_llm_over_raw_ml_advantage": round(max_llm_advantage, 5),
        "max_fusion_over_raw_ml_advantage": round(max_fusion_advantage, 5),
        "best_llm_context": best_llm,
        "best_fusion_context": best_fusion,
    }


def main():
    print("Loading data...")
    data = load()
    print(f"n={len(data)}")

    print("TOST equivalence (±0.03)...")
    tost = tost_auroc(data, margin=0.03)
    print(f"  observed diff={tost['observed_diff']}, CI90={tost['ci_90']}")
    print(f"  equivalence within ±0.03: {tost['equivalence_within_margin']}")

    print("TOST equivalence (±0.05)...")
    tost_05 = tost_auroc(data, margin=0.05)
    print(f"  equivalence within ±0.05: {tost_05['equivalence_within_margin']}")

    print("Decision-curve analysis across prevalences...")
    dc = decision_curve(data)
    advantage = max_net_benefit_advantage(dc)
    print(f"  max LLM advantage over raw ML: {advantage['max_llm_over_raw_ml_advantage']}")
    print(f"  max fusion advantage over raw ML: {advantage['max_fusion_over_raw_ml_advantage']}")

    out = {
        "tost_equivalence_margin_003": tost,
        "tost_equivalence_margin_005": tost_05,
        "decision_curve": dc,
        "max_net_benefit_advantage": advantage,
        "interpretation": {
            "tost_003": "At margin=0.03, equivalence established" if tost["equivalence_within_margin"] else "At margin=0.03, equivalence NOT established (CI exceeds margin)",
            "tost_005": "At margin=0.05, equivalence established" if tost_05["equivalence_within_margin"] else "At margin=0.05, equivalence NOT established",
            "decision_curve": "No (prevalence, threshold) combination yields net benefit for LLM or fusion over raw ML larger than 0.01" if advantage["max_fusion_over_raw_ml_advantage"] < 0.01 else "Some operating points favor fusion; see detail",
        },
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
