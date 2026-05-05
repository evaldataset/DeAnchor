"""Cross-dataset replication of main confirmatory findings on PaySim.

Addresses AC's external-validity requirement: replicates (i) fusion
incremental-value test and (ii) normative-combiner baseline on PaySim,
which shares the fraud-domain task but is drawn from a different
anonymized feature distribution (mobile-money transfers vs IEEE-CIS
card transactions).

The PaySim feature space is different from IEEE-CIS (transaction types:
PAYMENT/TRANSFER/CASH-OUT, recipient-type distinction, step-based time),
so a consistent negative result here strengthens the main finding.

Outputs: experiments/results/paysim_replication.json
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

BASE = Path(__file__).resolve().parent.parent
AWARE_FILES = [
    BASE / "experiments" / "results" / "explanations_paysim_50.jsonl",
    BASE / "experiments" / "results" / "explanations_paysim_gpt4o.jsonl",
]
BLIND_FILE = BASE / "experiments" / "results" / "scoreblind_paysim.jsonl"
OUTPUT = BASE / "experiments" / "results" / "paysim_replication.json"


def load_paired():
    aware_by_tid = {}
    for f in AWARE_FILES:
        if not f.exists():
            continue
        with open(f) as fh:
            for ln in fh:
                r = json.loads(ln)
                fp = r.get("fp_explanation") or {}
                fl = fp.get("fraud_likelihood") if isinstance(fp, dict) else None
                if not isinstance(fl, (int, float)):
                    continue
                orig = r.get("original", {})
                tid = orig.get("transaction_id")
                ml = orig.get("fraud_score")
                y = orig.get("is_fraud")
                if tid is None or ml is None or y is None:
                    continue
                # Prefer the first occurrence to avoid overwriting
                if tid not in aware_by_tid:
                    aware_by_tid[tid] = {"fl_aware": float(fl), "ml": float(ml), "y": int(y)}

    rows = []
    with open(BLIND_FILE) as f:
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
            rows.append({
                "tid": tid,
                "ml": a["ml"],
                "fl_aware": a["fl_aware"],
                "fl_blind": float(fl_blind),
                "y": a["y"],
            })
    return rows


def cv_auc(X, y, seed=42):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    preds = np.zeros(len(y))
    for tr, va in skf.split(X, y):
        lr = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
        preds[va] = lr.predict_proba(X[va])[:, 1]
    return float(roc_auc_score(y, preds)), preds


def lr_test(X_full, X_red, y):
    lr_full = LogisticRegression(max_iter=1000, C=1e6).fit(X_full, y)
    lr_red = LogisticRegression(max_iter=1000, C=1e6).fit(X_red, y)
    p_full = np.clip(lr_full.predict_proba(X_full)[:, 1], 1e-12, 1 - 1e-12)
    p_red = np.clip(lr_red.predict_proba(X_red)[:, 1], 1e-12, 1 - 1e-12)
    ll_full = float(np.sum(y * np.log(p_full) + (1 - y) * np.log(1 - p_full)))
    ll_red = float(np.sum(y * np.log(p_red) + (1 - y) * np.log(1 - p_red)))
    df = X_full.shape[1] - X_red.shape[1]
    lr_stat = 2 * (ll_full - ll_red)
    p = float(1 - stats.chi2.cdf(lr_stat, df)) if df > 0 else None
    return {"lr_stat": round(lr_stat, 4), "df": df, "p_value": p}


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


def tost(data, margin=0.03, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    ml = data[:, 0:1]; fl = data[:, 1:2]
    both = np.hstack([ml, fl])
    y = data[:, 2].astype(int)
    diffs = []
    for _ in range(n_boot):
        idx = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            diffs.append(cv_auc(both[idx], y[idx])[0] - cv_auc(ml[idx], y[idx])[0])
        except Exception:
            pass
    ci90 = np.percentile(diffs, [5, 95]).tolist()
    obs = cv_auc(both, y)[0] - cv_auc(ml, y)[0]
    return {
        "margin": margin,
        "observed_diff": round(float(obs), 5),
        "ci_90": [round(x, 5) for x in ci90],
        "equivalence_within_margin": bool(ci90[0] > -margin and ci90[1] < margin),
    }


def main():
    rows = load_paired()
    print(f"PaySim paired n={len(rows)}")
    if len(rows) < 30:
        raise SystemExit("too few paired records")

    ml = np.array([r["ml"] for r in rows]).reshape(-1, 1)
    fl_aware = np.array([r["fl_aware"] for r in rows]).reshape(-1, 1)
    fl_blind = np.array([r["fl_blind"] for r in rows]).reshape(-1, 1)
    y = np.array([r["y"] for r in rows])
    n_fp = int((y == 0).sum()); n_tp = int((y == 1).sum())
    print(f"FP={n_fp}, TP={n_tp}")

    auc_ml, _ = cv_auc(ml, y)
    auc_aware, _ = cv_auc(fl_aware, y)
    auc_blind, _ = cv_auc(fl_blind, y)
    auc_fusion, _ = cv_auc(np.hstack([ml, fl_aware]), y)
    auc_combiner, _ = cv_auc(np.hstack([ml, fl_blind]), y)

    ci_ml = bootstrap_ci(ml, y)
    ci_aware = bootstrap_ci(fl_aware, y)
    ci_blind = bootstrap_ci(fl_blind, y)
    ci_fusion = bootstrap_ci(np.hstack([ml, fl_aware]), y)
    ci_combiner = bootstrap_ci(np.hstack([ml, fl_blind]), y)

    # LR test for incremental value of LLM aware over raw ML
    lrt_aware = lr_test(np.hstack([ml, fl_aware]), ml, y)
    lrt_raw_over_aware = lr_test(np.hstack([ml, fl_aware]), fl_aware, y)

    # TOST equivalence
    data = np.hstack([ml, fl_aware, y.reshape(-1, 1)])
    tost_result = tost(data, margin=0.03)

    summary = {
        "dataset": "PaySim",
        "n": len(rows), "n_fp": n_fp, "n_tp": n_tp,
        "auroc": {
            "raw_ml": {"value": round(auc_ml, 4), "ci_95": [round(x, 4) for x in ci_ml]},
            "scoreaware_llm": {"value": round(auc_aware, 4), "ci_95": [round(x, 4) for x in ci_aware]},
            "scoreblind_llm": {"value": round(auc_blind, 4), "ci_95": [round(x, 4) for x in ci_blind]},
            "fusion_ml_plus_aware": {"value": round(auc_fusion, 4), "ci_95": [round(x, 4) for x in ci_fusion]},
            "combiner_ml_plus_blind": {"value": round(auc_combiner, 4), "ci_95": [round(x, 4) for x in ci_combiner]},
        },
        "lr_test_llm_over_raw": lrt_aware,
        "lr_test_raw_over_llm": lrt_raw_over_aware,
        "tost_equivalence_003": tost_result,
        "normative_verdict": (
            "score-blind LLM AUROC = " + str(round(auc_blind, 3)) +
            "; optimal combiner = " + str(round(auc_combiner, 3)) +
            " vs raw ML = " + str(round(auc_ml, 3))
        ),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
