"""Four API-based experiments using GPT-4o-mini and GPT-4o.

1. Medical cross-domain (GPT-4o-mini, n=30)
2. Paired staged vs standard (GPT-4o-mini, n=100)
3. Rich features (GPT-4o-mini, n=100)
4. GPT-4o controlled ablation (GPT-4o, n=50)

Outputs:
    experiments/results/medical_gpt4omini_audit.json
    experiments/results/medical_gpt4omini_raw.jsonl
    experiments/results/paired_staged_gpt4omini_audit.json
    experiments/results/rich_features_gpt4omini_audit.json
    experiments/results/rich_features_gpt4omini_raw.jsonl
    experiments/results/gpt4o_ablation_audit.json
    experiments/results/api_experiments_summary.json
"""

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm_explainer.llm_inference import _parse_json_response

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "experiments" / "results"
client = OpenAI()


def call_llm(prompt, system="", model="gpt-4o-mini", temperature=0.0):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        temperature=temperature, max_tokens=512,
    )
    return resp.choices[0].message.content.strip()


def cv_auc(X, y, seed=42):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    preds = np.zeros(len(y))
    for tr, va in skf.split(X, y):
        lr = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
        preds[va] = lr.predict_proba(X[va])[:, 1]
    return float(roc_auc_score(y, preds))


def lr_test(X_full, X_red, y):
    lr_f = LogisticRegression(max_iter=1000, C=1e6).fit(X_full, y)
    lr_r = LogisticRegression(max_iter=1000, C=1e6).fit(X_red, y)
    pf = np.clip(lr_f.predict_proba(X_full)[:, 1], 1e-12, 1-1e-12)
    pr = np.clip(lr_r.predict_proba(X_red)[:, 1], 1e-12, 1-1e-12)
    ll_f = float(np.sum(y*np.log(pf) + (1-y)*np.log(1-pf)))
    ll_r = float(np.sum(y*np.log(pr) + (1-y)*np.log(1-pr)))
    chi2 = 2*(ll_f - ll_r)
    p = float(1 - stats.chi2.cdf(chi2, 1))
    return {"chi2": round(chi2, 4), "p": p}


# ==========================================================
# 1. Medical cross-domain
# ==========================================================
def run_medical():
    print("\n" + "="*60)
    print("1. Medical Cross-Domain (GPT-4o-mini, n=30)")
    print("="*60)

    SYS = "You are a medical risk assessor. Evaluate cardiovascular risk based on the patient profile."
    TPL_SCORE = """Assess cardiovascular risk for this patient.

## Patient Profile
{profile}

## ML Risk Model Score
Risk probability: {score:.4f}

Respond ONLY with a JSON object:
{{"risk_likelihood": 0.0-1.0, "recommendation": "HIGH_RISK or LOW_RISK"}}"""

    TPL_BLIND = """Assess cardiovascular risk for this patient.

## Patient Profile
{profile}

Respond ONLY with a JSON object:
{{"risk_likelihood": 0.0-1.0, "recommendation": "HIGH_RISK or LOW_RISK"}}"""

    rng = random.Random(42)
    patients = []
    for i in range(30):
        is_hr = i < 15
        if is_hr:
            age = rng.randint(60, 82); bp_s = rng.randint(150, 180); bp_d = rng.randint(95, 110)
            chol = rng.randint(240, 290); bmi = round(rng.uniform(30, 38), 1)
            score = round(rng.uniform(0.75, 0.92), 4)
        else:
            age = rng.randint(30, 55); bp_s = rng.randint(110, 135); bp_d = rng.randint(70, 85)
            chol = rng.randint(170, 210); bmi = round(rng.uniform(21, 27), 1)
            score = round(rng.uniform(0.75, 0.92), 4)
        profile = (f"Age: {age} years\nBlood Pressure: {bp_s}/{bp_d} mmHg\n"
                   f"Total Cholesterol: {chol} mg/dL\nBMI: {bmi}\n"
                   f"Smoking: {'Yes' if is_hr else 'No'}\n"
                   f"Family History of CVD: {'Yes' if is_hr else 'No'}")
        patients.append({"profile": profile, "score": score, "is_high_risk": is_hr})

    raw = []
    for idx, pat in enumerate(patients):
        for cond, tpl in [("with_score", TPL_SCORE), ("without_score", TPL_BLIND)]:
            prompt = tpl.format(profile=pat["profile"], score=pat["score"]) if cond == "with_score" else tpl.format(profile=pat["profile"])
            resp = call_llm(prompt, SYS)
            parsed = _parse_json_response(resp)
            fl = parsed.get("risk_likelihood") if isinstance(parsed, dict) else None
            raw.append({"patient_idx": idx, "condition": cond, "is_high_risk": pat["is_high_risk"],
                        "ml_score": pat["score"], "risk_likelihood": fl if isinstance(fl, (int,float)) else None})
        if (idx+1) % 10 == 0:
            print(f"  [{idx+1}/30]")

    with open(RESULTS / "medical_gpt4omini_raw.jsonl", "w") as f:
        for r in raw: f.write(json.dumps(r) + "\n")

    aware, blind, ml_arr, y_arr = [], [], [], []
    for i in range(30):
        w = next((r for r in raw if r["patient_idx"]==i and r["condition"]=="with_score"), None)
        wo = next((r for r in raw if r["patient_idx"]==i and r["condition"]=="without_score"), None)
        if w and wo and isinstance(w["risk_likelihood"], (int,float)) and isinstance(wo["risk_likelihood"], (int,float)):
            aware.append(w["risk_likelihood"]); blind.append(wo["risk_likelihood"])
            ml_arr.append(w["ml_score"]); y_arr.append(int(w["is_high_risk"]))

    n = len(y_arr)
    print(f"  Valid pairs: {n}/30")
    if n < 10:
        return {"error": "too few valid", "n_valid": n}

    ml = np.array(ml_arr); fl_a = np.array(aware); fl_b = np.array(blind); y = np.array(y_arr)
    shift = fl_a - fl_b
    alpha = round(float(shift.mean() / ml.mean()), 4) if ml.mean() > 0 else None
    fp = y == 0; tp = y == 1

    try: auc_blind = cv_auc(fl_b.reshape(-1,1), y)
    except: auc_blind = None
    try: auc_aware = cv_auc(fl_a.reshape(-1,1), y)
    except: auc_aware = None

    audit = {
        "domain": "medical_cardiovascular", "model": "GPT-4o-mini", "n": n,
        "alpha": alpha,
        "paired_shift_mean": round(float(shift.mean()), 4),
        "paired_shift_p": float(stats.ttest_1samp(shift, 0).pvalue),
        "blind_delta": round(float(fl_b[tp].mean() - fl_b[fp].mean()), 4),
        "aware_delta": round(float(fl_a[tp].mean() - fl_a[fp].mean()), 4),
        "auroc_blind": round(auc_blind, 4) if auc_blind else None,
        "auroc_aware": round(auc_aware, 4) if auc_aware else None,
    }
    print(f"  α={alpha}, blind AUROC={auc_blind}, aware AUROC={auc_aware}")
    with open(RESULTS / "medical_gpt4omini_audit.json", "w") as f:
        json.dump(audit, f, indent=2)
    return audit


# ==========================================================
# 2. Paired staged (GPT-4o-mini)
# ==========================================================
def run_paired_staged():
    print("\n" + "="*60)
    print("2. Paired Staged vs Standard (GPT-4o-mini, n=100)")
    print("="*60)

    SYS = "You are a financial fraud investigator. Assess the transaction below for fraud risk."
    STD = """Assess this transaction for fraud risk.

Transaction Details:
{text}

ML Model Score: Fraud probability: {score:.4f}

Evaluate BOTH sides. Respond in JSON: {{"fraud_likelihood": 0.0-1.0, "recommendation": "RELEASE|ESCALATE"}}"""

    BLIND = """Assess this transaction for fraud risk based ONLY on the transaction details.

Transaction Details:
{text}

Evaluate BOTH sides. Respond in JSON: {{"fraud_likelihood": 0.0-1.0, "recommendation": "RELEASE|ESCALATE"}}"""

    INFORMED = """You previously assessed this transaction without knowing the ML score.
Your initial assessment was: fraud_likelihood = {blind_fl}

Now consider the ML fraud score:
ML Model Score: Fraud probability: {score:.4f}

Transaction Details:
{text}

Revise your assessment. Respond in JSON: {{"fraud_likelihood": 0.0-1.0, "recommendation": "RELEASE|ESCALATE"}}"""

    with open(RESULTS / "controlled_ablation_ieee_with_score.jsonl") as f:
        items = [json.loads(l) for l in f][:100]

    results = []
    for i, item in enumerate(items):
        orig = item.get("original", {})
        text = orig.get("text", ""); score = orig.get("fraud_score", 0.5); y = orig.get("is_fraud", 0)

        resp_std = call_llm(STD.format(text=text, score=score), SYS)
        fl_std = _parse_json_response(resp_std).get("fraud_likelihood") if isinstance(_parse_json_response(resp_std), dict) else None

        resp_blind = call_llm(BLIND.format(text=text), SYS)
        parsed_blind = _parse_json_response(resp_blind)
        fl_blind = parsed_blind.get("fraud_likelihood") if isinstance(parsed_blind, dict) else None
        fl_blind_val = fl_blind if isinstance(fl_blind, (int,float)) else 0.5

        resp_inf = call_llm(INFORMED.format(text=text, score=score, blind_fl=fl_blind_val), SYS)
        fl_staged = _parse_json_response(resp_inf).get("fraud_likelihood") if isinstance(_parse_json_response(resp_inf), dict) else None

        results.append({"y": y, "ml": score,
                        "fl_standard": fl_std if isinstance(fl_std, (int,float)) else None,
                        "fl_staged": fl_staged if isinstance(fl_staged, (int,float)) else None})
        if (i+1) % 25 == 0:
            print(f"  [{i+1}/100]")

    valid = [r for r in results if r["fl_standard"] is not None and r["fl_staged"] is not None]
    print(f"  Valid: {len(valid)}")

    y = np.array([r["y"] for r in valid]); fl_s = np.array([r["fl_standard"] for r in valid])
    fl_t = np.array([r["fl_staged"] for r in valid])
    fp = y==0; tp = y==1
    d_std = float(fl_s[tp].mean() - fl_s[fp].mean())
    d_stg = float(fl_t[tp].mean() - fl_t[fp].mean())
    try: auc_std = cv_auc(fl_s.reshape(-1,1), y); auc_stg = cv_auc(fl_t.reshape(-1,1), y)
    except: auc_std = auc_stg = None

    audit = {"n": len(valid), "model": "GPT-4o-mini",
             "standard_delta": round(d_std, 4), "staged_delta": round(d_stg, 4),
             "improvement_pct": round(100*(d_stg - d_std)/abs(d_std), 1) if d_std != 0 else None,
             "auroc_standard": round(auc_std, 4) if auc_std else None,
             "auroc_staged": round(auc_stg, 4) if auc_stg else None,
             "paired_diff_p": float(stats.ttest_1samp(fl_t - fl_s, 0).pvalue)}
    print(f"  Std Δ={d_std:.4f}, Staged Δ={d_stg:.4f}, improvement={audit['improvement_pct']}%")
    with open(RESULTS / "paired_staged_gpt4omini_audit.json", "w") as f:
        json.dump(audit, f, indent=2)
    return audit


# ==========================================================
# 3. Rich features (GPT-4o-mini)
# ==========================================================
def run_rich_features():
    print("\n" + "="*60)
    print("3. Rich Features (GPT-4o-mini, n=100)")
    print("="*60)

    SYS = "You are a senior financial fraud investigator conducting independent reviews of ML-flagged transactions."
    TPL_SCORE = """Assess this transaction for fraud risk.

Transaction Details:
{text}

ML Fraud Score: {score:.4f} (threshold: 0.50)

Provide balanced assessment as JSON: {{"fraud_likelihood": 0.0-1.0, "recommendation": "RELEASE|ESCALATE"}}"""

    TPL_BLIND = """Assess this transaction for fraud risk.

Transaction Details:
{text}

Provide balanced assessment as JSON: {{"fraud_likelihood": 0.0-1.0, "recommendation": "RELEASE|ESCALATE"}}"""

    blind_data = []
    with open(RESULTS / "scoreblind_paysim.jsonl") as f:
        for ln in f:
            r = json.loads(ln)
            orig = r.get("original", {}); ba = r.get("blind_assessment", {})
            fl = ba.get("fraud_likelihood") if isinstance(ba, dict) else None
            if isinstance(fl, (int,float)):
                blind_data.append({"tid": orig.get("transaction_id"), "ml": float(orig.get("fraud_score",0)),
                                   "y": int(orig.get("is_fraud",0)), "text": orig.get("text","")})

    rng = random.Random(42)
    merchants = ["Amazon Inc.", "Walmart Stores", "Shell Gas Station", "Uber Technologies",
                 "Netflix Subscription", "DoorDash Delivery", "Apple Store", "Target Corp"]
    devices = ["iPhone 15 Pro (iOS 17.4)", "Samsung Galaxy S24 (Android 14)", "Chrome on Windows 11"]
    histories = ["Account active 3 years, 847 txns, 0 disputes", "Account active 6 months, 23 txns, 1 dispute",
                 "New account (2 weeks), 4 txns", "Account active 1 year, 156 txns, 2 chargebacks"]

    results = []
    for i, item in enumerate(blind_data[:100]):
        rich = (f"{item['text']}\nMerchant: {rng.choice(merchants)}\nDevice: {rng.choice(devices)}\n"
                f"Account history: {rng.choice(histories)}\nLocation: consistent with prior\n"
                f"Velocity: 2 txns in last 24h (normal)")

        resp_a = call_llm(TPL_SCORE.format(text=rich, score=item["ml"]), SYS)
        fl_a = _parse_json_response(resp_a).get("fraud_likelihood") if isinstance(_parse_json_response(resp_a), dict) else None
        resp_b = call_llm(TPL_BLIND.format(text=rich), SYS)
        fl_b = _parse_json_response(resp_b).get("fraud_likelihood") if isinstance(_parse_json_response(resp_b), dict) else None

        results.append({"tid": item["tid"], "ml": item["ml"], "y": item["y"],
                        "fl_aware": fl_a if isinstance(fl_a, (int,float)) else None,
                        "fl_blind": fl_b if isinstance(fl_b, (int,float)) else None})
        if (i+1) % 25 == 0:
            print(f"  [{i+1}/100]")

    with open(RESULTS / "rich_features_gpt4omini_raw.jsonl", "w") as f:
        for r in results: f.write(json.dumps(r) + "\n")

    valid = [r for r in results if r["fl_aware"] is not None and r["fl_blind"] is not None]
    ml = np.array([r["ml"] for r in valid]); fl_a = np.array([r["fl_aware"] for r in valid])
    fl_b = np.array([r["fl_blind"] for r in valid]); y = np.array([r["y"] for r in valid])
    n = len(y)
    print(f"  Valid: {n}")

    auc_ml = cv_auc(ml.reshape(-1,1), y)
    auc_a = cv_auc(fl_a.reshape(-1,1), y); auc_b = cv_auc(fl_b.reshape(-1,1), y)
    auc_fus = cv_auc(np.column_stack([ml, fl_a]), y)
    lrt = lr_test(np.column_stack([ml, fl_a]), ml.reshape(-1,1), y)
    shift = fl_a - fl_b
    alpha = round(float(shift.mean() / ml.mean()), 4) if ml.mean() > 0 else None

    audit = {"model": "GPT-4o-mini", "n": n,
             "auroc_raw_ml": round(auc_ml, 4), "auroc_scoreaware": round(auc_a, 4),
             "auroc_scoreblind": round(auc_b, 4), "auroc_fusion": round(auc_fus, 4),
             "lr_test": lrt, "alpha_rich": alpha,
             "suppression": round(float(auc_b - auc_a), 4)}
    print(f"  blind={auc_b:.3f}, aware={auc_a:.3f}, suppression={auc_b-auc_a:.3f}, α={alpha}")
    with open(RESULTS / "rich_features_gpt4omini_audit.json", "w") as f:
        json.dump(audit, f, indent=2)
    return audit


# ==========================================================
# 4. GPT-4o controlled ablation
# ==========================================================
def run_gpt4o_ablation():
    print("\n" + "="*60)
    print("4. GPT-4o Controlled Ablation (n=50)")
    print("="*60)

    SYS = "You are a financial fraud investigator. Assess the transaction below for fraud risk."
    TPL = """Assess this transaction for fraud risk.

Transaction Details:
{text}
{score_line}
Evaluate BOTH sides. Respond in JSON: {{"fraud_likelihood": 0.0-1.0, "recommendation": "RELEASE|ESCALATE"}}"""

    with open(RESULTS / "controlled_ablation_ieee_with_score.jsonl") as f:
        items = [json.loads(l) for l in f][:50]

    results = []
    for i, item in enumerate(items):
        orig = item.get("original", {}); text = orig.get("text",""); score = orig.get("fraud_score",0.5)

        resp_w = call_llm(TPL.format(text=text, score_line=f"\nML Model Score: Fraud probability: {score:.4f}\n"), SYS, model="gpt-4o")
        fl_w = _parse_json_response(resp_w).get("fraud_likelihood") if isinstance(_parse_json_response(resp_w), dict) else None
        resp_wo = call_llm(TPL.format(text=text, score_line=""), SYS, model="gpt-4o")
        fl_wo = _parse_json_response(resp_wo).get("fraud_likelihood") if isinstance(_parse_json_response(resp_wo), dict) else None

        results.append({"y": orig.get("is_fraud",0), "ml": score,
                        "fl_with": fl_w if isinstance(fl_w, (int,float)) else None,
                        "fl_without": fl_wo if isinstance(fl_wo, (int,float)) else None})
        if (i+1) % 10 == 0:
            print(f"  [{i+1}/50]")

    valid = [r for r in results if r["fl_with"] is not None and r["fl_without"] is not None]
    print(f"  Valid: {len(valid)}")

    shifts = np.array([r["fl_with"] - r["fl_without"] for r in valid])
    ml = np.array([r["ml"] for r in valid])
    alpha = round(float(shifts.mean() / ml.mean()), 4) if ml.mean() > 0 else None
    t, p = stats.ttest_1samp(shifts, 0)

    audit = {"model": "GPT-4o", "n": len(valid),
             "mean_shift": round(float(shifts.mean()), 4),
             "shift_ci95": [round(float(np.percentile(shifts, 2.5)), 4), round(float(np.percentile(shifts, 97.5)), 4)],
             "shift_p": float(p), "alpha": alpha}
    print(f"  shift={shifts.mean():.4f}, α={alpha}, p={p:.2e}")
    with open(RESULTS / "gpt4o_ablation_audit.json", "w") as f:
        json.dump(audit, f, indent=2)
    return audit


# ==========================================================
def main():
    start = time.time()
    r1 = run_medical()
    r2 = run_paired_staged()
    r3 = run_rich_features()
    r4 = run_gpt4o_ablation()
    elapsed = time.time() - start

    summary = {"medical": r1, "paired_staged": r2, "rich_features": r3, "gpt4o_ablation": r4,
               "total_minutes": round(elapsed/60, 1)}
    with open(RESULTS / "api_experiments_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n{'='*60}\nALL DONE in {elapsed/60:.1f} min\n{'='*60}")
    print(json.dumps({k: {kk:vv for kk,vv in v.items() if kk not in ('details',)} if isinstance(v,dict) else v for k,v in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
