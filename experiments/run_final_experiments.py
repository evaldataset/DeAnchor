"""Final three experiments for NeurIPS submission hardening.

1. Paired staged-vs-standard (same 100 transactions, Qwen-7B)
2. Medical cross-domain retry (Gemma-9B, better JSON prompt)
3. Repeated-run variance check (20 transactions × 5 runs, Qwen-7B)

Outputs:
    experiments/results/paired_staged_audit.json
    experiments/results/medical_gemma_audit.json
    experiments/results/medical_gemma_raw.jsonl
    experiments/results/repeated_run_variance.json
"""

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm_explainer.llm_inference import _parse_json_response
from llm_explainer.prompts.fp_explanation import SYSTEM_PROMPT, build_prompt

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "experiments" / "results"

QWEN_DIR = os.environ.get(
    "QWEN_MODEL_DIR",
    str(Path.home() / ".cache/modelscope/Qwen/Qwen2.5-7B-Instruct"),
)


def find_snapshot(model_path):
    snap_dir = Path(model_path) / "snapshots"
    if snap_dir.exists():
        snaps = sorted(snap_dir.iterdir())
        if snaps:
            return str(snaps[-1])
    return model_path


def load_model(path):
    print(f"  Loading from {path}")
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"
    )
    model = AutoModelForCausalLM.from_pretrained(
        path, quantization_config=quant, device_map="auto", trust_remote_code=True
    )
    model.eval()
    print(f"  VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB")
    return model, tokenizer


def generate(model, tokenizer, prompt, system="", temperature=0.0):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = f"{system}\n\n{prompt}"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    gen_kwargs = dict(
        max_new_tokens=512, pad_token_id=tokenizer.eos_token_id,
    )
    if temperature == 0.0:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def cv_auc(X, y, seed=42):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    preds = np.zeros(len(y))
    for tr, va in skf.split(X, y):
        lr = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
        preds[va] = lr.predict_proba(X[va])[:, 1]
    return float(roc_auc_score(y, preds))


# ==========================================================
# EXPERIMENT 1: Paired staged-vs-standard
# ==========================================================
def run_paired_staged(model, tokenizer):
    print("\n" + "="*60)
    print("EXPERIMENT 1: Paired Staged vs Standard (n=100, Qwen-7B)")
    print("="*60)

    STAGED_SYSTEM = "You are a financial fraud investigator. Assess the transaction below for fraud risk."
    STAGED_BLIND = """Assess this transaction for fraud risk based ONLY on the transaction details.

Transaction Details:
{text}

Evaluate BOTH sides: evidence for fraud AND evidence against fraud.
Respond in JSON: {{"fraud_likelihood": 0.0-1.0, "evidence_for_fraud": [...], "evidence_against_fraud": [...], "recommendation": "RELEASE|ESCALATE"}}"""

    STAGED_INFORMED = """You previously assessed this transaction without knowing the ML score.
Your initial assessment was: fraud_likelihood = {blind_fl}

Now consider the ML fraud score:
ML Model Score: Fraud probability: {score:.4f}

Transaction Details:
{text}

Revise your assessment considering both your initial judgment and the ML score.
Respond in JSON: {{"fraud_likelihood": 0.0-1.0, "recommendation": "RELEASE|ESCALATE"}}"""

    STANDARD_PROMPT = """Assess this transaction for fraud risk.

Transaction Details:
{text}

ML Model Score: Fraud probability: {score:.4f}

Evaluate BOTH sides: evidence for fraud AND evidence against fraud.
Respond in JSON: {{"fraud_likelihood": 0.0-1.0, "recommendation": "RELEASE|ESCALATE"}}"""

    # Load the same 100 controlled-ablation transactions
    with open(RESULTS / "controlled_ablation_ieee_with_score.jsonl") as f:
        items = [json.loads(l) for l in f][:100]

    results = []
    for i, item in enumerate(items):
        orig = item.get("original", {})
        text = orig.get("text", "")
        score = orig.get("fraud_score", 0.5)
        y = orig.get("is_fraud", 0)
        tid = orig.get("transaction_id")

        # Standard (score-first)
        prompt_std = STANDARD_PROMPT.format(text=text, score=score)
        resp_std = generate(model, tokenizer, prompt_std, STAGED_SYSTEM)
        parsed_std = _parse_json_response(resp_std)
        fl_std = parsed_std.get("fraud_likelihood") if isinstance(parsed_std, dict) else None

        # Staged: blind first
        prompt_blind = STAGED_BLIND.format(text=text)
        resp_blind = generate(model, tokenizer, prompt_blind, STAGED_SYSTEM)
        parsed_blind = _parse_json_response(resp_blind)
        fl_blind = parsed_blind.get("fraud_likelihood") if isinstance(parsed_blind, dict) else None

        # Staged: informed (reveal score + prior blind assessment)
        fl_blind_val = fl_blind if isinstance(fl_blind, (int, float)) else 0.5
        prompt_inf = STAGED_INFORMED.format(text=text, score=score, blind_fl=fl_blind_val)
        resp_inf = generate(model, tokenizer, prompt_inf, STAGED_SYSTEM)
        parsed_inf = _parse_json_response(resp_inf)
        fl_staged = parsed_inf.get("fraud_likelihood") if isinstance(parsed_inf, dict) else None

        results.append({
            "tid": tid, "y": y, "ml_score": score,
            "fl_standard": fl_std if isinstance(fl_std, (int, float)) else None,
            "fl_staged": fl_staged if isinstance(fl_staged, (int, float)) else None,
            "fl_blind": fl_blind if isinstance(fl_blind, (int, float)) else None,
        })
        if (i+1) % 20 == 0:
            print(f"  Paired staged [{i+1}/100]")

    # Analysis: paired comparison
    valid = [r for r in results if r["fl_standard"] is not None and r["fl_staged"] is not None]
    print(f"  Valid pairs: {len(valid)}")

    if len(valid) < 30:
        audit = {"error": "too few valid", "n_valid": len(valid)}
    else:
        y = np.array([r["y"] for r in valid])
        fl_std = np.array([r["fl_standard"] for r in valid])
        fl_stg = np.array([r["fl_staged"] for r in valid])
        ml = np.array([r["ml_score"] for r in valid])

        # FP/TP separation for each
        fp_mask = y == 0; tp_mask = y == 1
        delta_std = float(fl_std[tp_mask].mean() - fl_std[fp_mask].mean()) if fp_mask.sum() > 0 and tp_mask.sum() > 0 else 0
        delta_stg = float(fl_stg[tp_mask].mean() - fl_stg[fp_mask].mean()) if fp_mask.sum() > 0 and tp_mask.sum() > 0 else 0

        # Paired difference in Delta
        # Per-transaction "discrimination contribution": sign(TP)*fl - sign(FP)*fl
        # Simpler: paired t-test on fl_staged - fl_standard
        diff = fl_stg - fl_std
        t_stat, p_val = stats.ttest_1samp(diff, 0)

        # AUROC comparison
        try:
            auc_std = cv_auc(fl_std.reshape(-1,1), y)
            auc_stg = cv_auc(fl_stg.reshape(-1,1), y)
        except:
            auc_std = auc_stg = None

        audit = {
            "n": len(valid),
            "n_fp": int(fp_mask.sum()), "n_tp": int(tp_mask.sum()),
            "standard": {
                "fp_mean": round(float(fl_std[fp_mask].mean()), 4),
                "tp_mean": round(float(fl_std[tp_mask].mean()), 4),
                "delta": round(delta_std, 4),
                "auroc": round(auc_std, 4) if auc_std else None,
            },
            "staged": {
                "fp_mean": round(float(fl_stg[fp_mask].mean()), 4),
                "tp_mean": round(float(fl_stg[tp_mask].mean()), 4),
                "delta": round(delta_stg, 4),
                "auroc": round(auc_stg, 4) if auc_stg else None,
            },
            "paired_diff_mean": round(float(diff.mean()), 4),
            "paired_diff_p": float(p_val),
            "staged_improvement_pct": round(100 * (delta_stg - delta_std) / abs(delta_std), 1) if delta_std != 0 else None,
        }
        print(f"  Standard Δ={delta_std:.4f}, Staged Δ={delta_stg:.4f}, improvement={audit.get('staged_improvement_pct')}%")

    with open(RESULTS / "paired_staged_audit.json", "w") as f:
        json.dump(audit, f, indent=2)
    return audit


# ==========================================================
# EXPERIMENT 2: Medical cross-domain (Gemma-9B)
# ==========================================================
def run_medical_gemma():
    print("\n" + "="*60)
    print("EXPERIMENT 2: Medical Cross-Domain (n=60, Gemma-9B)")
    print("="*60)

    gemma_path = find_snapshot(os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--gemma-2-9b-it"
    ))
    model, tokenizer = load_model(gemma_path)

    MED_SYSTEM = "You are a medical risk assessor. Evaluate cardiovascular risk based on the patient profile."

    MED_SCORE = """Assess cardiovascular risk for this patient.

Patient Profile:
{profile}

ML Risk Model Score: {score:.4f}

You must respond with ONLY a JSON object, no other text:
{{"risk_likelihood": <float 0.0-1.0>, "recommendation": "<HIGH_RISK or LOW_RISK>"}}"""

    MED_BLIND = """Assess cardiovascular risk for this patient.

Patient Profile:
{profile}

You must respond with ONLY a JSON object, no other text:
{{"risk_likelihood": <float 0.0-1.0>, "recommendation": "<HIGH_RISK or LOW_RISK>"}}"""

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
        patients.append({"profile": profile, "score": score, "is_high_risk": is_hr, "idx": i})

    raw = []
    for pat in patients:
        for cond, tpl in [("with_score", MED_SCORE), ("without_score", MED_BLIND)]:
            if cond == "with_score":
                prompt = tpl.format(profile=pat["profile"], score=pat["score"])
            else:
                prompt = tpl.format(profile=pat["profile"])
            resp = generate(model, tokenizer, prompt, MED_SYSTEM)
            parsed = _parse_json_response(resp)
            fl = parsed.get("risk_likelihood") if isinstance(parsed, dict) else None
            raw.append({
                "patient_idx": pat["idx"], "condition": cond,
                "is_high_risk": pat["is_high_risk"], "ml_score": pat["score"],
                "risk_likelihood": fl if isinstance(fl, (int, float)) else None,
                "raw_response": resp[:300],
            })
        if (pat["idx"]+1) % 10 == 0:
            valid_so_far = sum(1 for r in raw if r["risk_likelihood"] is not None)
            print(f"  Medical [{pat['idx']+1}/30] valid_responses={valid_so_far}/{len(raw)}")

    with open(RESULTS / "medical_gemma_raw.jsonl", "w") as f:
        for r in raw:
            f.write(json.dumps(r) + "\n")

    # Build paired arrays
    aware_fl, blind_fl, ml_scores, labels = [], [], [], []
    for i in range(30):
        w = next((r for r in raw if r["patient_idx"]==i and r["condition"]=="with_score"), None)
        wo = next((r for r in raw if r["patient_idx"]==i and r["condition"]=="without_score"), None)
        if (w and wo and isinstance(w["risk_likelihood"], (int,float))
                and isinstance(wo["risk_likelihood"], (int,float))):
            aware_fl.append(w["risk_likelihood"]); blind_fl.append(wo["risk_likelihood"])
            ml_scores.append(w["ml_score"]); labels.append(int(w["is_high_risk"]))

    del model, tokenizer
    torch.cuda.empty_cache()

    n_valid = len(labels)
    print(f"  Valid pairs: {n_valid}/30")

    if n_valid < 10:
        audit = {"error": "too few valid", "n_valid": n_valid}
    else:
        ml = np.array(ml_scores); fl_a = np.array(aware_fl); fl_b = np.array(blind_fl)
        y = np.array(labels)

        shift = fl_a - fl_b
        alpha = round(float(shift.mean() / ml.mean()), 4) if ml.mean() > 0 else None

        fp = y == 0; tp = y == 1
        delta_aware = float(fl_a[tp].mean() - fl_a[fp].mean()) if fp.sum() > 0 and tp.sum() > 0 else 0
        delta_blind = float(fl_b[tp].mean() - fl_b[fp].mean()) if fp.sum() > 0 and tp.sum() > 0 else 0

        try:
            auc_blind = cv_auc(fl_b.reshape(-1,1), y)
        except:
            auc_blind = None
        try:
            auc_aware = cv_auc(fl_a.reshape(-1,1), y)
        except:
            auc_aware = None

        audit = {
            "domain": "medical_cardiovascular",
            "model": "Gemma-2-9B",
            "n": n_valid,
            "n_high_risk": int(tp.sum()), "n_low_risk": int(fp.sum()),
            "alpha": alpha,
            "paired_shift_mean": round(float(shift.mean()), 4),
            "paired_shift_p": float(stats.ttest_1samp(shift, 0).pvalue) if len(shift) > 2 else None,
            "blind_delta": round(delta_blind, 4),
            "aware_delta": round(delta_aware, 4),
            "auroc_blind": round(auc_blind, 4) if auc_blind else None,
            "auroc_aware": round(auc_aware, 4) if auc_aware else None,
            "verdict": (
                "low_anchoring" if alpha is not None and abs(alpha) < 0.15
                else ("suppression" if auc_aware is not None and auc_blind is not None and auc_aware < auc_blind - 0.02
                      else "anchoring")
            ),
        }
        print(f"  Medical: α={alpha}, blind_Δ={delta_blind:.3f}, aware_Δ={delta_aware:.3f}, "
              f"blind_AUROC={auc_blind}, verdict={audit['verdict']}")

    with open(RESULTS / "medical_gemma_audit.json", "w") as f:
        json.dump(audit, f, indent=2)
    return audit


# ==========================================================
# EXPERIMENT 3: Repeated-run variance (Qwen-7B)
# ==========================================================
def run_repeated_variance(model, tokenizer):
    print("\n" + "="*60)
    print("EXPERIMENT 3: Repeated-Run Variance (20 tx × 5 runs)")
    print("="*60)

    with open(RESULTS / "controlled_ablation_ieee_with_score.jsonl") as f:
        items = [json.loads(l) for l in f][:20]

    PROMPT = """Assess this transaction for fraud risk.

Transaction Details:
{text}

ML Model Score: Fraud probability: {score:.4f}

Evaluate BOTH sides. Respond in JSON: {{"fraud_likelihood": 0.0-1.0, "recommendation": "RELEASE|ESCALATE"}}"""

    SYS = "You are a financial fraud investigator."

    all_results = []
    for i, item in enumerate(items):
        orig = item.get("original", {})
        text = orig.get("text", "")
        score = orig.get("fraud_score", 0.5)
        tid = orig.get("transaction_id")

        fls = []
        for run in range(5):
            resp = generate(model, tokenizer, PROMPT.format(text=text, score=score), SYS, temperature=0.0)
            parsed = _parse_json_response(resp)
            fl = parsed.get("fraud_likelihood") if isinstance(parsed, dict) else None
            fls.append(fl if isinstance(fl, (int, float)) else None)

        valid_fls = [f for f in fls if f is not None]
        all_results.append({
            "tid": tid,
            "ml_score": score,
            "is_fraud": orig.get("is_fraud"),
            "runs": fls,
            "n_valid": len(valid_fls),
            "mean": round(float(np.mean(valid_fls)), 4) if valid_fls else None,
            "std": round(float(np.std(valid_fls)), 4) if len(valid_fls) > 1 else 0.0,
            "range": round(float(max(valid_fls) - min(valid_fls)), 4) if len(valid_fls) > 1 else 0.0,
        })
        if (i+1) % 10 == 0:
            print(f"  Variance [{i+1}/20]")

    # Aggregate stats
    stds = [r["std"] for r in all_results if r["std"] is not None]
    ranges = [r["range"] for r in all_results if r["range"] is not None]

    audit = {
        "n_transactions": len(all_results),
        "n_runs_per_tx": 5,
        "temperature": 0.0,
        "within_item_std_mean": round(float(np.mean(stds)), 5) if stds else None,
        "within_item_std_max": round(float(np.max(stds)), 5) if stds else None,
        "within_item_range_mean": round(float(np.mean(ranges)), 5) if ranges else None,
        "within_item_range_max": round(float(np.max(ranges)), 5) if ranges else None,
        "n_perfectly_deterministic": sum(1 for r in all_results if r["std"] == 0.0 and r["n_valid"] == 5),
        "details": all_results,
    }
    print(f"  Mean within-item SD: {audit['within_item_std_mean']}")
    print(f"  Max within-item SD: {audit['within_item_std_max']}")
    print(f"  Perfectly deterministic: {audit['n_perfectly_deterministic']}/20")

    with open(RESULTS / "repeated_run_variance.json", "w") as f:
        json.dump(audit, f, indent=2)
    return audit


# ==========================================================
# MAIN
# ==========================================================
def main():
    start = time.time()

    # Load Qwen for experiments 1 and 3
    print("Loading Qwen-7B...")
    model_q, tok_q = load_model(QWEN_DIR)

    r1 = run_paired_staged(model_q, tok_q)
    r3 = run_repeated_variance(model_q, tok_q)

    # Free Qwen
    del model_q, tok_q
    torch.cuda.empty_cache()

    # Experiment 2: medical with Gemma
    r2 = run_medical_gemma()

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"ALL DONE in {elapsed/60:.1f} min")
    print(f"  Paired staged: improvement={r1.get('staged_improvement_pct','error')}%")
    print(f"  Medical: verdict={r2.get('verdict','error')}, α={r2.get('alpha')}")
    print(f"  Variance: mean_SD={r3.get('within_item_std_mean','error')}")

    summary = {"paired_staged": r1, "medical_gemma": r2, "repeated_variance": r3,
               "total_minutes": round(elapsed/60, 1)}
    with open(RESULTS / "final_experiments_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
