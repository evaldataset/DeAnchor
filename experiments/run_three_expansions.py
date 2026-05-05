"""Run all three NeurIPS expansion experiments with local Qwen2.5-7B.

1. Medical cross-domain replication (n=30 × 2 conditions)
2. PaySim n=200 expansion (generate 125 missing score-aware outputs)
3. Rich-features audit (enrich PaySim with synthetic merchant/device info)

All use local Qwen2.5-7B-Instruct to avoid API dependency.
Total runtime: ~2-3 hours on single GPU.

Outputs:
    experiments/results/medical_local_audit.json
    experiments/results/medical_local_raw.jsonl
    experiments/results/paysim_expanded_scoreaware.jsonl
    experiments/results/paysim_expanded_audit.json
    experiments/results/rich_features_raw.jsonl
    experiments/results/rich_features_audit.json
"""

import argparse
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

# Lazy imports for local Qwen inference. The audit toolkit and many downstream
# analyses do NOT require torch/transformers/bitsandbytes to be importable.
torch = None
AutoModelForCausalLM = None
AutoTokenizer = None
BitsAndBytesConfig = None


def _ensure_local_deps():
    """Import torch/transformers on demand; raise with a helpful message."""
    global torch, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    if torch is None:
        try:
            import torch as _torch
            from transformers import AutoModelForCausalLM as _AMC
            from transformers import AutoTokenizer as _AT
            from transformers import BitsAndBytesConfig as _BNB
            torch = _torch
            AutoModelForCausalLM = _AMC
            AutoTokenizer = _AT
            BitsAndBytesConfig = _BNB
        except ImportError as e:
            raise ImportError(
                "Local Qwen inference requires torch + transformers + bitsandbytes. "
                "Install with `pip install torch transformers bitsandbytes`, or "
                "re-run with --skip-local to skip the local-Qwen branch."
            ) from e


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm_explainer.llm_inference import _parse_json_response  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "experiments" / "results"
MODEL_DIR = os.environ.get(
    "QWEN_MODEL_DIR",
    str(Path.home() / ".cache/modelscope/Qwen/Qwen2.5-7B-Instruct"),
)


def wait_for_gpu(min_free_gb=20, poll_interval=30, max_wait_s=1800):
    """Block until at least min_free_gb is available on GPU.

    Bounded by max_wait_s (default 30 minutes) to avoid hanging indefinitely; raises RuntimeError after the timeout so
    callers can fall back to API-only.
    """
    import subprocess
    waited = 0
    while waited < max_wait_s:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            raise RuntimeError(
                "nvidia-smi unavailable; re-run with --skip-local for CPU-only flow."
            ) from e
        free_mb = int(result.stdout.strip())
        free_gb = free_mb / 1024
        if free_gb >= min_free_gb:
            print(f"  GPU free: {free_gb:.1f} GB >= {min_free_gb} GB. Proceeding.")
            return
        print(f"  GPU free: {free_gb:.1f} GB < {min_free_gb} GB. Waiting {poll_interval}s...")
        time.sleep(poll_interval)
        waited += poll_interval
    raise RuntimeError(
        f"GPU did not free up within {max_wait_s}s; re-run with --skip-local."
    )


def load_model():
    _ensure_local_deps()
    if not Path(MODEL_DIR).exists():
        raise FileNotFoundError(
            f"Local Qwen path not found: {MODEL_DIR}. Set $QWEN_MODEL_DIR or "
            f"re-run with --skip-local."
        )
    print(f"Loading Qwen2.5-7B from {MODEL_DIR}...")
    wait_for_gpu(min_free_gb=20)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, quantization_config=quant, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    print(f"Loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB")
    return model, tokenizer


def generate(model, tokenizer, prompt, system=""):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=512, temperature=0.0, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def cv_auc(X, y, seed=42):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    preds = np.zeros(len(y))
    for tr, va in skf.split(X, y):
        lr = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
        preds[va] = lr.predict_proba(X[va])[:, 1]
    return float(roc_auc_score(y, preds))


def bootstrap_auroc_ci(X, y, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            aucs.append(cv_auc(X[idx], y[idx]))
        except:
            pass
    return np.percentile(aucs, [2.5, 97.5]).tolist() if aucs else [None, None]


def lr_test(X_full, X_red, y):
    lr_f = LogisticRegression(max_iter=1000, C=1e6).fit(X_full, y)
    lr_r = LogisticRegression(max_iter=1000, C=1e6).fit(X_red, y)
    pf = np.clip(lr_f.predict_proba(X_full)[:, 1], 1e-12, 1-1e-12)
    pr = np.clip(lr_r.predict_proba(X_red)[:, 1], 1e-12, 1-1e-12)
    ll_f = float(np.sum(y*np.log(pf) + (1-y)*np.log(1-pf)))
    ll_r = float(np.sum(y*np.log(pr) + (1-y)*np.log(1-pr)))
    df = X_full.shape[1] - X_red.shape[1]
    chi2 = 2*(ll_f - ll_r)
    p = float(1 - stats.chi2.cdf(chi2, df)) if df > 0 else None
    return {"chi2": round(chi2, 4), "p": p}


def run_audit(ml, fl_aware, fl_blind, y, label=""):
    """Run the 4-step audit on arrays."""
    n = len(y)
    auc_ml = cv_auc(ml.reshape(-1,1), y)
    auc_aware = cv_auc(fl_aware.reshape(-1,1), y)
    auc_blind = cv_auc(fl_blind.reshape(-1,1), y)
    auc_fusion = cv_auc(np.column_stack([ml, fl_aware]), y)
    auc_combiner = cv_auc(np.column_stack([ml, fl_blind]), y)

    lrt = lr_test(np.column_stack([ml, fl_aware]), ml.reshape(-1,1), y)

    # TOST
    rng = np.random.RandomState(42)
    diffs = []
    for _ in range(1000):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y[idx])) < 2: continue
        try:
            d = cv_auc(np.column_stack([ml[idx], fl_aware[idx]]), y[idx]) - cv_auc(ml[idx].reshape(-1,1), y[idx])
            diffs.append(d)
        except: pass
    ci90 = np.percentile(diffs, [5, 95]).tolist() if diffs else [None, None]
    tost = ci90[0] > -0.03 and ci90[1] < 0.03 if ci90[0] is not None else False

    result = {
        "label": label, "n": n,
        "auroc_raw_ml": round(auc_ml, 4),
        "auroc_scoreaware_llm": round(auc_aware, 4),
        "auroc_scoreblind_llm": round(auc_blind, 4),
        "auroc_fusion": round(auc_fusion, 4),
        "auroc_combiner_ml_blind": round(auc_combiner, 4),
        "lr_test_llm_over_ml": lrt,
        "tost_003_equivalence": tost,
        "tost_ci90": [round(x, 5) for x in ci90] if ci90[0] is not None else None,
        "normative_verdict": (
            "rational_reliance" if auc_blind < 0.55
            else ("suppression" if auc_aware < auc_blind - 0.02 else "partial_integration")
        ),
    }
    return result


# ==========================================================
# EXPERIMENT 1: Medical cross-domain pilot (local Qwen)
# ==========================================================
def run_medical(model, tokenizer):
    print("\n" + "="*60)
    print("EXPERIMENT 1: Medical Cross-Domain Pilot (n=60)")
    print("="*60)

    SYSTEM = "You are a medical risk assessor. Evaluate cardiovascular risk."
    TPL_SCORE = "Assess cardiovascular risk.\n\n## Patient\n{profile}\n\n## ML Risk Score\nRisk: {score:.4f}\n\nJSON: {{\"risk_likelihood\": 0.0-1.0}}"
    TPL_BLIND = "Assess cardiovascular risk.\n\n## Patient\n{profile}\n\nJSON: {{\"risk_likelihood\": 0.0-1.0}}"

    rng = random.Random(42)
    patients = []
    for _ in range(30):
        is_hr = _ < 15
        if is_hr:
            age = rng.randint(60, 82); bp_s = rng.randint(150, 180); bp_d = rng.randint(95, 110)
            chol = rng.randint(240, 290); bmi = round(rng.uniform(30, 38), 1)
            score = round(rng.uniform(0.75, 0.92), 4)
        else:
            age = rng.randint(30, 55); bp_s = rng.randint(110, 135); bp_d = rng.randint(70, 85)
            chol = rng.randint(170, 210); bmi = round(rng.uniform(21, 27), 1)
            score = round(rng.uniform(0.75, 0.92), 4)
        profile = f"Age: {age}\nBP: {bp_s}/{bp_d}\nCholesterol: {chol}\nBMI: {bmi}\nSmoking: {'yes' if is_hr else 'no'}\nFamily CVD: {'yes' if is_hr else 'no'}"
        patients.append({"profile": profile, "score": score, "is_high_risk": is_hr})

    raw_results = []
    for i, pat in enumerate(patients):
        for cond, tpl in [("with_score", TPL_SCORE), ("without_score", TPL_BLIND)]:
            if cond == "with_score":
                prompt = tpl.format(profile=pat["profile"], score=pat["score"])
            else:
                prompt = tpl.format(profile=pat["profile"])
            resp = generate(model, tokenizer, prompt, SYSTEM)
            parsed = _parse_json_response(resp)
            fl = parsed.get("risk_likelihood") if isinstance(parsed, dict) else None
            raw_results.append({
                "patient_idx": i, "condition": cond, "is_high_risk": pat["is_high_risk"],
                "ml_score": pat["score"], "risk_likelihood": fl if isinstance(fl, (int,float)) else None,
            })
        if (i+1) % 10 == 0:
            print(f"  Medical [{i+1}/30]")

    # Save raw
    with open(RESULTS / "medical_local_raw.jsonl", "w") as f:
        for r in raw_results:
            f.write(json.dumps(r) + "\n")

    # Build arrays
    aware_fl, blind_fl, ml_scores, labels = [], [], [], []
    for i in range(30):
        w = next((r for r in raw_results if r["patient_idx"]==i and r["condition"]=="with_score"), None)
        wo = next((r for r in raw_results if r["patient_idx"]==i and r["condition"]=="without_score"), None)
        if w and wo and isinstance(w["risk_likelihood"], (int,float)) and isinstance(wo["risk_likelihood"], (int,float)):
            aware_fl.append(w["risk_likelihood"]); blind_fl.append(wo["risk_likelihood"])
            ml_scores.append(w["ml_score"]); labels.append(int(w["is_high_risk"]))

    if len(labels) < 10:
        print(f"  Only {len(labels)} valid pairs. Skipping audit.")
        return {"error": "too few valid pairs", "n_valid": len(labels)}

    ml = np.array(ml_scores); fl_a = np.array(aware_fl); fl_b = np.array(blind_fl); y = np.array(labels)
    audit = run_audit(ml, fl_a, fl_b, y, label="medical_local_qwen")

    # Paired shift
    shift = fl_a - fl_b
    audit["paired_shift_mean"] = round(float(shift.mean()), 4)
    audit["paired_shift_p"] = float(stats.ttest_1samp(shift, 0).pvalue) if len(shift) > 2 else None
    audit["alpha"] = round(float(shift.mean() / ml.mean()), 4) if ml.mean() > 0 else None

    with open(RESULTS / "medical_local_audit.json", "w") as f:
        json.dump(audit, f, indent=2)
    print(f"  Medical audit: α={audit.get('alpha')}, blind AUROC={audit['auroc_scoreblind_llm']}, verdict={audit['normative_verdict']}")
    return audit


# ==========================================================
# EXPERIMENT 2: PaySim n=200 expansion
# ==========================================================
def run_paysim_expansion(model, tokenizer):
    print("\n" + "="*60)
    print("EXPERIMENT 2: PaySim Expansion (target n=200)")
    print("="*60)

    from llm_explainer.prompts.fp_explanation import SYSTEM_PROMPT, build_prompt

    # Load existing score-blind
    blind_by_tid = {}
    with open(RESULTS / "scoreblind_paysim.jsonl") as f:
        for ln in f:
            r = json.loads(ln)
            orig = r.get("original", {})
            tid = orig.get("transaction_id")
            ba = r.get("blind_assessment", {})
            fl = ba.get("fraud_likelihood") if isinstance(ba, dict) else None
            if tid is not None and isinstance(fl, (int,float)):
                blind_by_tid[tid] = {
                    "fl_blind": float(fl), "ml": float(orig.get("fraud_score", 0)),
                    "y": int(orig.get("is_fraud", 0)), "text": orig.get("text", "")
                }
    print(f"  Loaded {len(blind_by_tid)} score-blind PaySim records")

    # Load existing score-aware
    aware_by_tid = {}
    for fname in ["explanations_paysim_50.jsonl", "explanations_paysim_gpt4o.jsonl"]:
        path = RESULTS / fname
        if not path.exists(): continue
        with open(path) as f:
            for ln in f:
                r = json.loads(ln)
                fp = r.get("fp_explanation", {})
                fl = fp.get("fraud_likelihood") if isinstance(fp, dict) else None
                orig = r.get("original", {})
                tid = orig.get("transaction_id")
                if tid is not None and isinstance(fl, (int,float)):
                    aware_by_tid[tid] = float(fl)

    # Identify TIDs that have blind but no aware → need generation
    need_gen = [tid for tid in blind_by_tid if tid not in aware_by_tid]
    print(f"  Already have {len(aware_by_tid)} score-aware. Need {len(need_gen)} more.")

    # Generate missing score-aware with local Qwen
    generated = []
    for i, tid in enumerate(need_gen[:125]):
        info = blind_by_tid[tid]
        if not info["text"]:
            continue
        prompt = build_prompt(info["text"], info["ml"], threshold=0.5)
        resp = generate(model, tokenizer, prompt, SYSTEM_PROMPT)
        parsed = _parse_json_response(resp)
        fl = parsed.get("fraud_likelihood") if isinstance(parsed, dict) else None
        generated.append({"transaction_id": tid, "fraud_likelihood": fl if isinstance(fl, (int,float)) else None, "raw": resp[:500]})
        if (i+1) % 25 == 0:
            print(f"  PaySim generation [{i+1}/{min(125, len(need_gen))}]")

    # Save generated
    with open(RESULTS / "paysim_expanded_scoreaware.jsonl", "w") as f:
        for g in generated:
            f.write(json.dumps(g) + "\n")

    # Merge: existing aware + new generated
    for g in generated:
        if g["fraud_likelihood"] is not None:
            aware_by_tid[g["transaction_id"]] = g["fraud_likelihood"]

    # Build paired arrays
    ml_arr, fl_a_arr, fl_b_arr, y_arr = [], [], [], []
    for tid, info in blind_by_tid.items():
        if tid in aware_by_tid:
            ml_arr.append(info["ml"]); fl_a_arr.append(aware_by_tid[tid])
            fl_b_arr.append(info["fl_blind"]); y_arr.append(info["y"])

    ml = np.array(ml_arr); fl_a = np.array(fl_a_arr); fl_b = np.array(fl_b_arr); y = np.array(y_arr)
    print(f"  Paired sample: n={len(y)} (FP={int((y==0).sum())}, TP={int((y==1).sum())})")

    if len(y) < 50:
        print("  Too few pairs. Skipping audit.")
        return {"error": "too few pairs", "n": len(y)}

    audit = run_audit(ml, fl_a, fl_b, y, label="paysim_expanded_qwen")
    with open(RESULTS / "paysim_expanded_audit.json", "w") as f:
        json.dump(audit, f, indent=2)
    print(f"  PaySim expanded: n={audit['n']}, LR p={audit['lr_test_llm_over_ml']['p']:.4f}, blind AUROC={audit['auroc_scoreblind_llm']}, verdict={audit['normative_verdict']}")
    return audit


# ==========================================================
# EXPERIMENT 3: Rich features audit
# ==========================================================
def run_rich_features(model, tokenizer):
    print("\n" + "="*60)
    print("EXPERIMENT 3: Rich Features Audit")
    print("="*60)

    from llm_explainer.prompts.fp_explanation import SYSTEM_PROMPT

    # Load PaySim blind data and enrich features
    blind_data = []
    with open(RESULTS / "scoreblind_paysim.jsonl") as f:
        for ln in f:
            r = json.loads(ln)
            orig = r.get("original", {})
            ba = r.get("blind_assessment", {})
            fl_blind = ba.get("fraud_likelihood") if isinstance(ba, dict) else None
            if isinstance(fl_blind, (int, float)):
                blind_data.append({
                    "tid": orig.get("transaction_id"),
                    "ml": float(orig.get("fraud_score", 0)),
                    "y": int(orig.get("is_fraud", 0)),
                    "text": orig.get("text", ""),
                    "fl_blind": float(fl_blind),
                })

    rng = random.Random(42)
    merchants = ["Amazon Inc.", "Walmart Stores", "Shell Gas Station", "Uber Technologies",
                 "Netflix Subscription", "DoorDash Delivery", "Apple Store", "Target Corp",
                 "CVS Pharmacy", "Home Depot", "Starbucks Coffee", "McDonald's"]
    devices = ["iPhone 15 Pro (iOS 17.4)", "Samsung Galaxy S24 (Android 14)",
               "Chrome on Windows 11", "Safari on MacBook Pro", "Mobile App v4.2.1"]
    histories = [
        "Account active 3 years, 847 prior transactions, 0 prior disputes",
        "Account active 6 months, 23 prior transactions, 1 resolved dispute",
        "Account active 5 years, 2,341 prior transactions, 0 disputes",
        "New account (2 weeks), 4 prior transactions, no history",
        "Account active 1 year, 156 prior transactions, 2 chargebacks",
    ]

    def enrich(text, rng):
        merchant = rng.choice(merchants)
        device = rng.choice(devices)
        history = rng.choice(histories)
        return (
            f"{text}\n"
            f"Merchant: {merchant}\n"
            f"Device: {device}\n"
            f"Account history: {history}\n"
            f"Location: consistent with prior transactions\n"
            f"Velocity: 2 transactions in last 24h (normal for this account)"
        )

    # Take first 100 (balanced by construction from scoreblind file)
    sample = blind_data[:100]

    TEMPLATE_SCORE = """You are a senior fraud analyst reviewing an ML-flagged transaction.

Transaction Details:
{text}

ML Fraud Score: {score:.4f} (threshold: 0.50)

Provide balanced assessment as JSON: {{"fraud_likelihood": 0.0-1.0, "recommendation": "RELEASE|ESCALATE"}}"""

    TEMPLATE_BLIND = """You are a senior fraud analyst reviewing a transaction.

Transaction Details:
{text}

Provide balanced assessment as JSON: {{"fraud_likelihood": 0.0-1.0, "recommendation": "RELEASE|ESCALATE"}}"""

    raw_results = []
    for i, item in enumerate(sample):
        enriched = enrich(item["text"], rng)

        # Score-aware
        prompt_a = TEMPLATE_SCORE.format(text=enriched, score=item["ml"])
        resp_a = generate(model, tokenizer, prompt_a, SYSTEM_PROMPT)
        parsed_a = _parse_json_response(resp_a)
        fl_a = parsed_a.get("fraud_likelihood") if isinstance(parsed_a, dict) else None

        # Score-blind
        prompt_b = TEMPLATE_BLIND.format(text=enriched)
        resp_b = generate(model, tokenizer, prompt_b, SYSTEM_PROMPT)
        parsed_b = _parse_json_response(resp_b)
        fl_b = parsed_b.get("fraud_likelihood") if isinstance(parsed_b, dict) else None

        raw_results.append({
            "tid": item["tid"], "ml": item["ml"], "y": item["y"],
            "fl_aware": fl_a if isinstance(fl_a, (int,float)) else None,
            "fl_blind_rich": fl_b if isinstance(fl_b, (int,float)) else None,
            "fl_blind_original": item["fl_blind"],
        })
        if (i+1) % 20 == 0:
            print(f"  Rich features [{i+1}/100]")

    with open(RESULTS / "rich_features_raw.jsonl", "w") as f:
        for r in raw_results:
            f.write(json.dumps(r) + "\n")

    # Build arrays (only valid parses)
    ml_arr, fl_a_arr, fl_b_arr, y_arr = [], [], [], []
    for r in raw_results:
        if r["fl_aware"] is not None and r["fl_blind_rich"] is not None:
            ml_arr.append(r["ml"]); fl_a_arr.append(r["fl_aware"])
            fl_b_arr.append(r["fl_blind_rich"]); y_arr.append(r["y"])

    ml = np.array(ml_arr); fl_a = np.array(fl_a_arr); fl_b = np.array(fl_b_arr); y = np.array(y_arr)
    print(f"  Rich features valid: n={len(y)} (FP={int((y==0).sum())}, TP={int((y==1).sum())})")

    if len(y) < 30:
        print("  Too few valid. Skipping audit.")
        return {"error": "too few valid", "n": len(y)}

    audit = run_audit(ml, fl_a, fl_b, y, label="rich_features_qwen")

    # Paired shift (rich blind vs score-aware)
    shift = fl_a - fl_b
    audit["paired_shift_mean"] = round(float(shift.mean()), 4)
    audit["alpha_rich"] = round(float(shift.mean() / ml.mean()), 4) if ml.mean() > 0 else None

    with open(RESULTS / "rich_features_audit.json", "w") as f:
        json.dump(audit, f, indent=2)
    print(f"  Rich features: α={audit.get('alpha_rich')}, blind AUROC={audit['auroc_scoreblind_llm']}, verdict={audit['normative_verdict']}")
    return audit


# ==========================================================
# MAIN
# ==========================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--skip-local", action="store_true",
        help="Skip the local-Qwen branch entirely (CPU-only / no GPU available). "
             "The script then exits with a message; downstream callers should "
             "use already-collected artifacts.",
    )
    parser.add_argument(
        "--qwen-path", default=None,
        help="Override the Qwen2.5-7B-Instruct local path (default: $QWEN_MODEL_DIR).",
    )
    args = parser.parse_args()

    if args.qwen_path:
        global MODEL_DIR
        MODEL_DIR = args.qwen_path

    if args.skip_local:
        print("--skip-local set: not loading local Qwen. "
              "Use the already-collected artifacts in experiments/results/ "
              "(paysim_expanded_*, rich_features_*, medical_local_*).")
        return 0

    start = time.time()
    try:
        model, tokenizer = load_model()
    except (ImportError, FileNotFoundError, RuntimeError) as e:
        print(f"\nERROR loading local Qwen: {e}\n"
              f"Re-run with --skip-local to bypass this branch.")
        return 2

    r1 = run_medical(model, tokenizer)
    r2 = run_paysim_expansion(model, tokenizer)
    r3 = run_rich_features(model, tokenizer)

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"ALL DONE in {elapsed/60:.1f} min")
    print(f"{'='*60}")
    print(f"Medical: verdict={r1.get('normative_verdict','error')}, α={r1.get('alpha')}")
    print(f"PaySim expanded: n={r2.get('n','?')}, verdict={r2.get('normative_verdict','error')}")
    print(f"Rich features: verdict={r3.get('normative_verdict','error')}, α={r3.get('alpha_rich')}")

    summary = {"medical": r1, "paysim_expanded": r2, "rich_features": r3,
               "total_minutes": round(elapsed/60, 1)}
    with open(RESULTS / "three_expansions_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
