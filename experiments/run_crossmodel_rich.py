"""Cross-model replication of rich-features audit on DeepSeek-14B and Gemma-9B.

Demonstrates that score-conditioned suppression is not a Qwen-7B-specific artifact
by replicating the enriched-features experiment on two additional model families.

Uses the same enriched PaySim texts from run_three_expansions.py.

Outputs:
    experiments/results/crossmodel_rich_audit.json
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
from llm_explainer.prompts.fp_explanation import SYSTEM_PROMPT

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "experiments" / "results"

MODELS = [
    {
        "name": "DeepSeek-R1-14B",
        "path": os.path.expanduser("~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-14B"),
        "use_snap": True,
    },
    {
        "name": "Gemma-2-9B",
        "path": os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-2-9b-it"),
        "use_snap": True,
    },
]

TEMPLATE_SCORE = """You are a senior fraud analyst reviewing an ML-flagged transaction.

Transaction Details:
{text}

ML Fraud Score: {score:.4f} (threshold: 0.50)

Provide balanced assessment as JSON: {{"fraud_likelihood": 0.0-1.0, "recommendation": "RELEASE|ESCALATE"}}"""

TEMPLATE_BLIND = """You are a senior fraud analyst reviewing a transaction.

Transaction Details:
{text}

Provide balanced assessment as JSON: {{"fraud_likelihood": 0.0-1.0, "recommendation": "RELEASE|ESCALATE"}}"""


def find_snapshot(model_path):
    snap_dir = Path(model_path) / "snapshots"
    if snap_dir.exists():
        snaps = sorted(snap_dir.iterdir())
        if snaps:
            return str(snaps[-1])
    return model_path


def load_model(model_info):
    path = find_snapshot(model_info["path"]) if model_info.get("use_snap") else model_info["path"]
    print(f"  Loading {model_info['name']} from {path}")
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


def generate(model, tokenizer, prompt, system=""):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = f"{system}\n\n{prompt}"
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


def prepare_enriched_data():
    """Load score-blind PaySim data and enrich with synthetic context."""
    blind_data = []
    with open(RESULTS / "scoreblind_paysim.jsonl") as f:
        for ln in f:
            r = json.loads(ln)
            orig = r.get("original", {})
            ba = r.get("blind_assessment", {})
            fl = ba.get("fraud_likelihood") if isinstance(ba, dict) else None
            if isinstance(fl, (int, float)):
                blind_data.append({
                    "tid": orig.get("transaction_id"),
                    "ml": float(orig.get("fraud_score", 0)),
                    "y": int(orig.get("is_fraud", 0)),
                    "text": orig.get("text", ""),
                    "fl_blind": float(fl),
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

    enriched = []
    for item in blind_data[:100]:
        rich_text = (
            f"{item['text']}\n"
            f"Merchant: {rng.choice(merchants)}\n"
            f"Device: {rng.choice(devices)}\n"
            f"Account history: {rng.choice(histories)}\n"
            f"Location: consistent with prior transactions\n"
            f"Velocity: 2 transactions in last 24h (normal for this account)"
        )
        item["rich_text"] = rich_text
        enriched.append(item)
    return enriched


def run_model_audit(model_info, data):
    name = model_info["name"]
    print(f"\n{'='*60}")
    print(f"Running rich-features audit with {name}")
    print(f"{'='*60}")

    model, tokenizer = load_model(model_info)

    results = []
    for i, item in enumerate(data):
        # Score-aware
        prompt_a = TEMPLATE_SCORE.format(text=item["rich_text"], score=item["ml"])
        resp_a = generate(model, tokenizer, prompt_a, SYSTEM_PROMPT)
        parsed_a = _parse_json_response(resp_a)
        fl_a = parsed_a.get("fraud_likelihood") if isinstance(parsed_a, dict) else None

        # Score-blind
        prompt_b = TEMPLATE_BLIND.format(text=item["rich_text"])
        resp_b = generate(model, tokenizer, prompt_b, SYSTEM_PROMPT)
        parsed_b = _parse_json_response(resp_b)
        fl_b = parsed_b.get("fraud_likelihood") if isinstance(parsed_b, dict) else None

        results.append({
            "tid": item["tid"], "ml": item["ml"], "y": item["y"],
            "fl_aware": fl_a if isinstance(fl_a, (int,float)) else None,
            "fl_blind": fl_b if isinstance(fl_b, (int,float)) else None,
        })
        if (i+1) % 20 == 0:
            valid = sum(1 for r in results if r["fl_aware"] is not None and r["fl_blind"] is not None)
            print(f"  [{i+1}/100] valid={valid}")

    # Free GPU
    del model, tokenizer
    torch.cuda.empty_cache()

    # Audit
    ml_arr, fl_a_arr, fl_b_arr, y_arr = [], [], [], []
    for r in results:
        if r["fl_aware"] is not None and r["fl_blind"] is not None:
            ml_arr.append(r["ml"]); fl_a_arr.append(r["fl_aware"])
            fl_b_arr.append(r["fl_blind"]); y_arr.append(r["y"])

    ml = np.array(ml_arr); fl_a = np.array(fl_a_arr); fl_b = np.array(fl_b_arr); y = np.array(y_arr)
    n = len(y)
    print(f"  Valid pairs: {n} (FP={int((y==0).sum())}, TP={int((y==1).sum())})")

    if n < 20:
        return {"model": name, "error": "too few valid", "n_valid": n}

    auc_ml = cv_auc(ml.reshape(-1,1), y)
    auc_aware = cv_auc(fl_a.reshape(-1,1), y)
    auc_blind = cv_auc(fl_b.reshape(-1,1), y)
    auc_fusion = cv_auc(np.column_stack([ml, fl_a]), y)
    lrt = lr_test(np.column_stack([ml, fl_a]), ml.reshape(-1,1), y)

    shift = fl_a - fl_b
    alpha = round(float(shift.mean() / ml.mean()), 4) if ml.mean() > 0 else None

    audit = {
        "model": name, "n": n,
        "auroc_raw_ml": round(auc_ml, 4),
        "auroc_scoreaware": round(auc_aware, 4),
        "auroc_scoreblind": round(auc_blind, 4),
        "auroc_fusion": round(auc_fusion, 4),
        "lr_test_llm_over_ml": lrt,
        "alpha_rich": alpha,
        "suppression": round(float(auc_blind - auc_aware), 4),
        "verdict": "suppression" if auc_aware < auc_blind - 0.02 else "no_suppression",
    }
    print(f"  {name}: blind={auc_blind:.3f}, aware={auc_aware:.3f}, suppression={auc_blind-auc_aware:.3f}, α={alpha}")
    return audit


def main():
    start = time.time()
    data = prepare_enriched_data()
    print(f"Prepared {len(data)} enriched transactions")

    all_results = {}
    for mi in MODELS:
        result = run_model_audit(mi, data)
        all_results[result["model"]] = result

    elapsed = time.time() - start
    all_results["total_minutes"] = round(elapsed/60, 1)

    with open(RESULTS / "crossmodel_rich_audit.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'='*60}")
    print(f"ALL DONE in {elapsed/60:.1f} min")
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
