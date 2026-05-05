"""SHAP-enhanced transaction text → LLM에게 SHAP 정보 제공 후 판단 개선 측정."""

import json, os, sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm_explainer.llm_inference import LLMConfig, LLMInference

RESULTS_DIR = Path(__file__).resolve().parent / "results"

SYSTEM = "You are a financial fraud investigator. Assess the transaction below for fraud risk."
PROMPT = """Assess this transaction for fraud risk.

## Transaction Details
{tx_text}

## ML Model Score
Fraud probability: {score:.4f}

## Feature Attribution (SHAP)
{shap_text}

Evaluate BOTH sides. Respond in JSON:
```json
{{"fraud_likelihood": 0.0-1.0, "recommendation": "ESCALATE|HOLD|RELEASE"}}
```"""


def run(shap_path, model="gpt-4o-mini", max_samples=50):
    config = LLMConfig(model_name=model, backend="openai", temperature=0.1)
    llm = LLMInference(config)
    llm.load()

    with open(shap_path) as f:
        shap_data = json.load(f)[:max_samples]

    results = []
    for i, s in enumerate(shap_data):
        prompt = PROMPT.format(
            tx_text=s["transaction_text"][:300],
            score=s["fraud_score"],
            shap_text=s["shap_explanation"],
        )
        r = llm.generate_json(prompt, SYSTEM)
        r["category"] = s["category"]
        r["fraud_score"] = s["fraud_score"]
        results.append(r)

        if (i+1) % 10 == 0:
            print(f"  [{i+1}/{len(shap_data)}]")

    fp_fl = [r.get("fraud_likelihood",0) for r in results if r["category"]=="false_positive" and isinstance(r.get("fraud_likelihood"),(int,float))]
    tp_fl = [r.get("fraud_likelihood",0) for r in results if r["category"]=="true_positive" and isinstance(r.get("fraud_likelihood"),(int,float))]

    print(f"\n=== SHAP-Enhanced LLM Results ===")
    print(f"FP: {np.mean(fp_fl):.3f} (n={len(fp_fl)})")
    print(f"TP: {np.mean(tp_fl):.3f} (n={len(tp_fl)})")
    print(f"Δ: {np.mean(tp_fl)-np.mean(fp_fl):+.3f}")
    print(f"\nCompare with standard (no SHAP): FP=0.873, TP=0.927, Δ=+0.054")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "shap_enhanced_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--shap", required=True)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max_samples", type=int, default=50)
    args = parser.parse_args()
    run(args.shap, args.model, args.max_samples)
