"""Dose-response: 같은 거래에 ML score를 0.5/0.7/0.9로 변화시키며 LLM 반응 측정."""

import argparse, json, os, sys, time
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

Evaluate BOTH sides. Respond in JSON:
```json
{{"fraud_likelihood": 0.0-1.0, "recommendation": "ESCALATE|HOLD|RELEASE"}}
```"""


def run(input_path, model="gpt-4o-mini", max_samples=50):
    config = LLMConfig(model_name=model, backend="openai", temperature=0.1)
    llm = LLMInference(config)
    llm.load()

    with open(input_path) as f:
        txns = [json.loads(line) for line in f][:max_samples]

    scores = [0.1, 0.3, 0.5, 0.7, 0.9]
    results = []

    for i, tx in enumerate(txns):
        row = {"original": tx, "responses": {}}
        for s in scores:
            prompt = PROMPT.format(tx_text=tx["text"], score=s)
            r = llm.generate_json(prompt, SYSTEM)
            row["responses"][str(s)] = r

        fl_values = {s: row["responses"][str(s)].get("fraud_likelihood", "?") for s in scores}
        print(f"  [{i+1}/{len(txns)}] {tx['category']}: " + " → ".join(f"{s}:{fl_values[s]}" for s in scores))
        results.append(row)

    # Analysis
    import numpy as np
    print(f"\n=== Dose-Response Analysis (n={len(results)}) ===")
    for s in scores:
        fp_fl = [r["responses"][str(s)].get("fraud_likelihood",0) for r in results
                 if r["original"]["category"]=="false_positive" and isinstance(r["responses"][str(s)].get("fraud_likelihood"),(int,float))]
        tp_fl = [r["responses"][str(s)].get("fraud_likelihood",0) for r in results
                 if r["original"]["category"]=="true_positive" and isinstance(r["responses"][str(s)].get("fraud_likelihood"),(int,float))]
        if fp_fl and tp_fl:
            print(f"  Score={s}: FP={np.mean(fp_fl):.3f}, TP={np.mean(tp_fl):.3f}, Δ={np.mean(tp_fl)-np.mean(fp_fl):+.3f}")

    # Slope analysis
    print(f"\n  Per-transaction slope (fl change per 0.1 score increase):")
    slopes = []
    for r in results:
        fls = [r["responses"][str(s)].get("fraud_likelihood") for s in scores]
        if all(isinstance(f,(int,float)) for f in fls):
            slope = (fls[4] - fls[2]) / 0.4  # (fl@0.9 - fl@0.5) / (0.9-0.5)
            slopes.append(slope)
    if slopes:
        print(f"    Mean slope: {np.mean(slopes):.3f} per 0.1 score unit")
        print(f"    This means: +0.1 ML score → +{np.mean(slopes)*0.1:.3f} LLM fraud_likelihood")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "dose_response.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max_samples", type=int, default=50)
    args = parser.parse_args()
    run(args.input, args.model, args.max_samples)
