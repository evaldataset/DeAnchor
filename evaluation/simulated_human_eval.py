"""Simulated Human Evaluation.

GPT-4o를 fraud analyst 역할로 설정하여 인간 평가를 시뮬레이션.
실제 human eval 전 pilot으로 사용.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_explainer.llm_inference import LLMConfig, LLMInference

EVAL_DIR = Path(__file__).resolve().parent.parent / "data" / "benchmarks" / "human_eval"

ANALYST_SYSTEM = """You are a senior fraud analyst with 15 years of experience at a major bank's Financial Crimes unit. You are evaluating AI-generated fraud explanations for their usefulness in your daily work.

Rate each explanation honestly as if you would use it in a real investigation. Be critical — you've seen many automated reports and know the difference between useful analysis and generic filler.

Your evaluation criteria:
- Coherence (1-5): Does the reasoning make sense? Are the conclusions supported by the evidence cited?
- Completeness (1-5): Does it cover the key aspects you'd want to know? Missing anything important?
- Clarity (1-5): Is it written clearly enough to include in a case file?
- Actionability (1-5): Does it tell you what to do next? Would it save you investigation time?
- Overall (1-5): Would you find this useful in your work?

Be realistic — most automated reports deserve a 3 (adequate). Reserve 5 for truly exceptional analysis."""

ANALYST_PROMPT = """You're reviewing an AI-generated explanation for a flagged transaction. Rate it as you would in your daily workflow.

## Transaction
{transaction_text}

## ML Fraud Score: {ml_score:.2f}

## AI-Generated Analysis
{explanation}

Rate 1-5 on each dimension. Be honest and critical.

```json
{{
    "coherence": <1-5>,
    "completeness": <1-5>,
    "clarity": <1-5>,
    "actionability": <1-5>,
    "overall": <1-5>,
    "would_use_in_investigation": true/false,
    "brief_feedback": "1-2 sentences as a fraud analyst"
}}
```"""


def run_simulated_eval(
    predictions_path: str,
    model: str = "gpt-4o-mini",
    n_samples: int = 50,
) -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set")
        return

    config = LLMConfig(model_name=model, backend="openai", temperature=0.3)
    llm = LLMInference(config)
    llm.load()

    with open(predictions_path) as f:
        preds = [json.loads(line) for line in f]

    # Balanced sampling
    import random
    random.seed(42)
    fp = [p for p in preds if p.get("original", {}).get("category") == "false_positive"]
    tp = [p for p in preds if p.get("original", {}).get("category") == "true_positive"]
    n_each = n_samples // 2
    samples = random.sample(fp, min(n_each, len(fp))) + random.sample(tp, min(n_each, len(tp)))
    random.shuffle(samples)

    print(f"Simulated human evaluation: {len(samples)} samples")

    results = []
    for i, pred in enumerate(samples):
        tx = pred.get("original", {})
        exp = pred.get("fp_explanation") or pred.get("anomaly_explanation", {})
        exp_text = json.dumps(exp, indent=2) if isinstance(exp, dict) else str(exp)

        prompt = ANALYST_PROMPT.format(
            transaction_text=tx.get("text", ""),
            ml_score=pred.get("ml_score", 0),
            explanation=exp_text[:800],
        )

        result = llm.generate_json(prompt, ANALYST_SYSTEM)
        result["category"] = tx.get("category", "unknown")
        result["sample_id"] = i + 1
        results.append(result)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(samples)}]")

    # Aggregate
    import numpy as np
    dims = ["coherence", "completeness", "clarity", "actionability", "overall"]

    print(f"\n=== Simulated Human Evaluation Results ===")
    summary = {}
    for dim in dims:
        scores = [r.get(dim, 0) for r in results if isinstance(r.get(dim), (int, float))]
        if scores:
            summary[dim] = {"mean": round(np.mean(scores), 2), "std": round(np.std(scores), 2), "n": len(scores)}
            print(f"  {dim:15s}: {np.mean(scores):.2f} ± {np.std(scores):.2f}")

    # would_use breakdown
    use_count = sum(1 for r in results if r.get("would_use_in_investigation") is True)
    print(f"\n  Would use in investigation: {use_count}/{len(results)} ({use_count/len(results):.0%})")

    # FP vs TP comparison
    fp_scores = [r.get("overall", 0) for r in results if r.get("category") == "false_positive" and isinstance(r.get("overall"), (int, float))]
    tp_scores = [r.get("overall", 0) for r in results if r.get("category") == "true_positive" and isinstance(r.get("overall"), (int, float))]
    if fp_scores and tp_scores:
        print(f"\n  FP overall: {np.mean(fp_scores):.2f} ± {np.std(fp_scores):.2f}")
        print(f"  TP overall: {np.mean(tp_scores):.2f} ± {np.std(tp_scores):.2f}")

    # Save
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_DIR / "simulated_human_eval_results.json"
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")

    # Sample feedback
    print(f"\n=== Sample Analyst Feedback ===")
    for r in results[:5]:
        fb = r.get("brief_feedback", "N/A")
        print(f"  S{r['sample_id']:03d} [{r['category']}] overall={r.get('overall','?')}: {fb}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--n_samples", type=int, default=50)
    args = parser.parse_args()

    run_simulated_eval(args.predictions, args.model, args.n_samples)


if __name__ == "__main__":
    main()
