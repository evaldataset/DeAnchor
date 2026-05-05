"""Controlled ablation: 프롬프트 동일, ML score만 제거/포함.

Confound 통제: 시스템 프롬프트, 출력 형식, 지시 모두 동일.
유일한 차이 = ML score 존재 여부.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm_explainer.llm_inference import LLMConfig, LLMInference

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# 동일한 시스템 프롬프트 (두 조건 모두)
SYSTEM = """You are a financial fraud investigator. Assess the transaction below for fraud risk. Provide a balanced, evidence-based assessment."""

# 동일한 출력 형식, 동일한 지시. 유일한 차이: {score_section}
PROMPT_TEMPLATE = """Assess this transaction for fraud risk.

## Transaction Details
{transaction_text}
{score_section}
Evaluate BOTH sides: evidence for fraud AND evidence against fraud.

Respond in JSON:
```json
{{
    "fraud_likelihood": 0.0-1.0,
    "evidence_for_fraud": ["..."],
    "evidence_against_fraud": ["..."],
    "recommendation": "ESCALATE_AS_FRAUD|HOLD_FOR_REVIEW|RELEASE",
    "confidence": 0.0-1.0
}}
```"""


def run_controlled(input_path, output_path, model="gpt-4o-mini", max_samples=None):
    backend = "local" if "/" in model else "openai"
    config = LLMConfig(model_name=model, backend=backend, temperature=0.1)
    llm = LLMInference(config)
    llm.load()

    with open(input_path) as f:
        txns = [json.loads(line) for line in f]
    if max_samples:
        txns = txns[:max_samples]

    print(f"Controlled ablation: {len(txns)} transactions, 2 conditions each")

    with_score = []
    without_score = []

    for i, tx in enumerate(txns):
        cat = tx.get("category", "?")
        ml_score = tx.get("fraud_score", 0.5)

        # Condition A: WITH score
        score_section = f"\n## ML Model Score\nFraud probability: {ml_score:.4f}\n"
        prompt_a = PROMPT_TEMPLATE.format(
            transaction_text=tx["text"], score_section=score_section
        )
        result_a = llm.generate_json(prompt_a, SYSTEM)

        # Condition B: WITHOUT score (identical prompt minus score section)
        prompt_b = PROMPT_TEMPLATE.format(
            transaction_text=tx["text"], score_section=""
        )
        result_b = llm.generate_json(prompt_b, SYSTEM)

        fl_a = result_a.get("fraud_likelihood", "?")
        fl_b = result_b.get("fraud_likelihood", "?")
        print(f"  [{i+1}/{len(txns)}] {cat}: with={fl_a}, without={fl_b}")

        with_score.append({"original": tx, "assessment": result_a, "condition": "with_score"})
        without_score.append({"original": tx, "assessment": result_b, "condition": "without_score"})

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    base = Path(output_path).stem
    for suffix, data in [("_with_score.jsonl", with_score), ("_without_score.jsonl", without_score)]:
        path = RESULTS_DIR / f"{base}{suffix}"
        with open(path, "w") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Saved to {RESULTS_DIR}/{base}_*.jsonl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="controlled_ablation")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    run_controlled(args.input, args.output, args.model, args.max_samples)


if __name__ == "__main__":
    main()
