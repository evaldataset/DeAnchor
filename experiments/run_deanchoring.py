"""De-anchoring 실험: 단계적 score 제공으로 anchoring bias 감소 시도.

3가지 condition:
  A) Score-first: ML score → 거래 텍스트 → 판단 (현재 방식)
  B) Score-blind: 거래 텍스트만 → 판단 (이미 실행)
  C) Staged: 1단계 blind 판단 → 2단계 score 확인 → 최종 판단 (제안 방법)
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

SYSTEM_PROMPT = """You are a senior financial fraud investigator conducting a two-stage assessment. Follow the instructions exactly."""

STAGE1_PROMPT = """STAGE 1: Assess this transaction based ONLY on its characteristics. Do NOT consider any ML model output.

## Transaction Details
{transaction_text}

Provide your initial independent assessment:
```json
{{
    "initial_fraud_likelihood": 0.0-1.0,
    "initial_assessment": "Your independent analysis",
    "key_observations": ["observation 1", "observation 2"]
}}
```"""

STAGE2_PROMPT = """STAGE 2: You previously assessed this transaction independently (Stage 1). Now consider the ML model's fraud score and provide your FINAL assessment.

## Your Stage 1 Assessment
- Initial fraud likelihood: {initial_fl}
- Initial assessment: {initial_assessment}

## ML Model Information
- Fraud score: {ml_score:.4f} ({confidence_level})

## Transaction Details (reminder)
{transaction_text}

Now provide your FINAL assessment, integrating your independent judgment with the ML score:
```json
{{
    "final_fraud_likelihood": 0.0-1.0,
    "adjustment_from_initial": "Did the ML score change your assessment? How?",
    "evidence_for_fraud": ["..."],
    "evidence_against_fraud": ["..."],
    "recommendation": "ESCALATE_AS_FRAUD|HOLD_FOR_REVIEW|RELEASE",
    "confidence": 0.0-1.0
}}
```"""


def run_staged(
    input_path: str,
    output_path: str,
    model: str = "gpt-4o-mini",
    max_samples: int | None = None,
    threshold: float = 0.5,
) -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        # Try local
        backend = "local" if "/" in model else "openai"
        if backend == "openai":
            print("OPENAI_API_KEY not set")
            return
    else:
        backend = "local" if "/" in model else "openai"

    config = LLMConfig(model_name=model, backend=backend, temperature=0.1)
    llm = LLMInference(config)
    llm.load()

    transactions = []
    with open(input_path) as f:
        for line in f:
            transactions.append(json.loads(line))

    if max_samples:
        transactions = transactions[:max_samples]

    print(f"Staged de-anchoring: {len(transactions)} transactions")
    start = time.time()

    with open(output_path, "w") as f:
        for i, tx in enumerate(transactions):
            print(f"  [{i+1}/{len(transactions)}] {tx.get('category', '?')}", end=" ", flush=True)

            # Stage 1: Blind assessment
            prompt1 = STAGE1_PROMPT.format(transaction_text=tx["text"])
            stage1 = llm.generate_json(prompt1, SYSTEM_PROMPT)

            initial_fl = stage1.get("initial_fraud_likelihood", 0.5)
            initial_assessment = stage1.get("initial_assessment", "")

            # Stage 2: Score-informed revision
            ml_score = tx.get("fraud_score", 0.5)
            if ml_score > 0.99:
                conf = "extremely confident >99%"
            elif ml_score > 0.95:
                conf = "very confident >95%"
            elif ml_score > 0.90:
                conf = "confident >90%"
            else:
                conf = f"moderately confident ({ml_score:.0%})"

            prompt2 = STAGE2_PROMPT.format(
                initial_fl=initial_fl,
                initial_assessment=initial_assessment[:200],
                ml_score=ml_score,
                confidence_level=conf,
                transaction_text=tx["text"],
            )
            stage2 = llm.generate_json(prompt2, SYSTEM_PROMPT)

            output = {
                "original": tx,
                "stage1_blind": stage1,
                "stage2_informed": stage2,
                "initial_fl": initial_fl if isinstance(initial_fl, (int, float)) else None,
                "final_fl": stage2.get("final_fraud_likelihood"),
                "ml_score": ml_score,
            }
            f.write(json.dumps(output, ensure_ascii=False) + "\n")

            final_fl = stage2.get("final_fraud_likelihood", "?")
            print(f"init={initial_fl} → final={final_fl}")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.0f}s → {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    output = args.output or str(RESULTS_DIR / "staged_deanchoring.jsonl")
    run_staged(args.input, output, args.model, args.max_samples, args.threshold)


if __name__ == "__main__":
    main()
