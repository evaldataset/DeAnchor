"""Score-blind 실험: ML 점수 없이 LLM 독립 판단.

Anchoring bias 제거 시 FP/TP 구분력 변화 측정.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_explainer.llm_inference import LLMConfig, LLMInference
from llm_explainer.prompts.fp_explanation_blind import SYSTEM_PROMPT, build_prompt

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def run_scoreblind(
    input_path: str,
    output_path: str,
    model: str = "gpt-4o-mini",
    max_samples: int | None = None,
) -> None:
    transactions = []
    with open(input_path) as f:
        for line in f:
            transactions.append(json.loads(line))

    if max_samples:
        transactions = transactions[:max_samples]

    # Auto-detect backend: local path → local, otherwise openai
    backend = "local" if "/" in model else "openai"
    config = LLMConfig(model_name=model, backend=backend)
    llm = LLMInference(config)
    llm.load()

    print(f"Score-blind evaluation: {len(transactions)} transactions")
    start = time.time()

    with open(output_path, "w") as f:
        for i, tx in enumerate(transactions):
            print(f"  [{i+1}/{len(transactions)}] {tx.get('category', '?')}", end=" ", flush=True)

            prompt = build_prompt(tx["text"])
            result = llm.generate_json(prompt, SYSTEM_PROMPT)

            output = {
                "original": tx,
                "blind_assessment": result,
            }
            f.write(json.dumps(output, ensure_ascii=False) + "\n")
            print(f"fl={result.get('fraud_likelihood', '?')}")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.0f}s → {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    output = args.output or str(RESULTS_DIR / "scoreblind_results.jsonl")
    run_scoreblind(args.input, output, args.model, args.max_samples)


if __name__ == "__main__":
    main()
