"""RAG 유/무 비교 실험.

동일 거래에 대해 RAG context가 설명 품질에 미치는 영향 측정.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_explainer.llm_inference import LLMConfig, LLMInference
from llm_explainer.prompts import anomaly_explanation
from llm_explainer.pipeline import _load_rag_searcher

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def run_rag_comparison(
    input_path: str,
    model: str = "gpt-4o-mini",
    max_samples: int | None = None,
) -> None:
    transactions = []
    with open(input_path) as f:
        for line in f:
            transactions.append(json.loads(line))

    if max_samples:
        transactions = transactions[:max_samples]

    config = LLMConfig(model_name=model, backend="openai")
    llm = LLMInference(config)
    llm.load()

    rag = _load_rag_searcher(None)
    print(f"RAG comparison: {len(transactions)} transactions, RAG={'loaded' if rag else 'none'}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_no_rag = RESULTS_DIR / "rag_comparison_no_rag.jsonl"
    out_with_rag = RESULTS_DIR / "rag_comparison_with_rag.jsonl"

    # Condition 1: No RAG
    print("\n=== Condition: No RAG ===")
    with open(out_no_rag, "w") as f:
        for i, tx in enumerate(transactions):
            print(f"  [{i+1}/{len(transactions)}]", end=" ", flush=True)
            prompt = anomaly_explanation.build_prompt(
                tx["text"], tx.get("fraud_score", 0.5), 0.5, None,
            )
            result = llm.generate_json(prompt, anomaly_explanation.SYSTEM_PROMPT)
            output = {"original": tx, "explanation": result, "condition": "no_rag"}
            f.write(json.dumps(output, ensure_ascii=False) + "\n")
            print("done")

    # Condition 2: With RAG
    print("\n=== Condition: With RAG ===")
    with open(out_with_rag, "w") as f:
        for i, tx in enumerate(transactions):
            print(f"  [{i+1}/{len(transactions)}]", end=" ", flush=True)

            similar_cases = None
            if rag:
                similar_cases = rag.search_for_transaction(
                    tx["text"], tx.get("fraud_score", 0.5), top_k=3,
                )

            prompt = anomaly_explanation.build_prompt(
                tx["text"], tx.get("fraud_score", 0.5), 0.5, similar_cases,
            )
            result = llm.generate_json(prompt, anomaly_explanation.SYSTEM_PROMPT)
            output = {
                "original": tx,
                "explanation": result,
                "condition": "with_rag",
                "retrieved_cases": similar_cases,
            }
            f.write(json.dumps(output, ensure_ascii=False) + "\n")
            print("done")

    print(f"\nResults: {out_no_rag}, {out_with_rag}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    run_rag_comparison(args.input, args.model, args.max_samples)


if __name__ == "__main__":
    main()
