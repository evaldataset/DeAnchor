"""실험 자동 실행 스크립트.

실험 설정 YAML에 따라 조건별 파이프라인 실행 + 평가.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_explainer.llm_inference import LLMConfig, LLMInference
from llm_explainer.pipeline import _load_rag_searcher, run_single_transaction

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def run_experiment(config_path: str, max_samples: int | None = None) -> None:
    """실험 설정에 따라 조건별 파이프라인 실행."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    exp_name = config["experiment"]["name"]
    print(f"=== Running: {exp_name} ===")
    print(f"  {config['experiment']['description']}")

    # 입력 데이터 로드
    input_path = BASE_DIR / config["data"]["input"]
    transactions = []
    with open(input_path) as f:
        for line in f:
            transactions.append(json.loads(line))

    if max_samples:
        transactions = transactions[:max_samples]
    print(f"\n  Input: {len(transactions)} transactions")

    # 모델 설정
    model_cfg = config["models"]["llm"]
    llm_config = LLMConfig(
        model_name=model_cfg["name"],
        backend=model_cfg["backend"],
    )
    llm = LLMInference(llm_config)
    llm.load()

    threshold = config.get("ml_baseline", {}).get("threshold", 0.5)

    # 조건별 실행
    for condition in config["conditions"]:
        cond_name = condition["name"]
        print(f"\n--- Condition: {cond_name} ---")
        print(f"  {condition['description']}")

        if not condition.get("llm", True):
            print("  [SKIP] No LLM in this condition")
            continue

        tasks = condition.get("tasks", ["explain", "classify", "fp_explain"])
        use_rag = condition.get("rag", False)

        rag_searcher = None
        if use_rag:
            rag_index = condition.get("rag_index")
            rag_searcher = _load_rag_searcher(rag_index)

        output_path = RESULTS_DIR / f"{exp_name}_{cond_name}.jsonl"
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for i, tx in enumerate(transactions):
                print(f"  [{i + 1}/{len(transactions)}]", end=" ", flush=True)

                result = run_single_transaction(
                    transaction_text=tx["text"],
                    ml_score=tx.get("fraud_score", 0.5),
                    threshold=threshold,
                    llm=llm,
                    rag_searcher=rag_searcher,
                    tasks=tasks,
                )
                result["original"] = tx
                result["condition"] = cond_name
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"\n  Saved: {output_path}")

    print(f"\n=== Experiment {exp_name} complete ===")


def main():
    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument("--config", type=str, required=True, help="Experiment config YAML")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    run_experiment(args.config, args.max_samples)


if __name__ == "__main__":
    main()
