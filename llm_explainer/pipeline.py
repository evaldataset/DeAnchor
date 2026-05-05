"""전체 파이프라인: ML 예측 → LLM 설명 생성.

1. ML 모델의 예측 결과 로드
2. 이상 거래 텍스트 변환
3. RAG로 유사 사례 검색
4. LLM으로 5가지 태스크 실행
5. 결과 JSONL 저장
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_explainer.llm_inference import LLMConfig, LLMInference
from llm_explainer.prompts import (
    anomaly_explanation,
    fp_explanation,
    fraud_classification,
    investigation_report,
    similar_cases,
)

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "experiments" / "results"


def run_single_transaction(
    transaction_text: str,
    ml_score: float,
    threshold: float = 0.5,
    llm: LLMInference | None = None,
    rag_searcher=None,
    tasks: list[str] | None = None,
) -> dict:
    """단일 거래에 대한 전체 파이프라인 실행."""
    if tasks is None:
        tasks = ["explain", "classify", "similar", "fp_explain", "report"]

    result = {
        "transaction_text": transaction_text,
        "ml_score": ml_score,
        "threshold": threshold,
    }

    # RAG 검색
    similar_case_results = None
    if rag_searcher and ("similar" in tasks or "report" in tasks):
        try:
            similar_case_results = rag_searcher.search_for_transaction(
                transaction_text, ml_score, top_k=3,
            )
            result["similar_cases"] = similar_case_results
        except Exception as e:
            print(f"  RAG search error: {e}")

    if llm is None:
        llm = LLMInference()

    # Task 1: 이상 거래 설명
    if "explain" in tasks:
        print("  [Task 1] Anomaly Explanation...")
        prompt = anomaly_explanation.build_prompt(
            transaction_text, ml_score, threshold, similar_case_results,
        )
        response = llm.generate_json(prompt, anomaly_explanation.SYSTEM_PROMPT)
        result["anomaly_explanation"] = response

    # Task 2: 사기 유형 분류
    if "classify" in tasks:
        print("  [Task 2] Fraud Classification...")
        prompt = fraud_classification.build_prompt(
            transaction_text, ml_score, similar_case_results,
        )
        response = llm.generate_json(prompt, fraud_classification.SYSTEM_PROMPT)
        result["fraud_classification"] = response

    # Task 3: 유사 사례 비교
    if "similar" in tasks and similar_case_results:
        print("  [Task 3] Similar Case Analysis...")
        prompt = similar_cases.build_prompt(
            transaction_text, ml_score, similar_case_results,
        )
        response = llm.generate_json(prompt, similar_cases.SYSTEM_PROMPT)
        result["similar_case_analysis"] = response

    # Task 4: 오탐 설명 (ML이 사기로 판단한 경우만)
    if "fp_explain" in tasks and ml_score >= threshold:
        print("  [Task 4] False Positive Explanation...")
        prompt = fp_explanation.build_prompt(transaction_text, ml_score, threshold)
        response = llm.generate_json(prompt, fp_explanation.SYSTEM_PROMPT)
        result["fp_explanation"] = response

    # Task 5: 조사 보고서
    if "report" in tasks:
        print("  [Task 5] Investigation Report...")
        anomaly_text = json.dumps(result.get("anomaly_explanation", {}), indent=2)
        classify_text = json.dumps(result.get("fraud_classification", {}), indent=2)
        prompt = investigation_report.build_prompt(
            transaction_text, ml_score, threshold,
            anomaly_text, classify_text, similar_case_results,
        )
        response = llm.generate_json(prompt, investigation_report.SYSTEM_PROMPT)
        result["investigation_report"] = response

    return result


def _load_rag_searcher(rag_index: str | None, rag_type: str = "auto"):
    """RAG 검색 엔진 로드."""
    if not rag_index:
        # 기본 경로에서 tfidf 인덱스 시도
        tfidf_path = BASE_DIR / "rag" / "tfidf_index"
        chroma_path = BASE_DIR / "rag" / "chroma_db"
        if tfidf_path.exists():
            from rag.tfidf_search import TfidfFraudSearcher
            searcher = TfidfFraudSearcher()
            searcher.load()
            return searcher
        elif chroma_path.exists():
            from rag.search import FraudCaseSearcher
            searcher = FraudCaseSearcher()
            searcher.load()
            return searcher
        return None

    rag_path = Path(rag_index)
    if rag_type == "tfidf" or (rag_path / "vectorizer.pkl").exists():
        from rag.tfidf_search import TfidfFraudSearcher
        searcher = TfidfFraudSearcher(index_dir=rag_index)
        searcher.load()
        return searcher
    else:
        from rag.search import FraudCaseSearcher
        searcher = FraudCaseSearcher(db_dir=rag_index)
        searcher.load()
        return searcher


def run_batch(
    input_path: str,
    output_path: str | None = None,
    model_name: str = "Qwen/Qwen2.5-14B-Instruct",
    backend: str = "local",
    rag_index: str | None = None,
    tasks: list[str] | None = None,
    max_samples: int | None = None,
    threshold: float = 0.5,
) -> None:
    """배치 파이프라인 실행."""
    # 입력 데이터 로드
    print("Loading transactions...")
    transactions = []
    with open(input_path) as f:
        for line in f:
            transactions.append(json.loads(line))

    if max_samples:
        transactions = transactions[:max_samples]
    print(f"  {len(transactions)} transactions to process")

    # LLM 로드
    config = LLMConfig(model_name=model_name, backend=backend)
    llm = LLMInference(config)
    llm.load()

    # RAG 검색 엔진
    rag_searcher = _load_rag_searcher(rag_index)

    # 출력 경로
    if output_path is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = str(RESULTS_DIR / "explanations.jsonl")

    # 실행
    print(f"\nProcessing {len(transactions)} transactions...")
    start_time = time.time()

    with open(output_path, "w") as f:
        for i, tx in enumerate(transactions):
            print(f"\n[{i + 1}/{len(transactions)}] Score={tx.get('fraud_score', 'N/A')}, "
                  f"Fraud={tx.get('is_fraud', 'N/A')}")

            result = run_single_transaction(
                transaction_text=tx["text"],
                ml_score=tx.get("fraud_score", 0.5),
                threshold=threshold,
                llm=llm,
                rag_searcher=rag_searcher,
                tasks=tasks,
            )
            result["original"] = tx
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    elapsed = time.time() - start_time
    print(f"\nDone! {len(transactions)} transactions in {elapsed:.0f}s")
    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="ML → LLM fraud explanation pipeline")
    parser.add_argument("--transaction", type=str, default=None, help="Single transaction text")
    parser.add_argument("--ml_score", type=float, default=0.9, help="ML fraud score")
    parser.add_argument("--input", type=str, default=None, help="Input JSONL file")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL file")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--backend", type=str, default="local",
                        choices=["local", "anthropic", "openai"],
                        help="LLM backend: local (HF), anthropic (Claude), openai (GPT)")
    parser.add_argument("--rag_index", type=str, default=None, help="RAG index path")
    parser.add_argument("--tasks", type=str, nargs="+",
                        default=["explain", "classify", "fp_explain", "report"],
                        choices=["explain", "classify", "similar", "fp_explain", "report"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if args.transaction:
        # 단건 테스트
        config = LLMConfig(model_name=args.model, backend=args.backend)
        llm = LLMInference(config)
        llm.load()

        rag_searcher = _load_rag_searcher(args.rag_index)

        result = run_single_transaction(
            args.transaction, args.ml_score, args.threshold,
            llm=llm, rag_searcher=rag_searcher, tasks=args.tasks,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.input:
        run_batch(
            args.input, args.output, args.model, args.backend, args.rag_index,
            args.tasks, args.max_samples, args.threshold,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
