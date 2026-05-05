"""오탐(False Positive) 감소 효과 측정.

ML 모델의 FP 예측에 대해 LLM이 "정상"으로 재분류한 비율 측정.
ML+LLM 결합 시 최종 Precision/Recall 변화 분석.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "experiments" / "results"


def analyze_fp_reduction(
    ml_predictions_path: str,
    llm_explanations_path: str,
    threshold: float = 0.5,
) -> dict:
    """FP 감소 효과 분석."""
    # ML 예측 로드
    ml_preds = pd.read_csv(ml_predictions_path)
    print(f"ML predictions: {len(ml_preds):,} transactions")

    # ML 성능 기준선
    y_true = ml_preds["true_label"].values
    y_ml_pred = ml_preds["predicted"].values

    ml_cm = confusion_matrix(y_true, y_ml_pred)
    ml_tn, ml_fp, ml_fn, ml_tp = ml_cm.ravel()

    print(f"\n=== ML Baseline ===")
    print(f"  TP={ml_tp}, FP={ml_fp}, FN={ml_fn}, TN={ml_tn}")
    print(f"  Precision: {ml_tp / (ml_tp + ml_fp):.4f}")
    print(f"  Recall: {ml_tp / (ml_tp + ml_fn):.4f}")

    # LLM 설명 결과 로드
    llm_results = []
    with open(llm_explanations_path) as f:
        for line in f:
            llm_results.append(json.loads(line))

    print(f"\nLLM explanations: {len(llm_results)} transactions")

    # LLM의 재평가 분석 (v2 balanced prompt 지원)
    fp_corrections = 0
    tp_preserved = 0
    llm_errors = 0

    for result in llm_results:
        original = result.get("original", {})
        true_label = original.get("is_fraud", -1)
        fp_exp = result.get("fp_explanation", {})

        if fp_exp.get("parse_error"):
            continue

        recommendation = fp_exp.get("recommendation", "").upper()
        # v2 프롬프트: fraud_likelihood 사용, v1: confidence_legitimate 사용
        fraud_likelihood = fp_exp.get("fraud_likelihood")
        confidence_legitimate = fp_exp.get("confidence_legitimate", 0)

        # LLM이 "정상"으로 판단하는 조건
        if fraud_likelihood is not None:
            # v2: fraud_likelihood < 0.4이면 정상으로 판단
            is_release = recommendation == "RELEASE" or fraud_likelihood < 0.4
        else:
            # v1: confidence_legitimate > 0.7이면 정상으로 판단
            is_release = recommendation == "RELEASE" or confidence_legitimate > 0.7

        if true_label == 0:  # 실제 정상 거래
            if is_release:
                fp_corrections += 1  # LLM이 정상으로 올바르게 판단
        elif true_label == 1:  # 실제 사기 거래
            if is_release:
                llm_errors += 1  # LLM이 사기를 정상으로 잘못 판단
            else:
                tp_preserved += 1

    print(f"\n=== LLM FP Analysis ===")
    print(f"  FP correctly identified as normal: {fp_corrections}")
    print(f"  TP preserved (fraud kept): {tp_preserved}")
    print(f"  LLM errors (fraud → normal): {llm_errors}")

    # ML+LLM 결합 성능
    combined_fp = ml_fp - fp_corrections
    combined_fn = ml_fn + llm_errors
    combined_tp = ml_tp - llm_errors
    combined_tn = ml_tn + fp_corrections

    if (combined_tp + combined_fp) > 0:
        combined_precision = combined_tp / (combined_tp + combined_fp)
    else:
        combined_precision = 0
    if (combined_tp + combined_fn) > 0:
        combined_recall = combined_tp / (combined_tp + combined_fn)
    else:
        combined_recall = 0

    print(f"\n=== ML + LLM Combined ===")
    print(f"  TP={combined_tp}, FP={combined_fp}, FN={combined_fn}, TN={combined_tn}")
    print(f"  Precision: {combined_precision:.4f} (was {ml_tp / (ml_tp + ml_fp):.4f})")
    print(f"  Recall: {combined_recall:.4f} (was {ml_tp / (ml_tp + ml_fn):.4f})")
    print(f"  FP reduction: {fp_corrections}/{ml_fp} = {fp_corrections / ml_fp:.1%}" if ml_fp > 0 else "")

    results = {
        "ml_baseline": {
            "tp": int(ml_tp), "fp": int(ml_fp), "fn": int(ml_fn), "tn": int(ml_tn),
            "precision": float(ml_tp / (ml_tp + ml_fp)) if (ml_tp + ml_fp) > 0 else 0,
            "recall": float(ml_tp / (ml_tp + ml_fn)) if (ml_tp + ml_fn) > 0 else 0,
        },
        "llm_analysis": {
            "fp_corrections": fp_corrections,
            "tp_preserved": tp_preserved,
            "llm_errors": llm_errors,
            "total_analyzed": len(llm_results),
        },
        "combined": {
            "tp": int(combined_tp), "fp": int(combined_fp),
            "fn": int(combined_fn), "tn": int(combined_tn),
            "precision": float(combined_precision),
            "recall": float(combined_recall),
            "fp_reduction_rate": float(fp_corrections / ml_fp) if ml_fp > 0 else 0,
        },
    }

    output_path = RESULTS_DIR / "fp_reduction_results.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Measure FP reduction effect")
    parser.add_argument("--ml_predictions", type=str, required=True)
    parser.add_argument("--llm_explanations", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    analyze_fp_reduction(args.ml_predictions, args.llm_explanations, args.threshold)


if __name__ == "__main__":
    main()
