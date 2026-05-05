"""사기 유형 분류 정확도 평가.

LLM의 사기 유형 분류 결과를 Ground Truth와 비교.
F1-score (유형별, macro) 측정.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent.parent / "experiments" / "results"

FRAUD_TYPES = [
    "Structuring",
    "Layering",
    "Identity Fraud",
    "Insider Trading",
    "Pump & Dump",
    "Account Takeover",
    "Legitimate",
]


def evaluate_classification(
    predictions_path: str,
    ground_truth_path: str | None = None,
) -> dict:
    """사기 유형 분류 정확도 평가."""
    with open(predictions_path) as f:
        predictions = [json.loads(line) for line in f]

    print(f"Evaluating {len(predictions)} classification results...")

    # Ground truth가 있으면 비교
    gt_labels = {}
    if ground_truth_path:
        with open(ground_truth_path) as f:
            for item in json.load(f):
                gt_labels[item["id"]] = item["fraud_type"]

    # 예측 분포 분석
    pred_types = []
    for pred in predictions:
        classification = pred.get("fraud_classification", {})
        if classification.get("parse_error"):
            continue
        pred_type = classification.get("primary_classification", "Unknown")
        pred_types.append(pred_type)

    type_dist = Counter(pred_types)
    print(f"\n=== Predicted Type Distribution ===")
    for fraud_type, count in type_dist.most_common():
        print(f"  {fraud_type:20s}: {count:4d} ({count / len(pred_types):.1%})")

    # 신뢰도 분포
    confidences = []
    for pred in predictions:
        classification = pred.get("fraud_classification", {})
        conf = classification.get("confidence", 0)
        if isinstance(conf, (int, float)):
            confidences.append(conf)

    if confidences:
        print(f"\n=== Confidence Distribution ===")
        print(f"  Mean: {np.mean(confidences):.3f}")
        print(f"  Median: {np.median(confidences):.3f}")
        print(f"  Min: {np.min(confidences):.3f}, Max: {np.max(confidences):.3f}")

    results = {
        "total_predictions": len(predictions),
        "valid_predictions": len(pred_types),
        "type_distribution": dict(type_dist),
        "confidence_stats": {
            "mean": float(np.mean(confidences)) if confidences else 0,
            "median": float(np.median(confidences)) if confidences else 0,
        },
    }

    # Ground truth 비교 (있는 경우)
    if gt_labels:
        correct = 0
        total = 0
        per_type = {t: {"tp": 0, "fp": 0, "fn": 0} for t in FRAUD_TYPES}

        for pred in predictions:
            classification = pred.get("fraud_classification", {})
            pred_type = classification.get("primary_classification")
            tx_id = pred.get("original", {}).get("transaction_id")

            if not pred_type or tx_id not in gt_labels:
                continue

            gt_type = gt_labels[tx_id]
            total += 1

            if pred_type == gt_type:
                correct += 1
                if gt_type in per_type:
                    per_type[gt_type]["tp"] += 1
            else:
                if pred_type in per_type:
                    per_type[pred_type]["fp"] += 1
                if gt_type in per_type:
                    per_type[gt_type]["fn"] += 1

        accuracy = correct / total if total > 0 else 0
        print(f"\n=== Accuracy (vs Ground Truth) ===")
        print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")

        # Per-type F1
        f1_scores = []
        for fraud_type, counts in per_type.items():
            tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
            if tp + fp + fn > 0:
                print(f"  {fraud_type:20s}: P={precision:.3f} R={recall:.3f} F1={f1:.3f}")

        macro_f1 = np.mean([f for f in f1_scores if f > 0]) if f1_scores else 0
        print(f"\n  Macro F1: {macro_f1:.4f}")

        results["accuracy"] = accuracy
        results["macro_f1"] = float(macro_f1)

    output_path = RESULTS_DIR / "classification_results.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate fraud type classification")
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--ground_truth", type=str, default=None)
    args = parser.parse_args()

    evaluate_classification(args.predictions, args.ground_truth)


if __name__ == "__main__":
    main()
