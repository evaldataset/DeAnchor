"""Human Evaluation 템플릿 생성.

LLM 설명 품질을 인간 평가자가 평가할 수 있도록
50건의 평가 시트를 CSV와 HTML로 생성.
"""

import argparse
import csv
import json
import random
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "experiments" / "results"
EVAL_DIR = Path(__file__).resolve().parent.parent / "data" / "benchmarks" / "human_eval"


def generate_eval_sheet(
    predictions_path: str,
    n_samples: int = 50,
    output_dir: str | None = None,
) -> None:
    out_dir = Path(output_dir) if output_dir else EVAL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(predictions_path) as f:
        preds = [json.loads(line) for line in f]

    # FP/TP 균형 샘플링
    fp = [p for p in preds if p.get("original", {}).get("category") == "false_positive"]
    tp = [p for p in preds if p.get("original", {}).get("category") == "true_positive"]

    random.seed(42)
    n_each = n_samples // 2
    sample_fp = random.sample(fp, min(n_each, len(fp)))
    sample_tp = random.sample(tp, min(n_each, len(tp)))
    samples = sample_fp + sample_tp
    random.shuffle(samples)

    print(f"Selected {len(sample_fp)} FP + {len(sample_tp)} TP = {len(samples)} samples")

    # CSV 생성
    csv_path = out_dir / "human_eval_sheet.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Sample_ID", "Transaction_Text", "ML_Score",
            "LLM_Explanation", "LLM_Recommendation",
            "Coherence_1to5", "Completeness_1to5",
            "Clarity_1to5", "Actionability_1to5",
            "Overall_1to5", "Evaluator_Notes",
        ])

        for i, pred in enumerate(samples):
            tx = pred.get("original", {})
            exp = pred.get("fp_explanation") or pred.get("anomaly_explanation", {})

            # 설명 텍스트 정리
            if isinstance(exp, dict):
                if "assessment" in exp:
                    exp_text = exp["assessment"]
                elif "risk_summary" in exp:
                    exp_text = exp["risk_summary"]
                else:
                    exp_text = json.dumps(exp, indent=1)
            else:
                exp_text = str(exp)

            rec = exp.get("recommendation", "N/A") if isinstance(exp, dict) else "N/A"

            writer.writerow([
                f"S{i+1:03d}",
                tx.get("text", "")[:300],
                f"{pred.get('ml_score', 0):.4f}",
                exp_text[:500],
                rec,
                "", "", "", "", "", "",  # 빈 평가 칸
            ])

    print(f"CSV: {csv_path}")

    # Ground truth 정답지 (평가자에게는 비공개)
    gt_path = out_dir / "human_eval_ground_truth.csv"
    with open(gt_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Sample_ID", "Category", "Is_Fraud", "ML_Score"])
        for i, pred in enumerate(samples):
            tx = pred.get("original", {})
            writer.writerow([
                f"S{i+1:03d}",
                tx.get("category", "unknown"),
                tx.get("is_fraud", -1),
                f"{pred.get('ml_score', 0):.4f}",
            ])

    print(f"Ground truth: {gt_path}")

    # 평가 가이드라인
    guide_path = out_dir / "evaluation_guidelines.md"
    with open(guide_path, "w") as f:
        f.write("""# Human Evaluation Guidelines

## Overview
You are evaluating the quality of AI-generated fraud explanations.
For each transaction, you will see the transaction details, ML fraud score,
and an AI-generated explanation. Rate each explanation on 5 dimensions.

## Rating Scale (1-5)
- **1 = Very Poor**: Incoherent, irrelevant, or factually wrong
- **2 = Poor**: Major logical gaps or missing critical information
- **3 = Adequate**: Generally reasonable but lacks depth or specificity
- **4 = Good**: Well-reasoned, specific, and useful with minor gaps
- **5 = Excellent**: Comprehensive, precise, and immediately actionable

## Dimensions
1. **Coherence**: Is the reasoning logical and internally consistent?
2. **Completeness**: Does the explanation address all relevant transaction aspects?
3. **Clarity**: Is the explanation clear, well-structured, and professional?
4. **Actionability**: Does it provide specific, useful next steps for investigation?
5. **Overall**: Your holistic assessment of explanation quality.

## Important Notes
- You do NOT know whether the transaction is actually fraudulent.
- Judge the explanation on its own merits, not whether you agree with the conclusion.
- Consider whether a fraud analyst would find this explanation helpful.
- Write brief notes for any score below 3 or above 4.
""")

    print(f"Guidelines: {guide_path}")
    print(f"\nTotal: {len(samples)} samples ready for human evaluation")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    generate_eval_sheet(args.predictions, args.n_samples, args.output_dir)


if __name__ == "__main__":
    main()
