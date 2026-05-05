"""ML 예측 + 거래 텍스트 결합 → LLM 파이프라인 입력 생성.

XGBoost의 fraud_score를 거래 텍스트에 결합하여
LLM 파이프라인 입력용 JSONL 파일 생성.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
ML_RESULTS_DIR = BASE_DIR / "ml_baseline" / "results"
OUTPUT_DIR = BASE_DIR / "experiments" / "results"


def prepare_input(
    dataset: str = "credit_card",
    model: str = "xgb",
    include_fp: bool = True,
    include_tp: bool = True,
    include_fn: bool = True,
    max_normal: int = 100,
) -> None:
    """LLM 파이프라인 입력 준비."""
    # ML 예측 로드
    if model == "xgb":
        pred_path = ML_RESULTS_DIR / f"predictions_{dataset}.csv"
    else:
        pred_path = ML_RESULTS_DIR / f"predictions_{model}_{dataset}.csv"

    preds = pd.read_csv(pred_path)
    print(f"ML predictions: {len(preds):,}")

    # 거래 텍스트 로드
    text_path = DATA_DIR / "transactions_text" / f"{dataset}_text.jsonl"
    texts = {}
    with open(text_path) as f:
        for i, line in enumerate(f):
            tx = json.loads(line)
            texts[i] = tx

    print(f"Transaction texts: {len(texts):,}")

    # CODE-016: 인덱스 정합성 검증
    if len(texts) != len(preds):
        print(f"  WARNING: Text ({len(texts):,}) != Predictions ({len(preds):,}) count mismatch!")
        print(f"  Run converter on full dataset without --max_samples to fix.")

    # 결합
    output_records = []

    # FP 사례 (ML=사기, 실제=정상) — 핵심 연구 대상
    if include_fp:
        fp_mask = (preds["predicted"] == 1) & (preds["true_label"] == 0)
        fp_indices = preds[fp_mask].index.tolist()
        for idx in fp_indices:
            if idx in texts:
                record = texts[idx].copy()
                record["fraud_score"] = float(preds.loc[idx, "fraud_score"])
                record["ml_predicted"] = 1
                record["category"] = "false_positive"
                output_records.append(record)
        print(f"  FP cases: {len(fp_indices)}")

    # TP 사례 (ML=사기, 실제=사기)
    if include_tp:
        tp_mask = (preds["predicted"] == 1) & (preds["true_label"] == 1)
        tp_indices = preds[tp_mask].index.tolist()
        for idx in tp_indices:
            if idx in texts:
                record = texts[idx].copy()
                record["fraud_score"] = float(preds.loc[idx, "fraud_score"])
                record["ml_predicted"] = 1
                record["category"] = "true_positive"
                output_records.append(record)
        print(f"  TP cases: {len(tp_indices)}")

    # FN 사례 (ML=정상, 실제=사기) — 놓친 사기
    if include_fn:
        fn_mask = (preds["predicted"] == 0) & (preds["true_label"] == 1)
        fn_indices = preds[fn_mask].index.tolist()
        for idx in fn_indices:
            if idx in texts:
                record = texts[idx].copy()
                record["fraud_score"] = float(preds.loc[idx, "fraud_score"])
                record["ml_predicted"] = 0
                record["category"] = "false_negative"
                output_records.append(record)
        print(f"  FN cases: {len(fn_indices)}")

    # 정상 거래 샘플 (비교용)
    if max_normal > 0:
        tn_mask = (preds["predicted"] == 0) & (preds["true_label"] == 0)
        tn_sample = preds[tn_mask].sample(n=min(max_normal, tn_mask.sum()), random_state=42)
        for idx in tn_sample.index:
            if idx in texts:
                record = texts[idx].copy()
                record["fraud_score"] = float(preds.loc[idx, "fraud_score"])
                record["ml_predicted"] = 0
                record["category"] = "true_negative"
                output_records.append(record)
        print(f"  TN sample: {len(tn_sample)}")

    # 저장
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"llm_input_{dataset}.jsonl"
    with open(output_path, "w") as f:
        for record in output_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nTotal: {len(output_records)} transactions")
    print(f"Saved to {output_path}")

    # 카테고리별 통계
    from collections import Counter
    cats = Counter(r["category"] for r in output_records)
    for cat, count in cats.most_common():
        print(f"  {cat}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Prepare LLM pipeline input")
    parser.add_argument("--dataset", default="credit_card")
    parser.add_argument("--model", default="xgb", choices=["xgb", "lgb"])
    parser.add_argument("--max_normal", type=int, default=100)
    args = parser.parse_args()

    prepare_input(args.dataset, args.model, max_normal=args.max_normal)


if __name__ == "__main__":
    main()
