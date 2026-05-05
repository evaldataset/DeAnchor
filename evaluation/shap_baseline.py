"""SHAP baseline: Feature attribution + 템플릿 기반 설명 생성.

LLM 설명 vs SHAP+템플릿 설명을 비교하기 위한 baseline.
SHAP values → 자연어 템플릿으로 변환 → LLM-as-Judge로 평가.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "experiments" / "results"


def compute_shap_explanations(dataset: str, n_samples: int = 50) -> list[dict]:
    """SHAP values 계산 + 템플릿 기반 설명 생성."""
    import shap

    # 모델 로드
    model_path = BASE_DIR / "ml_baseline" / "models" / f"xgb_{dataset}.json"
    model = xgb.Booster()
    model.load_model(str(model_path))

    # 데이터 로드
    data_path = BASE_DIR / "data" / "processed" / f"{dataset}.parquet"
    df = pd.read_parquet(data_path)
    y = df["isFraud"].values
    drop_cols = ["isFraud", "TransactionID"]
    X_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # 결측치 처리 (SHAP 계산용)
    cat_cols = X_df.select_dtypes(include=["object", "string"]).columns
    for col in cat_cols:
        X_df[col] = X_df[col].fillna("missing").astype("category").cat.codes
    X_df = X_df.fillna(X_df.median(numeric_only=True))

    feature_names = list(X_df.columns)

    # 예측 로드
    pred_path = BASE_DIR / "ml_baseline" / "results" / f"predictions_{dataset}.csv"
    preds = pd.read_csv(pred_path)

    # FP + TP 샘플 선택
    import random
    random.seed(42)
    fp_idx = preds[(preds["predicted"] == 1) & (preds["true_label"] == 0)].index.tolist()
    tp_idx = preds[(preds["predicted"] == 1) & (preds["true_label"] == 1)].index.tolist()

    n_each = n_samples // 2
    sample_fp = random.sample(fp_idx, min(n_each, len(fp_idx)))
    sample_tp = random.sample(tp_idx, min(n_each, len(tp_idx)))
    sample_idx = sample_fp + sample_tp
    random.shuffle(sample_idx)

    X_sample = X_df.iloc[sample_idx].values.astype(np.float64)
    y_sample = y[sample_idx]
    scores_sample = preds.loc[sample_idx, "fraud_score"].values

    # SHAP 계산
    print(f"Computing SHAP values for {len(sample_idx)} samples...")
    dmatrix = xgb.DMatrix(X_sample, feature_names=feature_names)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(dmatrix)

    # 템플릿 기반 설명 생성
    results = []
    for i in range(len(sample_idx)):
        sv = shap_values[i]
        top_k = 5
        top_idx_pos = np.argsort(sv)[-top_k:][::-1]  # fraud 방향
        top_idx_neg = np.argsort(sv)[:top_k]  # normal 방향

        # 템플릿 설명 생성
        fraud_factors = []
        normal_factors = []
        for j in top_idx_pos:
            if sv[j] > 0:
                fraud_factors.append(
                    f"{feature_names[j]} = {X_sample[i, j]:.2f} "
                    f"(SHAP: +{sv[j]:.3f}, increases fraud risk)"
                )
        for j in top_idx_neg:
            if sv[j] < 0:
                normal_factors.append(
                    f"{feature_names[j]} = {X_sample[i, j]:.2f} "
                    f"(SHAP: {sv[j]:.3f}, decreases fraud risk)"
                )

        explanation = f"ML Fraud Score: {scores_sample[i]:.4f}\n\n"
        explanation += "FACTORS INCREASING FRAUD RISK:\n"
        for f in fraud_factors[:3]:
            explanation += f"  - {f}\n"
        explanation += "\nFACTORS DECREASING FRAUD RISK:\n"
        for f in normal_factors[:3]:
            explanation += f"  - {f}\n"
        explanation += f"\nOverall: {len(fraud_factors)} features increase risk, "
        explanation += f"{len(normal_factors)} features decrease risk."

        # 거래 텍스트 로드
        text_path = BASE_DIR / "data" / "processed" / "transactions_text" / f"{dataset}_text.jsonl"
        tx_text = ""
        with open(text_path) as f:
            for line_idx, line in enumerate(f):
                if line_idx == sample_idx[i]:
                    tx_text = json.loads(line).get("text", "")
                    break

        results.append({
            "sample_idx": int(sample_idx[i]),
            "is_fraud": int(y_sample[i]),
            "category": "false_positive" if y_sample[i] == 0 else "true_positive",
            "fraud_score": float(scores_sample[i]),
            "transaction_text": tx_text,
            "shap_explanation": explanation,
            "top_features": {feature_names[j]: float(sv[j]) for j in top_idx_pos[:5]},
        })

    print(f"Generated {len(results)} SHAP explanations")
    return results


def compare_with_llm(
    shap_results: list[dict],
    llm_results_path: str,
    model: str = "gpt-4o-mini",
) -> dict:
    """SHAP 설명 vs LLM 설명을 LLM-as-Judge로 비교."""
    from llm_explainer.llm_inference import LLMConfig, LLMInference

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {"error": "OPENAI_API_KEY not set"}

    config = LLMConfig(model_name=model, backend="openai", temperature=0.0)
    llm = LLMInference(config)
    llm.load()

    # LLM 결과 로드 (매칭용)
    llm_explanations = {}
    with open(llm_results_path) as f:
        for line in f:
            d = json.loads(line)
            score = d.get("ml_score", 0)
            exp = d.get("fp_explanation") or d.get("anomaly_explanation", {})
            if isinstance(exp, dict):
                exp_text = json.dumps(exp, indent=1)[:600]
            else:
                exp_text = str(exp)[:600]
            llm_explanations[round(score, 4)] = exp_text

    COMPARE_PROMPT = """You are a fraud analyst comparing two explanations for the same suspicious transaction. Which explanation would be MORE USEFUL in your investigation?

## Transaction
{transaction_text}

## Explanation A (Feature Attribution)
{shap_explanation}

## Explanation B (AI Narrative)
{llm_explanation}

Rate EACH explanation 1-5, then pick the better one.

```json
{{
    "explanation_a_score": <1-5>,
    "explanation_b_score": <1-5>,
    "preferred": "A" or "B",
    "reason": "1 sentence why"
}}
```"""

    comparisons = []
    for i, sr in enumerate(shap_results[:30]):
        # 가장 가까운 score의 LLM 설명 매칭
        best_match = min(llm_explanations.keys(),
                         key=lambda k: abs(k - round(sr["fraud_score"], 4)),
                         default=None)
        llm_exp = llm_explanations.get(best_match, "No LLM explanation available")

        prompt = COMPARE_PROMPT.format(
            transaction_text=sr["transaction_text"][:300],
            shap_explanation=sr["shap_explanation"],
            llm_explanation=llm_exp,
        )

        result = llm.generate_json(prompt, "You are a senior fraud analyst evaluating explanation quality.")
        result["category"] = sr["category"]
        comparisons.append(result)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/30]")

    # 집계
    a_scores = [c.get("explanation_a_score", 0) for c in comparisons if isinstance(c.get("explanation_a_score"), (int, float))]
    b_scores = [c.get("explanation_b_score", 0) for c in comparisons if isinstance(c.get("explanation_b_score"), (int, float))]
    preferences = [c.get("preferred", "?") for c in comparisons]

    from collections import Counter
    pref_counts = Counter(preferences)

    summary = {
        "shap_mean_score": round(np.mean(a_scores), 2) if a_scores else 0,
        "llm_mean_score": round(np.mean(b_scores), 2) if b_scores else 0,
        "preference": dict(pref_counts),
        "n_comparisons": len(comparisons),
    }

    print(f"\n=== SHAP vs LLM Comparison ===")
    print(f"  SHAP (A) mean: {summary['shap_mean_score']}/5")
    print(f"  LLM  (B) mean: {summary['llm_mean_score']}/5")
    print(f"  Preferred: {dict(pref_counts)}")

    return {"summary": summary, "comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ieee_cis")
    parser.add_argument("--llm_results", default=None)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    shap_results = compute_shap_explanations(args.dataset, args.n_samples)

    # SHAP 설명 저장
    out_dir = RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    shap_path = out_dir / f"shap_explanations_{args.dataset}.json"
    with open(shap_path, "w") as f:
        json.dump(shap_results, f, indent=2, ensure_ascii=False)
    print(f"SHAP explanations saved to {shap_path}")

    # LLM 비교
    if args.llm_results:
        comparison = compare_with_llm(shap_results, args.llm_results)
        comp_path = out_dir / f"shap_vs_llm_{args.dataset}.json"
        with open(comp_path, "w") as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        print(f"Comparison saved to {comp_path}")


if __name__ == "__main__":
    main()
