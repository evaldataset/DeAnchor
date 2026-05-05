"""ML 기준선 모델 평가 + 비교.

XGBoost, LightGBM 결과를 비교하고 시각화.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "paper" / "figures"


def load_predictions(dataset: str) -> dict[str, pd.DataFrame]:
    """모델별 예측 결과 로드."""
    preds = {}
    for model in ["xgb", "lgb"]:
        prefix = "" if model == "xgb" else f"{model}_"
        # XGBoost: predictions_{dataset}.csv, LightGBM: predictions_lgb_{dataset}.csv
        if model == "xgb":
            path = RESULTS_DIR / f"predictions_{dataset}.csv"
        else:
            path = RESULTS_DIR / f"predictions_{model}_{dataset}.csv"

        if path.exists():
            preds[model] = pd.read_csv(path)
            print(f"Loaded {model}: {len(preds[model]):,} predictions")
    return preds


def compare_models(preds: dict[str, pd.DataFrame], dataset: str) -> None:
    """모델 성능 비교."""
    print(f"\n{'='*60}")
    print(f"Model Comparison: {dataset}")
    print(f"{'='*60}")

    results = []
    for model_name, df in preds.items():
        y_true = df["true_label"].values
        y_score = df["fraud_score"].values

        roc_auc = roc_auc_score(y_true, y_score)
        pr_auc = average_precision_score(y_true, y_score)

        # 최적 threshold의 F1
        y_pred = df["predicted"].values
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results.append({
            "Model": model_name.upper(),
            "ROC-AUC": roc_auc,
            "PR-AUC": pr_auc,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "FP Count": int(fp),
        })

        print(f"\n{model_name.upper()}:")
        print(f"  ROC-AUC:   {roc_auc:.4f}")
        print(f"  PR-AUC:    {pr_auc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1:        {f1:.4f}")
        print(f"  FP Count:  {fp:,}")

    # 비교 테이블 저장
    results_df = pd.DataFrame(results)
    table_path = Path(__file__).resolve().parent.parent / "paper" / "tables"
    table_path.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(table_path / f"baseline_comparison_{dataset}.csv", index=False)
    print(f"\nComparison table saved to {table_path / f'baseline_comparison_{dataset}.csv'}")

    return preds


def plot_curves(preds: dict[str, pd.DataFrame], dataset: str) -> None:
    """ROC, PR curve 시각화."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = {"xgb": "#2196F3", "lgb": "#4CAF50"}

    for model_name, df in preds.items():
        y_true = df["true_label"].values
        y_score = df["fraud_score"].values
        color = colors.get(model_name, "#999")

        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        ax1.plot(fpr, tpr, color=color, label=f"{model_name.upper()} (AUC={auc:.4f})")

        # PR
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        ax2.plot(rec, prec, color=color, label=f"{model_name.upper()} (AP={ap:.4f})")

    ax1.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    fraud_rate = preds[list(preds.keys())[0]]["true_label"].mean()
    ax2.axhline(y=fraud_rate, color="k", linestyle="--", alpha=0.3, label=f"Baseline ({fraud_rate:.4f})")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Fraud Detection Baseline - {dataset}", fontsize=14)
    plt.tight_layout()

    fig_path = FIGURES_DIR / f"baseline_curves_{dataset}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nCurves saved to {fig_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate and compare baseline models")
    parser.add_argument("--dataset", type=str, default="credit_card",
                        choices=["ieee_cis", "credit_card", "paysim"])
    args = parser.parse_args()

    preds = load_predictions(args.dataset)
    if not preds:
        print("No prediction files found. Train models first.")
        return

    compare_models(preds, args.dataset)
    plot_curves(preds, args.dataset)


if __name__ == "__main__":
    main()
