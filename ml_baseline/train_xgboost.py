"""XGBoost 기준선 사기 탐지 모델 학습.

클래스 불균형 처리 (scale_pos_weight) + 5-fold CV.
학습 후 모델, 예측 결과, False Positive 사례를 저장.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parent / "models"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_data(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """전처리된 데이터 로드. 인코딩/imputation 전 상태로 반환."""
    path = DATA_DIR / f"{dataset}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run feature_engineering.py first."
        )

    df = pd.read_parquet(path)
    y = df["isFraud"].values
    drop_cols = ["isFraud", "TransactionID"]
    X_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    print(f"Data: {X_df.shape[0]:,} samples, {X_df.shape[1]} features")
    print(f"Fraud rate: {y.mean():.4%} ({y.sum():,} / {len(y):,})")

    return df, X_df, y


def preprocess_fold(X_train_df: pd.DataFrame, X_val_df: pd.DataFrame):
    """Fold별 imputation + encoding (data leakage 방지).

    Train fold에서 fit → val fold에 transform.
    """
    # 카테고리 인코딩: train fit → val transform
    cat_cols = X_train_df.select_dtypes(include=["object", "string"]).columns.tolist()
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train_df = X_train_df.copy()
        X_val_df = X_val_df.copy()
        X_train_df[cat_cols] = enc.fit_transform(X_train_df[cat_cols])
        X_val_df[cat_cols] = enc.transform(X_val_df[cat_cols])

    # 결측치 imputation: train median fit → val transform
    imp = SimpleImputer(strategy="median")
    X_train = imp.fit_transform(X_train_df.values.astype(np.float64))
    X_val = imp.transform(X_val_df.values.astype(np.float64))

    return X_train, X_val


def train_and_evaluate(
    X_df: pd.DataFrame,
    y: np.ndarray,
    n_folds: int = 5,
    params: dict | None = None,
) -> tuple[np.ndarray, dict, list]:
    """Stratified K-Fold CV로 학습 + OOF 예측 (fold별 전처리)."""
    if params is None:
        neg, pos = np.bincount(y.astype(int))
        params = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": neg / pos,
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": 42,
        }

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))
    models = []
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_df, y)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")
        y_train, y_val = y[train_idx], y[val_idx]

        # Fold별 전처리: train fit → val transform (CODE-001 fix)
        X_train, X_val = preprocess_fold(
            X_df.iloc[train_idx], X_df.iloc[val_idx]
        )

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=100,
        )
        models.append(model)

        val_pred = model.predict(dval)
        oof_preds[val_idx] = val_pred

        auc = roc_auc_score(y_val, val_pred)
        ap = average_precision_score(y_val, val_pred)
        print(f"  ROC-AUC: {auc:.4f}, PR-AUC: {ap:.4f}")
        fold_metrics.append({"fold": fold + 1, "roc_auc": auc, "pr_auc": ap})

    # 전체 OOF 성능
    overall_auc = roc_auc_score(y, oof_preds)
    overall_ap = average_precision_score(y, oof_preds)
    print(f"\n=== Overall OOF Results ===")
    print(f"ROC-AUC: {overall_auc:.4f}")
    print(f"PR-AUC:  {overall_ap:.4f}")

    # 최적 threshold (F1 기준, OOF 전체에서 선택)
    # NOTE: This threshold is used ONLY for FP/TP pool construction (selecting
    # which transactions to send to the LLM). All downstream analyses use the
    # continuous fraud_score, not binary predictions. The threshold is not tuned
    # on a held-out test set — it is an operating-point heuristic for sample
    # selection, not a final evaluation metric.
    precisions, recalls, thresholds = precision_recall_curve(y, oof_preds)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]

    y_pred = (oof_preds >= best_threshold).astype(int)
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\nBest threshold: {best_threshold:.4f} (F1: {best_f1:.4f})")
    print(f"Confusion Matrix:")
    print(f"  TN={tn:,}  FP={fp:,}")
    print(f"  FN={fn:,}  TP={tp:,}")
    print(f"  FP rate: {fp / (fp + tn):.4%}")
    print(f"\n{classification_report(y, y_pred, target_names=['Normal', 'Fraud'])}")

    metrics = {
        "overall_roc_auc": overall_auc,
        "overall_pr_auc": overall_ap,
        "best_threshold": float(best_threshold),
        "best_f1": float(best_f1),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "fp_rate": float(fp / (fp + tn)),
        "fold_metrics": fold_metrics,
    }

    return oof_preds, metrics, models


def save_results(
    df: pd.DataFrame,
    oof_preds: np.ndarray,
    metrics: dict,
    models: list,
    dataset: str,
    output: str | None,
) -> None:
    """모델, 예측 결과, FP 사례 저장."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 모델 저장 (best fold + all folds for reproducibility)
    model_path = Path(output) if output else MODEL_DIR / f"xgb_{dataset}.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    best_fold = max(range(len(models)), key=lambda i: metrics["fold_metrics"][i]["pr_auc"])
    models[best_fold].save_model(str(model_path))
    print(f"\nBest model (fold {best_fold + 1}) saved to {model_path}")

    # Save all fold models for full reproducibility
    # NOTE: Each fold model was trained on a different 4/5 split with its own
    # fold-internal imputer/encoder. The saved best-fold model can reproduce
    # predictions only for its own validation fold. For full OOF reproduction,
    # all fold models + their preprocessors would need to be saved. We save
    # the fold provenance here; preprocessor saving is left to future work.
    fold_dir = model_path.parent / f"xgb_{dataset}_all_folds"
    fold_dir.mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(models):
        m.save_model(str(fold_dir / f"fold_{i}.json"))
    fold_meta = {
        "best_fold": best_fold,
        "n_folds": len(models),
        "fold_metrics": metrics["fold_metrics"],
        "threshold": metrics["best_threshold"],
        "threshold_note": "Tuned on all OOF predictions for FP/TP pool selection only",
    }
    with open(fold_dir / "fold_meta.json", "w") as f:
        import json
        json.dump(fold_meta, f, indent=2)
    print(f"All {len(models)} fold models saved to {fold_dir}/")

    # 예측 결과 저장
    pred_df = pd.DataFrame({
        "index": range(len(oof_preds)),
        "true_label": df["isFraud"].values,
        "fraud_score": oof_preds,
        "predicted": (oof_preds >= metrics["best_threshold"]).astype(int),
    })
    pred_path = RESULTS_DIR / f"predictions_{dataset}.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"Predictions saved to {pred_path}")

    # False Positive 사례 추출 (LLM 설명 시스템 입력용)
    threshold = metrics["best_threshold"]
    fp_mask = (oof_preds >= threshold) & (df["isFraud"].values == 0)
    fp_df = df[fp_mask].copy()
    fp_df["fraud_score"] = oof_preds[fp_mask]
    fp_df = fp_df.sort_values("fraud_score", ascending=False)

    fp_path = RESULTS_DIR / f"false_positives_{dataset}.csv"
    fp_df.to_csv(fp_path, index=False)
    print(f"False Positives saved to {fp_path} ({len(fp_df):,} cases)")

    # 메트릭 저장
    metrics["dataset"] = dataset
    metrics["timestamp"] = datetime.now().isoformat()
    metrics["false_positive_count"] = int(fp_mask.sum())
    metrics_path = RESULTS_DIR / f"metrics_{dataset}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost fraud detection baseline")
    parser.add_argument(
        "--dataset",
        type=str,
        default="credit_card",
        choices=["ieee_cis", "credit_card", "paysim"],
        help="Dataset to use",
    )
    parser.add_argument("--output", type=str, default=None, help="Model output path")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    args = parser.parse_args()

    df, X_df, y = load_data(args.dataset)
    oof_preds, metrics, models = train_and_evaluate(X_df, y, n_folds=args.folds)
    save_results(df, oof_preds, metrics, models, args.dataset, args.output)


if __name__ == "__main__":
    main()
