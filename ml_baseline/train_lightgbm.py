"""LightGBM 기준선 사기 탐지 모델 학습.

XGBoost와 동일한 평가 프레임워크 사용.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parent / "models"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_data(dataset: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    path = DATA_DIR / f"{dataset}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run feature_engineering.py first.")

    df = pd.read_parquet(path)
    y = df["isFraud"].values
    drop_cols = ["isFraud", "TransactionID"]
    X_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    print(f"Data: {X_df.shape[0]:,} samples, {X_df.shape[1]} features")
    print(f"Fraud rate: {y.mean():.4%} ({y.sum():,} / {len(y):,})")
    return df, X_df, y


def preprocess_fold_lgb(X_train_df, X_val_df):
    """Fold-internal preprocessing for LightGBM (leakage-free)."""
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.impute import SimpleImputer
    cat_cols = X_train_df.select_dtypes(include=["object", "string"]).columns.tolist()
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train_df = X_train_df.copy()
        X_val_df = X_val_df.copy()
        X_train_df[cat_cols] = enc.fit_transform(X_train_df[cat_cols])
        X_val_df[cat_cols] = enc.transform(X_val_df[cat_cols])
    imp = SimpleImputer(strategy="median")
    return imp.fit_transform(X_train_df.values.astype(np.float64)), imp.transform(X_val_df.values.astype(np.float64))


def train_and_evaluate(
    X_df: pd.DataFrame,
    y: np.ndarray,
    n_folds: int = 5,
) -> tuple[np.ndarray, dict, list]:
    neg, pos = np.bincount(y.astype(int))
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "scale_pos_weight": neg / pos,
        "max_depth": 6,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_jobs": -1,
        "random_state": 42,
        "verbose": -1,
    }

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))
    models = []
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_df, y)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")
        y_train, y_val = y[train_idx], y[val_idx]
        X_train, X_val = preprocess_fold_lgb(X_df.iloc[train_idx], X_df.iloc[val_idx])

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(100),
            ],
        )
        models.append(model)

        val_pred = model.predict(X_val)
        oof_preds[val_idx] = val_pred

        auc = roc_auc_score(y_val, val_pred)
        ap = average_precision_score(y_val, val_pred)
        print(f"  ROC-AUC: {auc:.4f}, PR-AUC: {ap:.4f}")
        fold_metrics.append({"fold": fold + 1, "roc_auc": auc, "pr_auc": ap})

    overall_auc = roc_auc_score(y, oof_preds)
    overall_ap = average_precision_score(y, oof_preds)
    print(f"\n=== Overall OOF Results ===")
    print(f"ROC-AUC: {overall_auc:.4f}")
    print(f"PR-AUC:  {overall_ap:.4f}")

    # NOTE: Threshold tuned on all OOF predictions for FP/TP pool construction only.
    # All downstream LLM analyses use continuous fraud_score, not binary predictions.
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
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = Path(output) if output else MODEL_DIR / f"lgb_{dataset}.txt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    best_fold = max(range(len(models)), key=lambda i: metrics["fold_metrics"][i]["pr_auc"])
    models[best_fold].save_model(str(model_path))
    print(f"\nBest model (fold {best_fold + 1}) saved to {model_path}")

    pred_df = pd.DataFrame({
        "index": range(len(oof_preds)),
        "true_label": df["isFraud"].values,
        "fraud_score": oof_preds,
        "predicted": (oof_preds >= metrics["best_threshold"]).astype(int),
    })
    pred_path = RESULTS_DIR / f"predictions_lgb_{dataset}.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"Predictions saved to {pred_path}")

    threshold = metrics["best_threshold"]
    fp_mask = (oof_preds >= threshold) & (df["isFraud"].values == 0)
    fp_df = df[fp_mask].copy()
    fp_df["fraud_score"] = oof_preds[fp_mask]
    fp_path = RESULTS_DIR / f"false_positives_lgb_{dataset}.csv"
    fp_df.to_csv(fp_path, index=False)
    print(f"False Positives saved to {fp_path} ({len(fp_df):,} cases)")

    metrics["dataset"] = dataset
    metrics["model"] = "lightgbm"
    metrics["timestamp"] = datetime.now().isoformat()
    metrics_path = RESULTS_DIR / f"metrics_lgb_{dataset}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM fraud detection baseline")
    parser.add_argument("--dataset", type=str, default="credit_card",
                        choices=["ieee_cis", "credit_card", "paysim"])
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    df, X_df, y = load_data(args.dataset)
    oof_preds, metrics, models = train_and_evaluate(X_df, y, n_folds=args.folds)
    save_results(df, oof_preds, metrics, models, args.dataset, args.output)


if __name__ == "__main__":
    main()
