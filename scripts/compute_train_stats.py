"""Training set에서만 통계를 계산하여 JSON으로 저장.

Data leakage 방지: 이 통계를 transaction_converter에 전달하면
test set 정보가 텍스트 설명에 누출되지 않음.
80% stratified train split 사용.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def compute_and_save(dataset: str) -> None:
    path = DATA_DIR / f"{dataset}.parquet"
    df = pd.read_parquet(path)

    # 80/20 stratified split (seed 고정)
    train_df, _ = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["isFraud"],
    )
    print(f"{dataset}: {len(df):,} total → {len(train_df):,} train (80%)")

    amt_col = "TransactionAmt" if "TransactionAmt" in train_df.columns else "Amount"
    stats = {
        "mean_amt": float(train_df[amt_col].mean()),
        "std_amt": float(train_df[amt_col].std()),
        "source": "80% stratified train split",
        "train_size": len(train_df),
        "total_size": len(df),
    }
    for col in train_df.columns:
        if col.startswith("C") and col[1:].isdigit():
            stats[f"{col}_p95"] = float(train_df[col].quantile(0.95))

    out_path = DATA_DIR / f"{dataset}_train_stats.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats saved to {out_path}")
    print(f"  mean_amt={stats['mean_amt']:.2f}, std_amt={stats['std_amt']:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all", choices=["credit_card", "paysim", "ieee_cis", "all"])
    args = parser.parse_args()

    datasets = ["credit_card", "paysim", "ieee_cis"] if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        compute_and_save(ds)


if __name__ == "__main__":
    main()
