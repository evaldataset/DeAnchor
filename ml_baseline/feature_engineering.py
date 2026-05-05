"""거래 데이터 Feature Engineering.

IEEE-CIS, Credit Card, PaySim 데이터셋에 대한 통합 전처리.
각 데이터셋에서 공통 feature set을 추출하여 ML 모델 학습에 사용.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def load_ieee_cis(data_path: Path) -> pd.DataFrame:
    """IEEE-CIS 데이터 로드 및 전처리."""
    tx_file = data_path / "train_transaction.csv"
    id_file = data_path / "train_identity.csv"

    if not tx_file.exists():
        raise FileNotFoundError(f"IEEE-CIS transaction file not found: {tx_file}")

    print(f"Loading IEEE-CIS transactions from {tx_file}...")
    tx = pd.read_csv(tx_file)
    print(f"  Transactions: {len(tx):,} rows, {tx.columns.size} columns")

    if id_file.exists():
        identity = pd.read_csv(id_file)
        tx = tx.merge(identity, on="TransactionID", how="left")
        print(f"  After identity merge: {tx.columns.size} columns")

    # 핵심 features 선택 (메모리 효율)
    core_features = [
        "TransactionID", "isFraud", "TransactionAmt", "TransactionDT",
        "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
        "addr1", "addr2", "P_emaildomain", "R_emaildomain",
        "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9",
        "C10", "C11", "C12", "C13", "C14",
        "D1", "D2", "D3", "D4", "D5", "D10", "D11", "D15",
    ]
    available = [c for c in core_features if c in tx.columns]
    tx = tx[available]

    return tx


def load_credit_card(data_path: Path) -> pd.DataFrame:
    """Credit Card 데이터 로드."""
    csv_file = data_path / "creditcard.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f"Credit Card file not found: {csv_file}")

    print(f"Loading Credit Card from {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"  {len(df):,} rows, Fraud rate: {df['Class'].mean():.4%}")

    df = df.rename(columns={"Class": "isFraud", "Amount": "TransactionAmt"})
    return df


def load_paysim(data_path: Path) -> pd.DataFrame:
    """PaySim 데이터 로드."""
    # PaySim 파일명은 여러 변형이 있음
    candidates = list(data_path.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV files found in {data_path}")

    csv_file = candidates[0]
    print(f"Loading PaySim from {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"  {len(df):,} rows, Fraud rate: {df['isFraud'].mean():.4%}")

    df = df.rename(columns={"amount": "TransactionAmt"})
    return df


def engineer_features(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """공통 Feature Engineering."""
    print(f"\nEngineering features for {dataset_name}...")

    # 금액 관련 features
    if "TransactionAmt" in df.columns:
        df["log_amount"] = np.log1p(df["TransactionAmt"])
        df["amount_decimal"] = df["TransactionAmt"] - df["TransactionAmt"].astype(int)
        df["is_round_amount"] = (df["amount_decimal"] == 0).astype(int)

    # 시간 관련 features (IEEE-CIS)
    if "TransactionDT" in df.columns:
        df["hour"] = (df["TransactionDT"] / 3600).astype(int) % 24
        df["day_of_week"] = (df["TransactionDT"] / 86400).astype(int) % 7
        df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # PaySim 시간 features
    if "step" in df.columns:
        df["hour"] = df["step"] % 24
        df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)

    # 카테고리 → 문자열 "missing" 으로 통일 (인코딩은 CV 루프에서 수행)
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = df[col].fillna("missing")

    # NOTE: 결측치 imputation과 카테고리 인코딩은 여기서 하지 않음.
    # Data leakage 방지를 위해 train_xgboost.py의 CV 루프 안에서
    # train fold 기준으로 fit → val fold에 transform 해야 함.

    print(f"  Final shape: {df.shape}")
    print(f"  Fraud rate: {df['isFraud'].mean():.4%}")

    return df


LOADERS = {
    "ieee_cis": load_ieee_cis,
    "credit_card": load_credit_card,
    "paysim": load_paysim,
}


def main():
    parser = argparse.ArgumentParser(description="Feature engineering for fraud detection")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to raw dataset directory or dataset name (ieee_cis, credit_card, paysim)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for processed data (default: data/processed/<dataset>.parquet)",
    )
    args = parser.parse_args()

    # 데이터셋 이름으로 입력한 경우
    if args.input in LOADERS:
        dataset_name = args.input
        data_path = RAW_DIR / dataset_name
    else:
        data_path = Path(args.input)
        dataset_name = data_path.name

    loader = LOADERS.get(dataset_name)
    if loader is None:
        print(f"Unknown dataset: {dataset_name}")
        print(f"Available: {list(LOADERS.keys())}")
        return

    df = loader(data_path)
    df = engineer_features(df, dataset_name)

    # 저장
    output_path = Path(args.output) if args.output else PROCESSED_DIR / f"{dataset_name}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"\nSaved to {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # 메타데이터 저장
    meta = {
        "dataset": dataset_name,
        "rows": len(df),
        "columns": list(df.columns),
        "fraud_rate": float(df["isFraud"].mean()),
        "feature_count": len(df.columns) - 1,
    }
    meta_path = output_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
