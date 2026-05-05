"""거래 데이터 → 자연어 변환.

ML 모델의 수치형 feature를 LLM이 이해할 수 있는 자연어 텍스트로 변환.
각 거래를 구조화된 텍스트 설명으로 변환하여 LLM 입력으로 사용.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# 거래 금액 범위 설명
def describe_amount(amount: float, mean_amount: float, std_amount: float) -> str:
    z_score = (amount - mean_amount) / (std_amount + 1e-8)
    if z_score > 3:
        size = "extremely large (>3 std above average)"
    elif z_score > 2:
        size = "very large (>2 std above average)"
    elif z_score > 1:
        size = "above average"
    elif z_score > -1:
        size = "typical range"
    else:
        size = "below average"

    is_round = amount == int(amount)
    parts = [f"${amount:,.2f} ({size})"]
    if is_round and amount >= 100:
        parts.append("round number")
    return ", ".join(parts)


def describe_time(hour: int | None, is_weekend: bool | None) -> str:
    parts = []
    if hour is not None:
        if 6 <= hour < 12:
            parts.append(f"morning ({hour}:00)")
        elif 12 <= hour < 18:
            parts.append(f"afternoon ({hour}:00)")
        elif 18 <= hour < 22:
            parts.append(f"evening ({hour}:00)")
        else:
            parts.append(f"late night/early morning ({hour}:00)")
    if is_weekend is not None:
        parts.append("weekend" if is_weekend else "weekday")
    return ", ".join(parts) if parts else "unknown"


def convert_ieee_cis_row(row: pd.Series, stats: dict) -> dict:
    """IEEE-CIS 거래 한 건을 자연어로 변환."""
    amount = row.get("TransactionAmt", 0)
    fraud_label = int(row.get("isFraud", -1))

    text_parts = []
    text_parts.append(f"Transaction Amount: {describe_amount(amount, stats['mean_amt'], stats['std_amt'])}")

    hour = row.get("hour")
    is_weekend = row.get("is_weekend")
    if hour is not None:
        text_parts.append(f"Time: {describe_time(int(hour), bool(is_weekend) if is_weekend is not None else None)}")

    product = row.get("ProductCD")
    if product is not None:
        text_parts.append(f"Product: {product}")

    email = row.get("P_emaildomain")
    if email is not None and email != "missing":
        text_parts.append(f"Email domain: {email}")

    # C features (count features)
    c_features = {k: v for k, v in row.items() if k.startswith("C") and k[1:].isdigit() and pd.notna(v)}
    if c_features:
        high_counts = {k: int(v) for k, v in c_features.items() if v > stats.get(f"{k}_p95", float("inf"))}
        if high_counts:
            text_parts.append(f"Unusual count features (>95th pct): {high_counts}")

    # D features (timedelta)
    d_features = {k: v for k, v in row.items() if k.startswith("D") and k[1:].isdigit() and pd.notna(v)}
    if d_features:
        text_parts.append(f"Time deltas: {len(d_features)} available")

    return {
        "transaction_id": row.get("TransactionID", "unknown"),
        "text": "\n".join(text_parts),
        "is_fraud": fraud_label,
        "fraud_score": row.get("fraud_score", None),
        "amount": float(amount),
    }


def convert_credit_card_row(row: pd.Series, stats: dict) -> dict:
    """Credit Card 거래 한 건을 자연어로 변환.

    PCA features를 해석 가능한 패턴으로 변환하여 LLM이 추론할 수 있도록 함.
    """
    amount = row.get("TransactionAmt", row.get("Amount", 0))
    fraud_label = int(row.get("isFraud", row.get("Class", -1)))

    text_parts = []
    text_parts.append(f"Transaction Amount: {describe_amount(amount, stats['mean_amt'], stats['std_amt'])}")

    time_val = row.get("Time")
    if time_val is not None:
        hours_elapsed = time_val / 3600
        hour_of_day = int(hours_elapsed) % 24
        text_parts.append(f"Time: {describe_time(hour_of_day, None)}")

    # PCA components → 해석 가능한 패턴으로 매핑
    # Credit Card 데이터셋의 V 컴포넌트는 PCA 변환된 값이지만,
    # 이상치 수와 방향으로 패턴을 추론할 수 있음
    v_features = {k: v for k, v in row.items() if k.startswith("V") and pd.notna(v)}
    outlier_vs = {k: round(float(v), 2) for k, v in v_features.items() if abs(v) > 3}
    mild_outliers = {k: round(float(v), 2) for k, v in v_features.items() if 2 < abs(v) <= 3}

    n_severe = len(outlier_vs)
    n_mild = len(mild_outliers)
    n_total_features = len(v_features)

    # 이상치 수에 따른 해석 가능한 설명
    if n_severe >= 5:
        text_parts.append(f"Behavioral pattern: HIGHLY ANOMALOUS - {n_severe} features show extreme deviation (>3 std)")
        text_parts.append(f"  This indicates the transaction significantly deviates from normal spending behavior across multiple dimensions")
    elif n_severe >= 2:
        text_parts.append(f"Behavioral pattern: MODERATELY ANOMALOUS - {n_severe} features show extreme deviation")
        text_parts.append(f"  Several transaction characteristics differ substantially from typical patterns")
    elif n_severe == 1:
        text_parts.append(f"Behavioral pattern: MILDLY ANOMALOUS - 1 feature with extreme deviation")
        text_parts.append(f"  One transaction characteristic is unusual, but most are normal")
    else:
        text_parts.append(f"Behavioral pattern: NORMAL - all {n_total_features} features within expected range")

    if mild_outliers:
        text_parts.append(f"Additional mild anomalies: {n_mild} features between 2-3 std from normal")

    # 금액 컨텍스트 강화
    if amount == 0:
        text_parts.append("Note: Zero-amount transaction (common for authorization checks or failed transactions)")
    elif amount < 1:
        text_parts.append("Note: Micro-transaction (< $1, could be a test transaction before larger fraud)")
    elif amount > stats['mean_amt'] + 3 * stats['std_amt']:
        text_parts.append(f"Note: Extremely large amount relative to average (${stats['mean_amt']:.2f})")

    # 금액 소수점 패턴
    decimal = round(amount - int(amount), 2)
    if amount > 0 and decimal == 0:
        text_parts.append("Amount pattern: Round number (no cents)")
    elif decimal in (0.99, 0.95):
        text_parts.append("Amount pattern: Retail pricing pattern ($X.99/$X.95)")

    return {
        "transaction_id": row.name if hasattr(row, "name") else "unknown",
        "text": "\n".join(text_parts),
        "is_fraud": fraud_label,
        "fraud_score": row.get("fraud_score", None),
        "amount": float(amount),
    }


def convert_paysim_row(row: pd.Series, stats: dict) -> dict:
    """PaySim 거래 한 건을 자연어로 변환."""
    amount = row.get("TransactionAmt", row.get("amount", 0))
    fraud_label = int(row.get("isFraud", -1))

    text_parts = []
    text_parts.append(f"Transaction Amount: {describe_amount(amount, stats['mean_amt'], stats['std_amt'])}")

    tx_type = row.get("type")
    if tx_type is not None:
        text_parts.append(f"Transaction type: {tx_type}")

    old_balance_orig = row.get("oldbalanceOrg")
    new_balance_orig = row.get("newbalanceOrig")
    if old_balance_orig is not None and new_balance_orig is not None:
        balance_change = new_balance_orig - old_balance_orig
        text_parts.append(
            f"Sender balance: ${old_balance_orig:,.2f} → ${new_balance_orig:,.2f} "
            f"(change: ${balance_change:,.2f})"
        )

    old_balance_dest = row.get("oldbalanceDest")
    new_balance_dest = row.get("newbalanceDest")
    if old_balance_dest is not None and new_balance_dest is not None:
        text_parts.append(
            f"Receiver balance: ${old_balance_dest:,.2f} → ${new_balance_dest:,.2f}"
        )

    is_flagged = row.get("isFlaggedFraud")
    if is_flagged:
        text_parts.append("System flagged: YES (amount > $200,000 transfer)")

    return {
        "transaction_id": row.name if hasattr(row, "name") else "unknown",
        "text": "\n".join(text_parts),
        "is_fraud": fraud_label,
        "fraud_score": row.get("fraud_score", None),
        "amount": float(amount),
    }


CONVERTERS = {
    "ieee_cis": convert_ieee_cis_row,
    "credit_card": convert_credit_card_row,
    "paysim": convert_paysim_row,
}


def compute_stats(df: pd.DataFrame, dataset: str) -> dict:
    """변환에 필요한 통계치 계산.

    NOTE: Data leakage 방지를 위해, 이 함수에 전달하는 df는 반드시 training split만 포함해야 합니다. 전체 데이터셋을 넣으면 test 정보가 텍스트 설명에 누출됩니다.
    """
    amt_col = "TransactionAmt" if "TransactionAmt" in df.columns else "Amount"
    stats = {
        "mean_amt": float(df[amt_col].mean()),
        "std_amt": float(df[amt_col].std()),
    }
    # C features percentile (IEEE-CIS)
    for col in df.columns:
        if col.startswith("C") and col[1:].isdigit():
            stats[f"{col}_p95"] = float(df[col].quantile(0.95))
    return stats


def convert_dataset(
    input_path: str,
    output_dir: str,
    dataset: str,
    max_samples: int | None = None,
    stats_path: str | None = None,
) -> None:
    """데이터셋 전체를 자연어로 변환.

    Args:
        stats_path: 사전 계산된 통계 JSON 경로. 제공 시 해당 통계 사용 (leakage-free). 미제공 시 입력 데이터에서 계산 (전체 데이터 = train일 때만 안전).
    """
    df = pd.read_parquet(input_path) if input_path.endswith(".parquet") else pd.read_csv(input_path)

    if max_samples and len(df) > max_samples:
        fraud_df = df[df["isFraud"] == 1]
        normal_df = df[df["isFraud"] == 0].sample(
            n=min(max_samples - len(fraud_df), len(df[df["isFraud"] == 0])),
            random_state=42,
        )
        df = pd.concat([fraud_df, normal_df]).reset_index(drop=True)
        print(f"Sampled {len(df):,} transactions (all {len(fraud_df):,} fraud + {len(normal_df):,} normal)")

    # 통계: 외부 제공 또는 입력 데이터에서 계산
    if stats_path:
        with open(stats_path) as f:
            stats = json.load(f)
        print(f"Using pre-computed stats from {stats_path}")
    else:
        import warnings
        warnings.warn(
            "No --stats-path provided. Computing stats from input data. "
            "This is safe ONLY if the input is the training set. "
            "For evaluation/test data, always provide a train-fitted stats file "
            "to prevent distribution leakage.",
            stacklevel=2,
        )
        stats = compute_stats(df, dataset)
        print(f"WARNING: Stats computed from input data. Safe only if input = training set.")
    converter = CONVERTERS[dataset]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    out_file = output_path / f"{dataset}_text.jsonl"

    count = 0
    with open(out_file, "w") as f:
        for idx, row in df.iterrows():
            result = converter(row, stats)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            count += 1
            if count % 10000 == 0:
                print(f"  Converted {count:,} transactions...")

    print(f"\nConverted {count:,} transactions to {out_file}")
    print(f"  Fraud: {df['isFraud'].sum():,}, Normal: {(df['isFraud'] == 0).sum():,}")


def main():
    parser = argparse.ArgumentParser(description="Convert transactions to natural language")
    parser.add_argument("--input", type=str, required=True, help="Input parquet/csv path")
    parser.add_argument(
        "--output",
        type=str,
        default=str(DATA_DIR / "processed" / "transactions_text"),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(CONVERTERS.keys()),
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max samples (fraud cases always included)",
    )
    parser.add_argument(
        "--stats_path",
        type=str,
        default=None,
        help="Pre-computed stats JSON (from training set) for leakage-free conversion",
    )
    args = parser.parse_args()

    convert_dataset(args.input, args.output, args.dataset, args.max_samples, args.stats_path)


if __name__ == "__main__":
    main()
