"""공개 사기 탐지 데이터셋 다운로드 스크립트.

지원 데이터셋:
  - ieee_cis: IEEE-CIS Fraud Detection (Kaggle Competition)
  - credit_card: Credit Card Fraud Detection (Kaggle Dataset)
  - paysim: PaySim Synthetic Financial Dataset (Kaggle Dataset)
"""

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

DATASETS = {
    "ieee_cis": {
        "type": "competition",
        "slug": "ieee-fraud-detection",
        "dest": "ieee_cis",
        "description": "IEEE-CIS Fraud Detection (~1.2GB, 590K transactions)",
    },
    "credit_card": {
        "type": "dataset",
        "slug": "mlg-ulb/creditcardfraud",
        "dest": "credit_card",
        "description": "Credit Card Fraud Detection (144MB, 284K transactions)",
    },
    "paysim": {
        "type": "dataset",
        "slug": "ealaxi/paysim1",
        "dest": "paysim",
        "description": "PaySim Synthetic (470MB, 6.3M transactions)",
    },
}


def download_dataset(name: str) -> None:
    info = DATASETS[name]
    dest = DATA_DIR / info["dest"]
    dest.mkdir(parents=True, exist_ok=True)

    # 이미 파일이 있으면 스킵
    existing = list(dest.glob("*.csv"))
    if existing:
        print(f"[SKIP] {name}: {len(existing)} CSV files already exist in {dest}")
        return

    print(f"[DOWNLOAD] {info['description']}")
    print(f"  -> {dest}")

    try:
        if info["type"] == "competition":
            subprocess.run(
                ["kaggle", "competitions", "download", "-c", info["slug"], "-p", str(dest)],
                check=True,
            )
        else:
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", info["slug"], "-p", str(dest)],
                check=True,
            )
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to download {name}: {e}")
        print("  Kaggle API 인증을 확인하세요: ~/.kaggle/kaggle.json")
        sys.exit(1)

    # zip 파일 해제
    for zf in dest.glob("*.zip"):
        print(f"  Extracting {zf.name}...")
        with zipfile.ZipFile(zf, "r") as z:
            z.extractall(dest)
        zf.unlink()
        print(f"  Removed {zf.name}")

    csv_count = len(list(dest.glob("*.csv")))
    print(f"[DONE] {name}: {csv_count} CSV files extracted")


def list_datasets() -> None:
    print("Available datasets:")
    for name, info in DATASETS.items():
        dest = DATA_DIR / info["dest"]
        existing = list(dest.glob("*.csv"))
        status = f"({len(existing)} files)" if existing else "(not downloaded)"
        print(f"  {name:15s} {info['description']} {status}")


def main():
    parser = argparse.ArgumentParser(description="Download fraud detection datasets")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()) + ["all"],
        help="Dataset to download (or 'all')",
    )
    parser.add_argument("--list", action="store_true", help="List available datasets")
    args = parser.parse_args()

    if args.list or not args.dataset:
        list_datasets()
        return

    targets = DATASETS.keys() if args.dataset == "all" else [args.dataset]
    for name in targets:
        download_dataset(name)


if __name__ == "__main__":
    main()
