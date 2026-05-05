"""Calibration 분석: fraud_likelihood의 ECE + reliability diagram."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def compute_ece(predictions: list[float], labels: list[int], n_bins: int = 10) -> dict:
    """Expected Calibration Error 계산."""
    preds = np.array(predictions)
    labs = np.array(labels)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    ece = 0.0
    bins_data = []
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (preds >= lo) & (preds < hi) if i < n_bins - 1 else (preds >= lo) & (preds <= hi)

        if mask.sum() == 0:
            continue

        bin_acc = labs[mask].mean()
        bin_conf = preds[mask].mean()
        bin_size = mask.sum()
        ece += (bin_size / len(preds)) * abs(bin_acc - bin_conf)

        bins_data.append({
            "bin": f"[{lo:.1f},{hi:.1f})",
            "count": int(bin_size),
            "mean_confidence": round(float(bin_conf), 3),
            "mean_accuracy": round(float(bin_acc), 3),
            "gap": round(float(abs(bin_acc - bin_conf)), 3),
        })

    return {
        "ece": round(float(ece), 4),
        "n_bins": n_bins,
        "n_samples": len(preds),
        "bins": bins_data,
    }


def analyze_calibration(results_path: str, key: str = "fp_explanation") -> dict:
    """실험 결과에서 calibration 분석."""
    with open(results_path) as f:
        results = [json.loads(line) for line in f]

    preds = []
    labels = []
    for r in results:
        assessment = r.get(key) or r.get("blind_assessment") or r.get("assessment", {})
        fl = assessment.get("fraud_likelihood")
        is_fraud = r.get("original", {}).get("is_fraud")

        if isinstance(fl, (int, float)) and is_fraud in (0, 1):
            preds.append(float(fl))
            labels.append(int(is_fraud))

    if not preds:
        return {"error": "No valid predictions"}

    ece_result = compute_ece(preds, labels)

    print(f"ECE: {ece_result['ece']:.4f} (n={ece_result['n_samples']})")
    print(f"Bins:")
    for b in ece_result["bins"]:
        print(f"  {b['bin']:12s} n={b['count']:4d}  conf={b['mean_confidence']:.3f}  acc={b['mean_accuracy']:.3f}  gap={b['gap']:.3f}")

    return ece_result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--key", default="fp_explanation")
    args = parser.parse_args()
    analyze_calibration(args.input, args.key)
