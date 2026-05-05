"""Anchoring coefficient α computation.

α = (mean_fl_aware - mean_fl_blind) / mean_ml_score

Measures how much of LLM output is determined by the ML score
vs. independent reasoning from transaction features.
"""

import json
import numpy as np
from pathlib import Path


def compute_alpha(
    aware_path: str,
    blind_path: str,
    aware_key: str = "fp_explanation",
    blind_key: str = "blind_assessment",
) -> dict:
    """Compute anchoring coefficient from paired aware/blind experiments."""

    def load_fl(path, key):
        fl_list, ml_list = [], []
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                assessment = r.get(key) or r.get("assessment", {})
                fl = assessment.get("fraud_likelihood")
                ml = r.get("ml_score") or r.get("original", {}).get("fraud_score", 0)
                if isinstance(fl, (int, float)):
                    fl_list.append(fl)
                    ml_list.append(ml)
        return np.array(fl_list), np.array(ml_list)

    fl_aware, ml_aware = load_fl(aware_path, aware_key)
    fl_blind, ml_blind = load_fl(blind_path, blind_key)

    mean_aware = np.mean(fl_aware)
    mean_blind = np.mean(fl_blind)
    mean_ml = np.mean(ml_aware)

    alpha = (mean_aware - mean_blind) / mean_ml if mean_ml > 0 else 0

    result = {
        "alpha": round(float(alpha), 4),
        "mean_fl_aware": round(float(mean_aware), 4),
        "mean_fl_blind": round(float(mean_blind), 4),
        "mean_ml_score": round(float(mean_ml), 4),
        "shift": round(float(mean_aware - mean_blind), 4),
        "n_aware": len(fl_aware),
        "n_blind": len(fl_blind),
    }

    print(f"α = ({mean_aware:.3f} - {mean_blind:.3f}) / {mean_ml:.3f} = {alpha:.4f}")
    print(f"  Shift: {mean_aware - mean_blind:+.3f} points")
    print(f"  n: aware={len(fl_aware)}, blind={len(fl_blind)}")

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--aware", required=True)
    parser.add_argument("--blind", required=True)
    parser.add_argument("--aware_key", default="fp_explanation")
    parser.add_argument("--blind_key", default="blind_assessment")
    args = parser.parse_args()
    compute_alpha(args.aware, args.blind, args.aware_key, args.blind_key)
