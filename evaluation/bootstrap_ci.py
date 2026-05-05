"""Bootstrap 신뢰구간 + 통계적 유의성 검정.

NeurIPS 수준 통계 보고를 위한 유틸리티.
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats


def bootstrap_ci(
    data: list[float],
    stat_func=np.mean,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Bootstrap 신뢰구간 계산."""
    rng = np.random.RandomState(seed)
    data = np.array(data)
    n = len(data)

    boot_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        boot_stats.append(stat_func(sample))

    boot_stats = np.array(boot_stats)
    alpha = (1 - ci) / 2

    return {
        "mean": float(stat_func(data)),
        "std": float(np.std(data)),
        "ci_lower": float(np.percentile(boot_stats, alpha * 100)),
        "ci_upper": float(np.percentile(boot_stats, (1 - alpha) * 100)),
        "ci_level": ci,
        "n": n,
        "n_bootstrap": n_bootstrap,
    }


def paired_permutation_test(
    group_a: list[float],
    group_b: list[float],
    n_permutations: int = 10000,
    seed: int = 42,
) -> dict:
    """Paired permutation test (sign-flip / exchangeability) for two paired groups.

    Requires len(group_a) == len(group_b) where position i refers to the same unit. Tests H0: mean difference d_i = a_i
    - b_i is symmetric around 0, by randomly flipping the sign of each d_i independently. This is the correct test for
    paired data such as within-subject repeated measures.
    """
    rng = np.random.RandomState(seed)
    a = np.array(group_a, dtype=float)
    b = np.array(group_b, dtype=float)
    if len(a) != len(b):
        raise ValueError(
            f"paired_permutation_test requires equal lengths, got {len(a)} vs {len(b)}. "
            "Use two_sample_permutation_test for unpaired data."
        )

    diffs = a - b
    observed_mean_diff = float(np.mean(diffs))

    # Sign-flip permutation: flip each diff's sign with p=0.5
    n = len(diffs)
    count = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1.0, 1.0], size=n)
        perm_mean = float(np.mean(signs * diffs))
        if abs(perm_mean) >= abs(observed_mean_diff):
            count += 1

    p_value = count / n_permutations

    return {
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_diff": observed_mean_diff,
        "observed_diff": observed_mean_diff,
        "p_value": float(p_value),
        "test_type": "paired_sign_flip",
        "significant_005": p_value < 0.05,
        "n_permutations": n_permutations,
    }


def two_sample_permutation_test(
    group_a: list[float],
    group_b: list[float],
    n_permutations: int = 10000,
    seed: int = 42,
) -> dict:
    """Two-sample (unpaired) permutation test.

    Tests H0: group_a and group_b come from the same distribution, by pooling and randomly reassigning labels
    n_permutations times. Use this when observations in a and b are NOT paired (e.g., different units).
    """
    rng = np.random.RandomState(seed)
    a = np.array(group_a, dtype=float)
    b = np.array(group_b, dtype=float)

    observed_diff = float(np.mean(a) - np.mean(b))
    combined = np.concatenate([a, b])
    n_a = len(a)

    count = 0
    for _ in range(n_permutations):
        perm = rng.permutation(combined)
        perm_diff = float(np.mean(perm[:n_a]) - np.mean(perm[n_a:]))
        if abs(perm_diff) >= abs(observed_diff):
            count += 1

    p_value = count / n_permutations
    return {
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "observed_diff": observed_diff,
        "p_value": float(p_value),
        "test_type": "two_sample_label_shuffle",
        "significant_005": p_value < 0.05,
        "n_permutations": n_permutations,
        "n_a": n_a,
        "n_b": len(b),
    }


def analyze_experiment_with_ci(results_path: str) -> dict:
    """실험 결과에 Bootstrap CI를 추가."""
    with open(results_path) as f:
        results = [json.loads(line) for line in f]

    fp_fl, tp_fl = [], []
    for r in results:
        cat = r.get("original", {}).get("category", "")
        # Score-blind or score-aware
        assessment = r.get("blind_assessment") or r.get("fp_explanation", {})
        fl = assessment.get("fraud_likelihood")

        if isinstance(fl, (int, float)):
            if cat == "false_positive":
                fp_fl.append(fl)
            elif cat == "true_positive":
                tp_fl.append(fl)

    if not fp_fl or not tp_fl:
        return {"error": "No FP or TP data found"}

    fp_ci = bootstrap_ci(fp_fl)
    tp_ci = bootstrap_ci(tp_fl)
    # FP and TP are different transactions → unpaired test
    diff_test = two_sample_permutation_test(tp_fl, fp_fl)

    analysis = {
        "fp": fp_ci,
        "tp": tp_ci,
        "separation": {
            "delta": float(np.mean(tp_fl) - np.mean(fp_fl)),
            "permutation_test": diff_test,
        },
    }

    print(f"FP fraud_likelihood: {fp_ci['mean']:.3f} [{fp_ci['ci_lower']:.3f}, {fp_ci['ci_upper']:.3f}]")
    print(f"TP fraud_likelihood: {tp_ci['mean']:.3f} [{tp_ci['ci_lower']:.3f}, {tp_ci['ci_upper']:.3f}]")
    print(f"Delta: {analysis['separation']['delta']:+.3f} (p={diff_test['p_value']:.4f})")

    return analysis


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    result = analyze_experiment_with_ci(args.input)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to {args.output}")
