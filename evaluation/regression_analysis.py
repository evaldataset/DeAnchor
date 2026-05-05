"""Transaction-level regression analysis for score reliance.

Replaces ad-hoc alpha with proper statistical modeling:
- Logistic regression: fraud_likelihood ~ score_present * ml_score
- Reports coefficient, CI, interaction term
- Optionally fits mixed-effects model with transaction random intercept
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS_DIR = Path(__file__).resolve().parent.parent / "experiments" / "results"


def load_controlled_ablation(with_path: str, without_path: str) -> pd.DataFrame:
    """Load paired controlled ablation data into a long-format DataFrame."""
    rows = []

    for path, condition in [(with_path, 1), (without_path, 0)]:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                tx = r.get("original", {})
                assessment = r.get("assessment", {})
                fl = assessment.get("fraud_likelihood")
                if not isinstance(fl, (int, float)):
                    continue
                rows.append({
                    "transaction_id": tx.get("transaction_id", hash(tx.get("text", "")[:50])),
                    "category": tx.get("category", "unknown"),
                    "is_fraud": 1 if tx.get("category") == "true_positive" else 0,
                    "ml_score": tx.get("fraud_score", 0.5),
                    "score_present": condition,
                    "fraud_likelihood": fl,
                })

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} observations ({df['score_present'].sum()} with score, "
          f"{(~df['score_present'].astype(bool)).sum()} without)")
    return df


def run_ols_regression(df: pd.DataFrame) -> dict:
    """OLS regression: fraud_likelihood ~ score_present * ml_score + is_fraud."""
    from sklearn.linear_model import LinearRegression

    # Prepare features
    X = df[["score_present", "ml_score", "is_fraud"]].copy()
    X["score_x_ml"] = X["score_present"] * X["ml_score"]
    y = df["fraud_likelihood"].values

    model = LinearRegression()
    model.fit(X, y)

    # Manual coefficient CIs via bootstrap
    n_boot = 5000
    rng = np.random.RandomState(42)
    coefs_boot = []
    for _ in range(n_boot):
        idx = rng.choice(len(y), len(y), replace=True)
        m = LinearRegression().fit(X.iloc[idx], y[idx])
        coefs_boot.append(np.concatenate([[m.intercept_], m.coef_]))
    coefs_boot = np.array(coefs_boot)

    feature_names = ["intercept", "score_present", "ml_score", "is_fraud", "score_x_ml"]
    results = {}
    print("\n=== OLS Regression: fraud_likelihood ~ score_present * ml_score + is_fraud ===")
    print(f"{'Feature':<20} {'Coef':>8} {'95% CI':>20} {'Significant':>12}")
    print("-" * 65)

    all_coefs = np.concatenate([[model.intercept_], model.coef_])
    for i, name in enumerate(feature_names):
        coef = all_coefs[i]
        ci_lo, ci_hi = np.percentile(coefs_boot[:, i], [2.5, 97.5])
        sig = "Yes" if (ci_lo > 0 or ci_hi < 0) else "No"
        print(f"{name:<20} {coef:>8.4f} [{ci_lo:>8.4f}, {ci_hi:>8.4f}] {sig:>12}")
        results[name] = {
            "coefficient": round(float(coef), 4),
            "ci_lower": round(float(ci_lo), 4),
            "ci_upper": round(float(ci_hi), 4),
            "significant": sig == "Yes",
        }

    # R-squared
    y_pred = model.predict(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"\nR² = {r2:.4f}")
    results["r_squared"] = round(float(r2), 4)

    return results


def run_simple_analysis(df: pd.DataFrame) -> dict:
    """Simple paired analysis comparing with/without score conditions."""
    with_score = df[df["score_present"] == 1]["fraud_likelihood"].values
    without_score = df[df["score_present"] == 0]["fraud_likelihood"].values

    shift = np.mean(with_score) - np.mean(without_score)
    t_stat, p_val = sp_stats.ttest_rel(with_score, without_score) if len(with_score) == len(without_score) else sp_stats.ttest_ind(with_score, without_score)

    # Cohen's d for paired
    diff = with_score - without_score if len(with_score) == len(without_score) else None
    if diff is not None:
        d = np.mean(diff) / np.std(diff, ddof=1)
    else:
        pooled_std = np.sqrt((np.var(with_score) + np.var(without_score)) / 2)
        d = shift / pooled_std

    print("\n=== Simple Paired Analysis ===")
    print(f"Mean with score:    {np.mean(with_score):.4f}")
    print(f"Mean without score: {np.mean(without_score):.4f}")
    print(f"Shift:              {shift:+.4f}")
    print(f"Cohen's d:          {d:.4f}")
    print(f"t-test p-value:     {p_val:.6f}")

    # By category
    print("\n=== By Category ===")
    for cat in ["false_positive", "true_positive"]:
        ws = df[(df["score_present"] == 1) & (df["category"] == cat)]["fraud_likelihood"]
        wo = df[(df["score_present"] == 0) & (df["category"] == cat)]["fraud_likelihood"]
        delta = ws.mean() - wo.mean()
        print(f"  {cat}: with={ws.mean():.3f}, without={wo.mean():.3f}, shift={delta:+.3f}")

    return {
        "shift": round(float(shift), 4),
        "cohens_d": round(float(d), 4),
        "p_value": round(float(p_val), 6),
    }


def main():
    parser = argparse.ArgumentParser(description="Transaction-level regression for score reliance")
    parser.add_argument("--with_score",
                        default=str(RESULTS_DIR / "controlled_ablation_ieee_with_score.jsonl"))
    parser.add_argument("--without_score",
                        default=str(RESULTS_DIR / "controlled_ablation_ieee_without_score.jsonl"))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    df = load_controlled_ablation(args.with_score, args.without_score)

    simple = run_simple_analysis(df)
    regression = run_ols_regression(df)

    results = {
        "simple_analysis": simple,
        "regression": regression,
        "n_observations": len(df),
        "n_transactions": len(df) // 2,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    return results


if __name__ == "__main__":
    main()
