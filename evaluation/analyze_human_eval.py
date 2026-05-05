"""Human evaluation 결과 분석.

Google Forms CSV 응답을 분석하여 논문에 보고할 통계를 생성.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_responses(csv_path: str) -> pd.DataFrame:
    """Google Forms CSV 응답 로드.

    Expected columns (after manual mapping):
    - evaluator_id: 평가자 식별 (순번)
    - Q01_coherence ~ Q30_coherence: 각 거래별 일관성 점수 (1-5)
    - Q01_completeness ~ Q30_completeness
    - Q01_clarity ~ Q30_clarity
    - Q01_actionability ~ Q30_actionability
    - Q01_overall ~ Q30_overall
    - Q01_trust ~ Q30_trust: Yes/Partially/No
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} evaluator responses")
    return df


def analyze_scores(df: pd.DataFrame, items_path: str) -> dict:
    """점수 분석: 차원별 평균, CI, 카테고리별 비교."""
    with open(items_path) as f:
        items = json.load(f)

    # Build item metadata
    item_meta = {}
    for it in items:
        item_meta[it["id"]] = {
            "category": it["category"],
            "ml_score": it["ml_score"],
        }

    dimensions = ["coherence", "completeness", "clarity", "actionability", "overall"]
    results = {"dimensions": {}, "by_category": {}, "inter_rater": {}}

    # Dimension-level analysis
    print("\n=== Dimension-Level Scores ===")
    for dim in dimensions:
        cols = [c for c in df.columns if dim in c.lower()]
        if not cols:
            print(f"  {dim}: no columns found, skipping")
            continue

        all_scores = df[cols].values.flatten()
        all_scores = all_scores[~np.isnan(all_scores)]

        mean = np.mean(all_scores)
        std = np.std(all_scores)
        n = len(all_scores)

        # Bootstrap CI
        rng = np.random.RandomState(42)
        boot_means = [np.mean(rng.choice(all_scores, n, replace=True)) for _ in range(10000)]
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

        results["dimensions"][dim] = {
            "mean": round(float(mean), 2),
            "std": round(float(std), 2),
            "ci_lower": round(float(ci_lo), 2),
            "ci_upper": round(float(ci_hi), 2),
            "n": int(n),
        }
        print(f"  {dim:15s}: {mean:.2f} ± {std:.2f}  [{ci_lo:.2f}, {ci_hi:.2f}]  (n={n})")

    # Category-level (FP vs TP)
    print("\n=== FP vs TP Comparison (Overall) ===")
    for cat in ["false_positive", "true_positive"]:
        cat_ids = [qid for qid, meta in item_meta.items() if meta["category"] == cat]
        cat_cols = []
        for qid in cat_ids:
            matching = [c for c in df.columns if qid.lower() in c.lower() and "overall" in c.lower()]
            cat_cols.extend(matching)

        if cat_cols:
            scores = df[cat_cols].values.flatten()
            scores = scores[~np.isnan(scores)]
            mean = np.mean(scores)
            results["by_category"][cat] = {"mean": round(float(mean), 2), "n": int(len(scores))}
            print(f"  {cat}: {mean:.2f} (n={len(scores)})")

    # Inter-rater agreement (Krippendorff's alpha approximation)
    print("\n=== Inter-Rater Agreement ===")
    overall_cols = [c for c in df.columns if "overall" in c.lower()]
    if len(df) >= 2 and overall_cols:
        from itertools import combinations
        corrs = []
        for r1, r2 in combinations(range(len(df)), 2):
            v1 = df.iloc[r1][overall_cols].values.astype(float)
            v2 = df.iloc[r2][overall_cols].values.astype(float)
            mask = ~(np.isnan(v1) | np.isnan(v2))
            if mask.sum() > 5:
                corr = np.corrcoef(v1[mask], v2[mask])[0, 1]
                if not np.isnan(corr):
                    corrs.append(corr)
        if corrs:
            mean_corr = np.mean(corrs)
            results["inter_rater"]["mean_correlation"] = round(float(mean_corr), 3)
            results["inter_rater"]["n_pairs"] = len(corrs)
            print(f"  Mean pairwise correlation: {mean_corr:.3f} ({len(corrs)} pairs)")

    # Trust distribution
    print("\n=== Trust Distribution ===")
    trust_cols = [c for c in df.columns if "trust" in c.lower()]
    if trust_cols:
        all_trust = df[trust_cols].values.flatten()
        all_trust = [str(t) for t in all_trust if pd.notna(t)]
        from collections import Counter
        trust_counts = Counter(all_trust)
        total = len(all_trust)
        results["trust"] = {}
        for label, count in trust_counts.most_common():
            pct = count / total * 100
            results["trust"][label] = {"count": count, "percent": round(pct, 1)}
            print(f"  {label}: {count} ({pct:.1f}%)")

    return results


def generate_latex_table(results: dict) -> str:
    """논문 삽입용 LaTeX 테이블 생성."""
    dims = results.get("dimensions", {})
    rows = []
    for dim, vals in dims.items():
        rows.append(
            f"{dim.capitalize()} & {vals['mean']:.2f} & "
            f"[{vals['ci_lower']:.2f}, {vals['ci_upper']:.2f}] & {vals['n']}"
        )

    table = r"""\begin{table}[t]
\centering\small
\caption{Human expert evaluation of LLM-generated fraud explanations (1--5 scale, $n$ evaluators $\times$ 30 transactions).}
\label{tab:human_eval}
\begin{tabular}{lccc}
\toprule
Dimension & Mean & 95\% CI & $n$ \\
\midrule
""" + " \\\\\n".join(rows) + r""" \\
\bottomrule
\end{tabular}
\end{table}"""
    return table


def main():
    parser = argparse.ArgumentParser(description="Analyze human evaluation responses")
    parser.add_argument("--responses", required=True, help="Path to Google Forms CSV export")
    parser.add_argument("--items", default="evaluation/human_eval_items.json")
    parser.add_argument("--output", default=None, help="Save results JSON")
    args = parser.parse_args()

    df = load_responses(args.responses)
    results = analyze_scores(df, args.items)

    print("\n=== LaTeX Table ===")
    print(generate_latex_table(results))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
