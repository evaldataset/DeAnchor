"""Blind LLM-as-Judge + Inter-rater reliability.

CODE-005: Ground truth 없이 설명 품질 평가
CODE-006: 3회 반복으로 inter-rater agreement 측정
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_explainer.llm_inference import LLMConfig, LLMInference

RESULTS_DIR = Path(__file__).resolve().parent.parent / "experiments" / "results"

JUDGE_SYSTEM = """You are an expert evaluator assessing the quality of financial fraud explanations. Rate each explanation on its own merits.

Rating Scale:
1 = Very Poor: Incoherent or irrelevant
2 = Poor: Major logical errors or missing key information
3 = Adequate: Generally reasonable but lacks depth
4 = Good: Well-reasoned and useful with minor gaps
5 = Excellent: Comprehensive, logical, and actionable"""

JUDGE_BLIND = """Evaluate the following fraud explanation. You do NOT know whether this transaction is actually fraudulent.

## Transaction
{transaction_text}

## AI-Generated Explanation
{explanation}

Rate each dimension 1-5:
1. **Coherence**: Is the reasoning logical and internally consistent?
2. **Completeness**: Does the explanation address all relevant aspects?
3. **Clarity**: Is it clear, well-structured, and professional?
4. **Actionability**: Does it provide useful guidance for an investigator?

Respond in JSON:
```json
{{
    "coherence": 1-5,
    "completeness": 1-5,
    "clarity": 1-5,
    "actionability": 1-5,
    "overall": 1-5,
    "strengths": ["..."],
    "weaknesses": ["..."]
}}
```"""

JUDGE_WITH_GT = """Evaluate this fraud explanation against ground truth.

## Transaction
{transaction_text}

## Ground Truth: {ground_truth_label}

## AI-Generated Explanation
{explanation}

Rate 1-5:
1. **Factual Accuracy**: Given the ground truth, is the explanation correct?
2. **Completeness**: Does it cover relevant aspects?
3. **Clarity**: Is it clear and professional?
4. **Actionability**: Does it provide useful guidance?

Respond in JSON:
```json
{{
    "factual_accuracy": 1-5,
    "completeness": 1-5,
    "clarity": 1-5,
    "actionability": 1-5,
    "overall": 1-5
}}
```"""


def run_judge(
    predictions_path: str,
    model: str = "gpt-4o-mini",
    max_samples: int = 30,
    n_runs: int = 3,
) -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set")
        return

    config = LLMConfig(model_name=model, backend="openai", temperature=0.0)
    llm = LLMInference(config)
    llm.load()

    with open(predictions_path) as f:
        preds = [json.loads(line) for line in f][:max_samples]

    print(f"Evaluating {len(preds)} explanations, {n_runs} runs each")

    all_runs_blind = []
    all_runs_gt = []

    for run_idx in range(n_runs):
        print(f"\n--- Run {run_idx+1}/{n_runs} ---")
        run_blind = []
        run_gt = []

        for i, pred in enumerate(preds):
            tx = pred.get("original", {})
            exp = pred.get("anomaly_explanation") or pred.get("fp_explanation", {})
            if exp.get("parse_error"):
                continue

            exp_text = json.dumps(exp, indent=2) if isinstance(exp, dict) else str(exp)
            tx_text = tx.get("text", "")

            # Blind evaluation
            prompt = JUDGE_BLIND.format(
                transaction_text=tx_text, explanation=exp_text,
            )
            result = llm.generate_json(prompt, JUDGE_SYSTEM)
            run_blind.append(result)

            # With ground truth
            gt_label = "Fraudulent" if tx.get("is_fraud", 0) == 1 else "Legitimate"
            prompt_gt = JUDGE_WITH_GT.format(
                transaction_text=tx_text,
                ground_truth_label=gt_label,
                explanation=exp_text,
            )
            result_gt = llm.generate_json(prompt_gt, JUDGE_SYSTEM)
            run_gt.append(result_gt)

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(preds)}]")

        all_runs_blind.append(run_blind)
        all_runs_gt.append(run_gt)

    # Aggregate results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Blind scores
    dims_blind = ["coherence", "completeness", "clarity", "actionability", "overall"]
    print("\n=== BLIND Evaluation (no ground truth) ===")
    blind_summary = {}
    for dim in dims_blind:
        all_scores = []
        per_run = []
        for run in all_runs_blind:
            run_scores = [r.get(dim, 0) for r in run if isinstance(r.get(dim), (int, float))]
            per_run.append(np.mean(run_scores) if run_scores else 0)
            all_scores.extend(run_scores)
        mean = np.mean(all_scores) if all_scores else 0
        std = np.std(per_run) if len(per_run) > 1 else 0
        blind_summary[dim] = {"mean": round(mean, 2), "std_across_runs": round(std, 3), "n": len(all_scores)}
        print(f"  {dim:15s}: {mean:.2f} ± {std:.3f} (n={len(all_scores)})")

    # GT scores
    dims_gt = ["factual_accuracy", "completeness", "clarity", "actionability", "overall"]
    print("\n=== WITH Ground Truth Evaluation ===")
    gt_summary = {}
    for dim in dims_gt:
        all_scores = []
        per_run = []
        for run in all_runs_gt:
            run_scores = [r.get(dim, 0) for r in run if isinstance(r.get(dim), (int, float))]
            per_run.append(np.mean(run_scores) if run_scores else 0)
            all_scores.extend(run_scores)
        mean = np.mean(all_scores) if all_scores else 0
        std = np.std(per_run) if len(per_run) > 1 else 0
        gt_summary[dim] = {"mean": round(mean, 2), "std_across_runs": round(std, 3), "n": len(all_scores)}
        print(f"  {dim:15s}: {mean:.2f} ± {std:.3f} (n={len(all_scores)})")

    # Inter-rater agreement (Krippendorff's alpha approximation via pairwise correlation)
    print("\n=== Inter-Run Agreement ===")
    for dim in dims_blind:
        run_vectors = []
        for run in all_runs_blind:
            vec = [r.get(dim, 0) for r in run if isinstance(r.get(dim), (int, float))]
            run_vectors.append(vec)

        if len(run_vectors) >= 2 and all(len(v) == len(run_vectors[0]) for v in run_vectors):
            # Pairwise correlations
            from itertools import combinations
            corrs = []
            for r1, r2 in combinations(range(len(run_vectors)), 2):
                corr = np.corrcoef(run_vectors[r1], run_vectors[r2])[0, 1]
                corrs.append(corr)
            mean_corr = np.mean(corrs) if corrs else 0
            # Exact agreement
            agreements = sum(
                1 for j in range(len(run_vectors[0]))
                if all(run_vectors[r][j] == run_vectors[0][j] for r in range(len(run_vectors)))
            )
            exact_pct = agreements / len(run_vectors[0]) if run_vectors[0] else 0
            print(f"  {dim:15s}: corr={mean_corr:.3f}, exact_agree={exact_pct:.0%}")

    # Save
    output = {
        "blind_evaluation": blind_summary,
        "gt_evaluation": gt_summary,
        "n_runs": n_runs,
        "n_samples": len(preds),
        "model": model,
    }
    out_path = RESULTS_DIR / "judge_evaluation_final.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max_samples", type=int, default=30)
    parser.add_argument("--n_runs", type=int, default=3)
    args = parser.parse_args()

    run_judge(args.predictions, args.model, args.max_samples, args.n_runs)


if __name__ == "__main__":
    main()
