"""Counterfactual score injection experiment.

Injects fake ML scores (0.15 for TP, 0.95 for FP) to test whether the LLM's
score anchoring is causally asymmetric: low scores suppress output even on
true fraud, but high scores do not amplify output on false positives.

Requires OPENAI_API_KEY.

Usage:
    python experiments/run_counterfactual.py \
        --input experiments/results/llm_input_ieee_cis_100.jsonl \
        --model gpt-4o-mini \
        --n_per_category 30

Outputs:
    experiments/results/causal_anchoring.json (summary)
    experiments/results/causal_anchoring_raw.jsonl (per-sample)
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy import stats as sp_stats

from evaluation.bootstrap_ci import paired_permutation_test
from llm_explainer.llm_inference import LLMConfig, LLMInference
from llm_explainer.prompts.fp_explanation import build_prompt as build_aware_prompt

RESULTS = Path(__file__).resolve().parent / "results"


def run_condition(llm: LLMInference, txs: list[dict], fake_score: float) -> list[dict]:
    """Run the aware prompt on a list of txs using a forced fake ML score."""
    results = []
    for i, tx in enumerate(txs):
        text = tx.get("text", "")
        prompt = build_aware_prompt(text, fake_score, threshold=0.5)
        from llm_explainer.prompts.fp_explanation import SYSTEM_PROMPT
        resp = llm.generate_json(prompt, SYSTEM_PROMPT)
        fl = resp.get("fraud_likelihood") if isinstance(resp, dict) else None
        results.append({
            "transaction_id": tx.get("transaction_id", i),
            "text": text,
            "category": tx.get("category", ""),
            "real_ml_score": tx.get("fraud_score", 0),
            "injected_score": fake_score,
            "fraud_likelihood": fl if isinstance(fl, (int, float)) else None,
        })
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(txs)}]")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--n_per_category", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set"); return

    with open(args.input) as f:
        all_txs = [json.loads(l) for l in f]
    fp = [t for t in all_txs if t.get("category") == "false_positive"]
    tp = [t for t in all_txs if t.get("category") == "true_positive"]
    import random
    rng = random.Random(args.seed)
    fp = rng.sample(fp, min(args.n_per_category, len(fp)))
    tp = rng.sample(tp, min(args.n_per_category, len(tp)))

    cfg = LLMConfig(model_name=args.model, backend="openai", temperature=0.0)
    llm = LLMInference(cfg); llm.load()

    print(f"Running counterfactual: FP n={len(fp)}, TP n={len(tp)}")

    # Baseline: real score (each tx uses its own ML score)
    def run_with_real_scores(txs):
        out = []
        for i, tx in enumerate(txs):
            text = tx.get("text", "")
            real = tx.get("fraud_score", 0.5)
            prompt = build_aware_prompt(text, real, threshold=0.5)
            from llm_explainer.prompts.fp_explanation import SYSTEM_PROMPT
            resp = llm.generate_json(prompt, SYSTEM_PROMPT)
            fl = resp.get("fraud_likelihood") if isinstance(resp, dict) else None
            out.append({
                "transaction_id": tx.get("transaction_id", i),
                "category": tx.get("category", ""),
                "real_ml_score": real, "injected_score": real,
                "condition": "real",
                "fraud_likelihood": fl if isinstance(fl, (int, float)) else None,
            })
            if (i + 1) % 10 == 0: print(f"  [{i+1}/{len(txs)}]")
        return out

    print("--- TP real ---"); tp_real = run_with_real_scores(tp)
    print("--- FP real ---"); fp_real = run_with_real_scores(fp)

    print("\n--- TP fake LOW (0.15) ---"); tp_fake = run_condition(llm, tp, 0.15)
    for r in tp_fake: r["condition"] = "fake_low"
    print("--- FP fake HIGH (0.95) ---"); fp_fake = run_condition(llm, fp, 0.95)
    for r in fp_fake: r["condition"] = "fake_high"

    # Save raw
    RESULTS.mkdir(parents=True, exist_ok=True)
    raw = tp_real + fp_real + tp_fake + fp_fake
    with open(RESULTS / "causal_anchoring_raw.jsonl", "w") as f:
        for r in raw:
            f.write(json.dumps(r) + "\n")

    # Paired analysis: same tx at real vs fake
    def fls(lst): return [x["fraud_likelihood"] for x in lst
                          if isinstance(x.get("fraud_likelihood"), (int, float))]
    tp_real_fls = fls(tp_real); tp_fake_fls = fls(tp_fake)
    fp_real_fls = fls(fp_real); fp_fake_fls = fls(fp_fake)

    n_tp = min(len(tp_real_fls), len(tp_fake_fls))
    n_fp = min(len(fp_real_fls), len(fp_fake_fls))

    tp_paired = paired_permutation_test(
        tp_fake_fls[:n_tp], tp_real_fls[:n_tp]
    )
    fp_paired = paired_permutation_test(
        fp_fake_fls[:n_fp], fp_real_fls[:n_fp]
    )

    # Interaction test: |Δ_TP| - |Δ_FP|
    delta_tp = np.array([tp_fake_fls[i] - tp_real_fls[i] for i in range(n_tp)])
    delta_fp = np.array([fp_fake_fls[i] - fp_real_fls[i] for i in range(n_fp)])

    from evaluation.bootstrap_ci import two_sample_permutation_test
    interaction = two_sample_permutation_test(
        list(np.abs(delta_tp)), list(np.abs(delta_fp))
    )

    summary = {
        "n_tp": n_tp,
        "n_fp": n_fp,
        "tp": {
            "real_mean": round(float(np.mean(tp_real_fls[:n_tp])), 4),
            "fake_low_mean": round(float(np.mean(tp_fake_fls[:n_tp])), 4),
            "delta": round(float(np.mean(delta_tp)), 4),
            "paired_p_value": tp_paired["p_value"],
        },
        "fp": {
            "real_mean": round(float(np.mean(fp_real_fls[:n_fp])), 4),
            "fake_high_mean": round(float(np.mean(fp_fake_fls[:n_fp])), 4),
            "delta": round(float(np.mean(delta_fp)), 4),
            "paired_p_value": fp_paired["p_value"],
        },
        "interaction": {
            "abs_delta_tp_minus_abs_delta_fp": round(
                float(np.mean(np.abs(delta_tp)) - np.mean(np.abs(delta_fp))), 4),
            "p_value": interaction["p_value"],
        },
    }

    with open(RESULTS / "causal_anchoring.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {RESULTS / 'causal_anchoring.json'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
