"""Multi-persona Human Evaluation.

5명의 다른 fraud analyst 페르소나로 평가.
SHAP vs LLM vs SHAP+LLM 3-way 비교.
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

PERSONAS = [
    {"name": "Senior Analyst (15yr)", "system": "You are a senior fraud analyst with 15 years at a major bank. You value precision and quantified evidence. You are skeptical of AI-generated narratives."},
    {"name": "Junior Analyst (2yr)", "system": "You are a junior fraud analyst with 2 years of experience. You appreciate clear explanations that help you learn. You find detailed narratives more helpful than raw numbers."},
    {"name": "Compliance Officer", "system": "You are a BSA/AML compliance officer. You need explanations that can be included in regulatory filings (SARs). You value structured, legally defensible language."},
    {"name": "Team Lead", "system": "You are a fraud investigation team lead managing 10 analysts. You need explanations that help you prioritize cases and allocate resources efficiently."},
    {"name": "Risk Manager", "system": "You are a risk management executive. You need high-level summaries that support strategic decisions about fraud detection systems and resource allocation."},
]

EVAL_PROMPT = """You are evaluating THREE explanation approaches for the SAME flagged transaction. Rate each approach independently.

## Transaction
{transaction_text}
ML Fraud Score: {ml_score:.4f}

## Approach A: SHAP Feature Attribution
{shap_explanation}

## Approach B: LLM Narrative
{llm_explanation}

## Approach C: SHAP + LLM Combined
{combined_explanation}

Rate EACH approach 1-5 on: Usefulness, Clarity, Actionability, Trust.
Then rank them 1st/2nd/3rd.

```json
{{
    "approach_a": {{"usefulness": N, "clarity": N, "actionability": N, "trust": N}},
    "approach_b": {{"usefulness": N, "clarity": N, "actionability": N, "trust": N}},
    "approach_c": {{"usefulness": N, "clarity": N, "actionability": N, "trust": N}},
    "ranking": ["A/B/C", "A/B/C", "A/B/C"],
    "brief_rationale": "1 sentence"
}}
```"""


def run_multi_persona(
    shap_path: str, llm_path: str, model: str = "gpt-4o-mini", n_samples: int = 20,
):
    config = LLMConfig(model_name=model, backend="openai", temperature=0.3)
    llm = LLMInference(config)
    llm.load()

    # Load SHAP explanations
    with open(shap_path) as f:
        shap_data = json.load(f)

    # Load LLM explanations
    llm_exps = {}
    with open(llm_path) as f:
        for line in f:
            d = json.loads(line)
            score = round(d.get("ml_score", 0), 4)
            exp = d.get("fp_explanation", {})
            llm_exps[score] = json.dumps(exp, indent=1)[:500] if isinstance(exp, dict) else str(exp)[:500]

    samples = shap_data[:n_samples]
    all_results = []

    for persona in PERSONAS:
        print(f"\n=== Persona: {persona['name']} ===")
        persona_results = []

        for i, s in enumerate(samples):
            # Match LLM explanation
            best_key = min(llm_exps.keys(), key=lambda k: abs(k - round(s["fraud_score"], 4)), default=None)
            llm_exp = llm_exps.get(best_key, "N/A")

            # Combined explanation
            combined = f"QUANTITATIVE EVIDENCE:\n{s['shap_explanation']}\n\nNARRATIVE CONTEXT:\n{llm_exp[:300]}"

            prompt = EVAL_PROMPT.format(
                transaction_text=s["transaction_text"][:250],
                ml_score=s["fraud_score"],
                shap_explanation=s["shap_explanation"],
                llm_explanation=llm_exp[:400],
                combined_explanation=combined[:500],
            )

            result = llm.generate_json(prompt, persona["system"])
            result["persona"] = persona["name"]
            result["sample_idx"] = i
            result["category"] = s["category"]
            persona_results.append(result)

        all_results.extend(persona_results)
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(samples)}]")

    # Aggregate
    print(f"\n{'='*60}")
    print(f"MULTI-PERSONA EVALUATION RESULTS ({len(PERSONAS)} personas × {n_samples} samples)")
    print(f"{'='*60}")

    for approach, key in [("A (SHAP)", "approach_a"), ("B (LLM)", "approach_b"), ("C (Combined)", "approach_c")]:
        scores = []
        for r in all_results:
            a = r.get(key, {})
            if isinstance(a, dict):
                for dim in ["usefulness", "clarity", "actionability", "trust"]:
                    v = a.get(dim)
                    if isinstance(v, (int, float)):
                        scores.append(v)
        mean = np.mean(scores) if scores else 0
        print(f"  {approach:15s}: {mean:.2f}/5 (n={len(scores)})")

    # Ranking analysis
    rankings = {"A": [], "B": [], "C": []}
    for r in all_results:
        rank = r.get("ranking", [])
        if isinstance(rank, list) and len(rank) == 3:
            for pos, choice in enumerate(rank):
                choice = choice.strip().upper()
                if choice in rankings:
                    rankings[choice].append(pos + 1)

    print(f"\n  Rankings (mean position, lower=better):")
    for approach, positions in rankings.items():
        if positions:
            print(f"    {approach}: {np.mean(positions):.2f} (1st={positions.count(1)}, 2nd={positions.count(2)}, 3rd={positions.count(3)})")

    # Per-persona breakdown
    print(f"\n  Per-Persona Overall Scores:")
    for persona in PERSONAS:
        p_results = [r for r in all_results if r.get("persona") == persona["name"]]
        for approach, key in [("A", "approach_a"), ("B", "approach_b"), ("C", "approach_c")]:
            scores = []
            for r in p_results:
                a = r.get(key, {})
                if isinstance(a, dict):
                    for dim in ["usefulness", "clarity", "actionability", "trust"]:
                        v = a.get(dim)
                        if isinstance(v, (int, float)):
                            scores.append(v)
            mean = np.mean(scores) if scores else 0
            print(f"    {persona['name']:25s} | {approach}: {mean:.1f}", end="")
        print()

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "multi_persona_eval.json"
    with open(out_path, "w") as f:
        json.dump({"results": all_results, "personas": [p["name"] for p in PERSONAS]}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shap", required=True)
    parser.add_argument("--llm", required=True)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--n_samples", type=int, default=20)
    args = parser.parse_args()
    run_multi_persona(args.shap, args.llm, args.model, args.n_samples)


if __name__ == "__main__":
    main()
