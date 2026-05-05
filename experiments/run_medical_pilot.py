"""Medical risk assessment cross-domain pilot.

Constructs synthetic patient profiles with interpretable features (age, BP,
cholesterol, BMI) and runs a controlled score-aware vs score-blind ablation
analogous to the fraud setting, to test whether anchoring is weaker when
features are semantically rich.

This is a PILOT with n=30 synthetic patients — not real clinical data.
Results should be interpreted as exploratory evidence for a feature-
informativeness hypothesis, not as a validated cross-domain claim.

Usage:
    python experiments/run_medical_pilot.py --model gpt-4o-mini

Outputs:
    experiments/results/medical_domain.json
    experiments/results/medical_pilot_raw.jsonl
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from evaluation.bootstrap_ci import bootstrap_ci, two_sample_permutation_test
from llm_explainer.llm_inference import LLMConfig, LLMInference

RESULTS = Path(__file__).resolve().parent / "results"

SYSTEM = (
    "You are a medical risk assessor. Evaluate whether the patient is at high "
    "risk of cardiovascular events based on the profile."
)

TEMPLATE_WITH_SCORE = """Assess cardiovascular risk for this patient.

## Patient Profile
{profile}

## ML Risk Model Score
Risk probability: {score:.4f}

Respond in JSON:
```json
{{"risk_likelihood": 0.0-1.0, "recommendation": "HIGH_RISK|MODERATE|LOW_RISK"}}
```"""

TEMPLATE_NO_SCORE = """Assess cardiovascular risk for this patient.

## Patient Profile
{profile}

Respond in JSON:
```json
{{"risk_likelihood": 0.0-1.0, "recommendation": "HIGH_RISK|MODERATE|LOW_RISK"}}
```"""


def make_profile(is_high_risk: bool, rng: random.Random) -> tuple[str, float]:
    """Generate a synthetic patient profile with a plausible ML risk score."""
    if is_high_risk:
        age = rng.randint(60, 82)
        bp_sys = rng.randint(150, 180)
        bp_dia = rng.randint(95, 110)
        chol = rng.randint(240, 290)
        bmi = round(rng.uniform(30, 38), 1)
        score = round(rng.uniform(0.75, 0.92), 4)
    else:
        age = rng.randint(30, 55)
        bp_sys = rng.randint(110, 135)
        bp_dia = rng.randint(70, 85)
        chol = rng.randint(170, 210)
        bmi = round(rng.uniform(21, 27), 1)
        score = round(rng.uniform(0.75, 0.92), 4)  # same range → anchor test

    profile = (
        f"Age: {age} years\n"
        f"Blood pressure: {bp_sys}/{bp_dia} mmHg\n"
        f"Total cholesterol: {chol} mg/dL\n"
        f"BMI: {bmi}\n"
        f"Smoking: {'yes' if is_high_risk else 'no'}\n"
        f"Family history of CVD: {'yes' if is_high_risk else 'no'}"
    )
    return profile, score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--n_per_class", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set"); return

    rng = random.Random(args.seed)

    # Generate synthetic patients
    patients = []
    for _ in range(args.n_per_class):
        profile, score = make_profile(True, rng)
        patients.append({"profile": profile, "score": score, "is_high_risk": True})
    for _ in range(args.n_per_class):
        profile, score = make_profile(False, rng)
        patients.append({"profile": profile, "score": score, "is_high_risk": False})
    rng.shuffle(patients)

    cfg = LLMConfig(model_name=args.model, backend="openai", temperature=0.0)
    llm = LLMInference(cfg); llm.load()

    raw = []
    for cond_name, template in [("with_score", TEMPLATE_WITH_SCORE),
                                ("without_score", TEMPLATE_NO_SCORE)]:
        print(f"\n--- {cond_name} ---")
        for i, p in enumerate(patients):
            prompt = template.format(profile=p["profile"], score=p["score"])
            resp = llm.generate_json(prompt, SYSTEM)
            fl = resp.get("risk_likelihood") if isinstance(resp, dict) else None
            raw.append({
                "patient_index": i, "condition": cond_name,
                "is_high_risk": p["is_high_risk"], "ml_score": p["score"],
                "risk_likelihood": fl if isinstance(fl, (int, float)) else None,
            })
            if (i + 1) % 10 == 0: print(f"  [{i+1}/{len(patients)}]")

    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "medical_pilot_raw.jsonl", "w") as f:
        for r in raw:
            f.write(json.dumps(r) + "\n")

    # Compute alpha
    def get_fl(cond, high):
        return [r["risk_likelihood"] for r in raw
                if r["condition"] == cond and r["is_high_risk"] == high
                and isinstance(r.get("risk_likelihood"), (int, float))]

    ws_low = get_fl("with_score", False)
    ws_high = get_fl("with_score", True)
    wos_low = get_fl("without_score", False)
    wos_high = get_fl("without_score", True)

    ws_mean = np.mean(ws_low + ws_high) if (ws_low + ws_high) else 0
    wos_mean = np.mean(wos_low + wos_high) if (wos_low + wos_high) else 0
    ml_mean = np.mean([r["ml_score"] for r in raw if r["condition"] == "with_score"])
    alpha = (ws_mean - wos_mean) / ml_mean if ml_mean > 0 else 0

    summary = {
        "domain": "medical_risk_synthetic",
        "n_per_class": args.n_per_class,
        "alpha": round(float(alpha), 4),
        "with_score": {
            "low_risk": round(float(np.mean(ws_low)) if ws_low else 0, 4),
            "high_risk": round(float(np.mean(ws_high)) if ws_high else 0, 4),
        },
        "without_score": {
            "low_risk": round(float(np.mean(wos_low)) if wos_low else 0, 4),
            "high_risk": round(float(np.mean(wos_high)) if wos_high else 0, 4),
        },
        "note": "SYNTHETIC data; pilot evidence only; not validated clinical population",
    }

    with open(RESULTS / "medical_domain.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {RESULTS / 'medical_domain.json'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
