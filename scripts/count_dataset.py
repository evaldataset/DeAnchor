"""Count released LLM-response dataset for the Datasheet (Appendix B).

Walks experiments/results/ and emits:
  - total LLM responses (parseable, valid)
  - paired score-aware/score-blind matched pairs
  - per-experiment counts

Usage:
    python scripts/count_dataset.py [--output dataset_card.json]
"""

import argparse
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "experiments" / "results"

EXP_GROUPS = {
    "controlled_ablation_ieee": [
        "controlled_ablation_ieee_with_score.jsonl",
        "controlled_ablation_ieee_without_score.jsonl",
    ],
    "scoreaware_paysim": [
        "scoreaware_paysim.jsonl",
        "scoreblind_paysim.jsonl",
    ],
    "explanations_ieee": [
        "explanations_ieee_cis_gpt4o.jsonl",
        "explanations_ieee_cis_mixed.jsonl",
        "explanations_ieee_cis_fixed.jsonl",
    ],
    "explanations_paysim": [
        "explanations_paysim_50.jsonl",
        "explanations_paysim_gpt4o.jsonl",
        "explanations_paysim_mixed.jsonl",
    ],
    "dose_response": ["dose_response.jsonl"],
    "rich_features": [
        "scoreaware_rich.jsonl", "scoreblind_rich.jsonl",
        "scoreaware_rich_gemma.jsonl", "scoreblind_rich_gemma.jsonl",
        "scoreaware_rich_gpt4omini.jsonl", "scoreblind_rich_gpt4omini.jsonl",
    ],
    "medical": [
        "medical_local_raw.jsonl", "medical_n100_raw.jsonl",
    ],
    "extension_battery": [
        "mitigation_zoo_raw.jsonl",
        "prior_dose_response_raw.jsonl",
        "prompt_paraphrase_raw.jsonl",
        "score_format_raw.jsonl",
    ],
    "reasoning_models": [
        "ablation_o1mini.json", "ablation_gem_pro_thinking.json",
        "ablation_o3mini_n50.json", "ablation_gem_pro_thinking_n50.json",
    ],
    "cross_model_extended": [
        "ablation_claude_haiku.json", "ablation_gemini_flash.json",
    ],
    "mechanism_priming": ["mechanism_raw.jsonl"],
    "multiseed": ["multiseed_ablation_audit.json"],
    "paysim_singlemodel": ["paysim_singlemodel_raw.jsonl"],
    "k2_stratified_mitigation": ["mitigation_stratified_raw.jsonl"],
}


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def count_paired_records(path: Path, fields: tuple = ("fl_aware", "fl_blind")) -> int:
    """Count records with both paired fields present and parseable as numeric."""
    if not path.exists():
        return 0
    n = 0
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if all(isinstance(r.get(k), (int, float)) for k in fields):
            n += 1
    return n


def count_llm_responses(path: Path) -> int:
    """Count records with at least one parseable LLM-output field."""
    if not path.exists():
        return 0
    n = 0
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        candidates = []
        # Direct top-level fields
        if isinstance(r.get("fraud_likelihood"), (int, float)):
            candidates.append(r["fraud_likelihood"])
        if isinstance(r.get("risk_likelihood"), (int, float)):
            candidates.append(r["risk_likelihood"])
        # Nested under {assessment, blind_assessment, fp_explanation,
        #               anomaly_explanation, fraud_classification, ...}
        for sub in ("assessment", "blind_assessment", "fp_explanation",
                    "anomaly_explanation", "fraud_classification",
                    "stage1_blind", "stage2_informed"):
            v = r.get(sub)
            if isinstance(v, dict) and isinstance(v.get("fraud_likelihood"), (int, float)):
                candidates.append(v["fraud_likelihood"])
        # Top-level fl_* (from extension battery)
        for k in ("fl_aware", "fl_blind", "fl_base", "fl_normal", "fl_primed",
                  "fl_p0", "fl_p1", "fl_decimal", "fl_quantize", "fl_bayesian",
                  "fl_adversary", "fl_precommit", "fl_ensemble",
                  "initial_fl", "final_fl"):
            if isinstance(r.get(k), (int, float)):
                candidates.append(r[k])
        # responses dict (dose response)
        if isinstance(r.get("responses"), dict):
            for v in r["responses"].values():
                if isinstance(v, dict) and isinstance(v.get("fraud_likelihood"), (int, float)):
                    candidates.append(v["fraud_likelihood"])
        if any(c is not None for c in candidates):
            n += 1
    return n


def count_all() -> dict:
    out = {"by_experiment": {}, "totals": {}}
    grand_total = 0
    paired_total = 0

    # Auto-enumerate all *.jsonl response-like files (excluding inputs and audit fixtures)
    SKIP = {"llm_input_", "audit_fixture_"}
    by_file: dict[str, int] = {}
    for path in sorted(RESULTS.glob("*.jsonl")):
        if any(path.name.startswith(p) for p in SKIP):
            continue
        n = count_llm_responses(path)
        if n > 0:
            by_file[path.name] = n
            grand_total += n

    # Paired pairs: count by transaction_id intersection across canonical aware/blind pairs
    aware_path = RESULTS / "controlled_ablation_ieee_with_score.jsonl"
    blind_path = RESULTS / "controlled_ablation_ieee_without_score.jsonl"
    if aware_path.exists() and blind_path.exists():
        aware_ids = set()
        for line in open(aware_path):
            try:
                r = json.loads(line)
                aware_ids.add(r["original"].get("transaction_id"))
            except Exception:
                continue
        blind_ids = set()
        for line in open(blind_path):
            try:
                r = json.loads(line)
                blind_ids.add(r["original"].get("transaction_id"))
            except Exception:
                continue
        paired_total += len(aware_ids & blind_ids)
    # Other paired sources
    for fname in ("paysim_singlemodel_raw.jsonl", "audit_fixture_ieee_50.jsonl"):
        p = RESULTS / fname
        if p.exists():
            paired_total += count_paired_records(p, fields=("fl_aware", "fl_blind"))
    for fname in ("scoreaware_paysim.jsonl",):
        p_aware = RESULTS / fname
        p_blind = RESULTS / fname.replace("aware", "blind")
        if p_aware.exists() and p_blind.exists():
            # paired by transaction order
            paired_total += min(count_jsonl(p_aware), count_jsonl(p_blind))
    # explanations + scoreblind paired (200)
    for fname in ("scoreaware_ieee_cis_200.jsonl",):
        p_aware = RESULTS / fname
        p_blind = RESULTS / "scoreblind_ieee_cis_200_final.jsonl"
        if p_aware.exists() and p_blind.exists():
            paired_total += min(count_jsonl(p_aware), count_jsonl(p_blind))
    # rich features
    for fname in ("rich_features_raw.jsonl",):
        p = RESULTS / fname
        if p.exists():
            paired_total += count_jsonl(p)  # paired internally

    # Deduplicate: don't double-count paired with grand_total
    paired_total = min(paired_total, grand_total)

    out["by_file"] = by_file
    out["totals"] = {
        "total_records": grand_total,
        "total_paired": paired_total,
        "approximate": True,
    }
    out["dataset_card_summary"] = {
        "approx_total_responses": round(grand_total, -2),
        "approx_paired": round(paired_total, -2),
        "n_models": 8,
        "n_feature_regimes": 4,
    }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=RESULTS / "dataset_card.json")
    args = p.parse_args()

    counts = count_all()
    args.output.write_text(json.dumps(counts, indent=2))

    print("=" * 70)
    print("Released LLM-response Dataset Card")
    print("=" * 70)
    for fname, n in sorted(counts["by_file"].items(), key=lambda x: -x[1]):
        print(f"  {fname:55s}  n={n:5d}")
    print("-" * 70)
    print(f"  {'TOTAL records':55s}  n={counts['totals']['total_records']:5d}")
    print(f"  {'TOTAL paired':55s}  n={counts['totals']['total_paired']:5d}")
    print()
    print(f"Approximate dataset-card claim: ~{counts['dataset_card_summary']['approx_total_responses']} responses, "
          f"~{counts['dataset_card_summary']['approx_paired']} paired.")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
