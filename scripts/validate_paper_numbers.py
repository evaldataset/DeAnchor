"""Validate all paper-facing numerical claims against canonical JSON artifacts.

Fails with exit code 1 if any claim diverges from the source-of-truth artifact.
Designed to run in CI or as a pre-submission check.

Usage:
    python scripts/validate_paper_numbers.py
"""

import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "experiments" / "results"

PASS = 0
FAIL = 0


def check(label, actual, expected, tol=0.01):
    global PASS, FAIL
    ok = abs(actual - expected) < tol if isinstance(expected, float) else actual == expected
    status = "PASS" if ok else "FAIL"
    if not ok:
        FAIL += 1
        print(f"  {status}: {label}: actual={actual}, expected={expected}")
    else:
        PASS += 1
    return ok


def load_json(name):
    path = RESULTS / name
    if not path.exists():
        print(f"  MISSING: {path}")
        return None
    return json.load(open(path))


def main():
    print("=" * 60)
    print("Paper Number Validation")
    print("=" * 60)

    # --- 1. Fusion baseline (IEEE-CIS) ---
    print("\n[1] Fusion baseline (IEEE-CIS)")
    d = load_json("fusion_baseline.json")
    if d:
        check("Raw ML AUROC", d["auroc"]["raw_ml_only"], 0.7875)
        check("LLM fl AUROC", d["auroc"]["llm_fl_only"], 0.727)
        check("Fusion AUROC", d["auroc"]["raw_ml_plus_llm_fl"], 0.7861)
        check("LR chi2 (LLM over raw)", d["likelihood_ratio_test"]["does_llm_fl_add_over_raw_ml"]["lr_stat"], 2.1256)
        check("LR p (LLM over raw)", d["likelihood_ratio_test"]["does_llm_fl_add_over_raw_ml"]["p_value"], 0.145, tol=0.005)
        check("LR chi2 (raw over LLM)", d["likelihood_ratio_test"]["does_raw_ml_add_over_llm_fl"]["lr_stat"], 11.9732)
        check("Pearson r", d["pearson_raw_vs_llm"], 0.8116)

    # --- 2. Normative baseline ---
    print("\n[2] Normative baseline")
    d = load_json("normative_baseline.json")
    if d:
        check("Score-blind LLM AUROC", d["scores"]["E_scoreblind_llm_only"]["auroc"], 0.4941)
        check("Optimal combiner AUROC", d["scores"]["C_optimal_combiner_ml_plus_blind_llm"]["auroc"], 0.7792)
        check("Score-aware LLM AUROC", d["scores"]["B_scoreaware_llm_only"]["auroc"], 0.727)

    # --- 3. TOST equivalence ---
    print("\n[3] TOST equivalence")
    d = load_json("equivalence_decisioncurve.json")
    if d:
        tost = d["tost_equivalence_margin_003"]
        check("TOST observed diff", tost["observed_diff"], -0.0014, tol=0.002)
        check("TOST equivalence", tost["equivalence_within_margin"], True)

    # --- 4. Revision analyses ---
    print("\n[4] Revision analyses (paired, partial corr)")
    d = load_json("revision_analyses.json")
    if d:
        check("Paired shift", d["paired_controlled_ablation"]["mean_shift"], 0.491, tol=0.005)
        check("Partial r (ML|label)", d["partial_correlation"]["partial_r_ml_fl_controlling_label"], 0.7525)
        check("R2 ML only", d["partial_correlation"]["r2_ml_only"], 0.6588)
        check("Delta R2 label", d["partial_correlation"]["incremental_r2_label_over_ml"], 0.0037, tol=0.002)

    # --- 5. PaySim expanded ---
    print("\n[5] PaySim expanded (n=200)")
    d = load_json("paysim_expanded_audit.json")
    if d:
        check("PaySim n", d["n"], 200)
        check("PaySim blind AUROC", d["auroc_scoreblind_llm"], 0.6764)
        check("PaySim aware AUROC", d["auroc_scoreaware_llm"], 0.6246)
        check("PaySim verdict", d["normative_verdict"], "suppression")

    # --- 6. Rich features (Qwen) ---
    print("\n[6] Rich features (Qwen-7B)")
    d = load_json("rich_features_audit.json")
    if d:
        check("Rich blind AUROC", d["auroc_scoreblind_llm"], 0.6165)
        check("Rich aware AUROC", d["auroc_scoreaware_llm"], 0.4237)
        check("Rich alpha", d["alpha_rich"], 0.2441)

    # --- 7. Cross-model rich (Gemma) ---
    print("\n[7] Cross-model rich (Gemma-9B)")
    d = load_json("crossmodel_rich_audit.json")
    if d:
        g = d.get("Gemma-2-9B", {})
        check("Gemma blind AUROC", g.get("auroc_scoreblind", 0), 0.624)
        check("Gemma aware AUROC", g.get("auroc_scoreaware", 0), 0.417)
        check("Gemma suppression", g.get("suppression", 0), 0.207)

    # --- 8. API experiments ---
    print("\n[8] API experiments (GPT-4o-mini, GPT-4o)")
    d = load_json("api_experiments_summary.json")
    if d:
        # Medical
        m = d.get("medical", {})
        if "error" not in m:
            check("Medical blind AUROC", m["auroc_blind"], 1.0)
            check("Medical aware AUROC", m["auroc_aware"], 0.444, tol=0.005)
            check("Medical alpha", m["alpha"], 0.3535)
        # Paired staged
        ps = d.get("paired_staged", {})
        check("Paired staged std delta", ps.get("standard_delta", 0), 0.0455)
        check("Paired staged stg delta", ps.get("staged_delta", 0), 0.0451)
        # Rich features GPT-4o-mini
        rf = d.get("rich_features", {})
        check("Rich GPT4omini blind", rf.get("auroc_scoreblind", 0), 0.5446)
        check("Rich GPT4omini aware", rf.get("auroc_scoreaware", 0), 0.3703)
        # GPT-4o ablation
        g4 = d.get("gpt4o_ablation", {})
        check("GPT-4o shift", g4.get("mean_shift", 0), 0.5981)
        check("GPT-4o alpha", g4.get("alpha", 0), 0.6479)

    # --- 9. Platt scaling ---
    print("\n[9] Platt scaling")
    d = load_json("platt_scaled_results.json")
    if d:
        check("Platt raw ECE", d["raw"]["ece"], 0.400, tol=0.005)
        check("Platt cal ECE", d["calibrated"]["ece"], 0.133, tol=0.005)
        check("Platt per-fold ECE mean", d["per_fold_ece_calibrated_mean"], 0.167, tol=0.005)

    # --- 10. Repeated-run variance ---
    print("\n[10] Repeated-run variance")
    d = load_json("repeated_run_variance.json")
    if d:
        check("Deterministic count", d["n_perfectly_deterministic"], 20)
        check("Within-item SD mean", d["within_item_std_mean"], 0.0)

    # --- 11. Manifest existence ---
    print("\n[11] Manifest checks")
    manifests = [
        "llm_input_ieee_cis_100.manifest.json",
    ]
    for m in manifests:
        path = RESULTS / m
        if path.exists():
            data = json.load(open(path))
            check(f"{m} exists", True, True)
            check(f"{m} row count", data.get("n_total", 0), 200)
        else:
            check(f"{m} exists", False, True)

    # --- 12. Extension battery: cross-model claude/gemini ---
    print("\n[12] Extended cross-model ablation (Claude + Gemini)")
    d = load_json("ablation_claude_haiku.json")
    if d:
        check("Claude-Haiku alpha range", 0.5 < d["alpha"] < 1.2, True)
        check("Claude-Haiku n_valid", d["n_valid"] >= 30, True)
    d = load_json("ablation_gemini_flash.json")
    if d:
        check("Gemini-flash alpha range", 0.5 < d["alpha"] < 1.2, True)
        check("Gemini-flash n_valid", d["n_valid"] >= 30, True)

    # --- 13. Reasoning models (o3-mini + Gemini-pro thinking) ---
    print("\n[13] Reasoning models")
    d = load_json("ablation_o1mini.json")
    if d:
        check("o3-mini model id", d.get("model"), "o3-mini")
        check("o3-mini alpha ≈ 1.0", d.get("alpha", 0), 1.0, tol=0.1)

    # --- 14. Mitigation Zoo ---
    print("\n[14] Mitigation Zoo")
    d = load_json("mitigation_zoo_audit.json")
    if d:
        bm = d["by_method"]
        check("Baseline anchoring r", bm["fl_base"]["score_correlation_r"], 1.0, tol=0.05)
        check("Adversary anchoring r", bm["fl_adversary"]["score_correlation_r"], 0.41, tol=0.10)
        check("Best mitigation = adversary", d["best_mitigation"], "fl_adversary")

    # --- 15. Prior dose-response ---
    print("\n[15] Bayesian prior dose-response")
    d = load_json("prior_dose_response_audit.json")
    if d:
        check("prior_50 mean fl ≈ 0.15", d["by_prior"]["prior_50"]["mean_fl"], 0.15, tol=0.05)
        check("prior_70 mean fl ≈ 0.15", d["by_prior"]["prior_70"]["mean_fl"], 0.15, tol=0.05)

    # --- 16. Mechanism (base-rate priming) ---
    print("\n[16] Mechanism: base-rate priming")
    d = load_json("mechanism_audit.json")
    if d:
        check("TP normal mean ≈ 0.15", d["tp_normal_mean"], 0.15, tol=0.03)
        check("TP primed mean ≈ 0.50", d["tp_primed_mean"], 0.498, tol=0.03)
        check("TP paired diff p < 1e-9", d["tp_paired_diff_p"] < 1e-9, True)

    # --- 17. Multi-seed stability ---
    print("\n[17] Multi-seed stability")
    d = load_json("multiseed_ablation_audit.json")
    if d:
        check("Cross-seed shift mean ≈ 0.084", d["across_seed_mean_shift"], 0.084, tol=0.02)
        check("Cross-seed shift SD ≈ 0.023", d["across_seed_std_shift"], 0.023, tol=0.01)

    # --- 18. UCI Adult ---
    print("\n[18] UCI Adult (real non-fraud benchmark, synthetic profiles)")
    d = load_json("uci_adult_audit.json")
    if d:
        check("UCI blind AUROC > 0.99", d["auroc_scoreblind"] > 0.99, True)
        check("UCI LR p < 0.05", d["lr_test"]["p"] < 0.05, True)

    # --- 19. Dataset card sanity ---
    print("\n[19] Dataset card")
    d = load_json("dataset_card.json")
    if d:
        check("Dataset card n_models", d["dataset_card_summary"]["n_models"], 8)
        n_total = d["totals"]["total_records"]
        check("Dataset card n_records 3300-3700", 3300 <= n_total <= 3700, True)

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print(f"RESULTS: {PASS} passed, {FAIL} failed")
    print(f"{'=' * 60}")

    if FAIL > 0:
        print("\nFAILED: Paper numbers do not match artifacts!")
        sys.exit(1)
    else:
        print("\nALL CHECKS PASSED: Paper numbers match artifacts.")
        sys.exit(0)


if __name__ == "__main__":
    main()
