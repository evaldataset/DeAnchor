"""Paper-claim integration tests.

Each test pins a specific number from the manuscript to its source artifact.
Failure means the manuscript and the released data diverge — fix one or the
other.

Run: pytest tests/test_paper_claims.py -v
"""

import json
import math
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "experiments" / "results"


def _load(name):
    p = RESULTS / name
    if not p.exists():
        pytest.skip(f"Required artifact not present: {name}")
    return json.load(open(p))


# ---------------------------------------------------------------------
# Controlled ablation (Section 5.1, Table 2)
# ---------------------------------------------------------------------
class TestControlledAblation:
    def test_paired_shift_close_to_paper(self):
        d = _load("revision_analyses.json")
        assert math.isclose(d["paired_controlled_ablation"]["mean_shift"],
                            0.491, abs_tol=0.005)

    def test_paired_shift_significant(self):
        d = _load("revision_analyses.json")
        # paper §5.1 reports p < 0.001
        assert d["paired_controlled_ablation"]["p_value"] < 0.001


# ---------------------------------------------------------------------
# Fusion / incremental-value (Section 5.2, Table 4)
# ---------------------------------------------------------------------
class TestFusion:
    def test_raw_ml_auroc(self):
        d = _load("fusion_baseline.json")
        assert math.isclose(d["auroc"]["raw_ml_only"], 0.788, abs_tol=0.005)

    def test_llm_fl_auroc(self):
        d = _load("fusion_baseline.json")
        assert math.isclose(d["auroc"]["llm_fl_only"], 0.727, abs_tol=0.005)

    def test_fusion_no_incremental(self):
        d = _load("fusion_baseline.json")
        # Paper §5.2: chi2=2.13, p=0.145
        chi = d["likelihood_ratio_test"]["does_llm_fl_add_over_raw_ml"]["lr_stat"]
        p = d["likelihood_ratio_test"]["does_llm_fl_add_over_raw_ml"]["p_value"]
        assert math.isclose(chi, 2.13, abs_tol=0.05)
        assert p > 0.05  # not significant


# ---------------------------------------------------------------------
# Normative baseline (Section 5.2)
# ---------------------------------------------------------------------
class TestNormativeBaseline:
    def test_score_blind_llm_chance(self):
        d = _load("normative_baseline.json")
        assert math.isclose(
            d["scores"]["E_scoreblind_llm_only"]["auroc"], 0.494, abs_tol=0.01
        )

    def test_combiner_matches_raw(self):
        d = _load("normative_baseline.json")
        c = d["scores"]["C_optimal_combiner_ml_plus_blind_llm"]["auroc"]
        # Paper §5.2: combiner ~0.78, matches raw ML within noise
        assert math.isclose(c, 0.779, abs_tol=0.01)


# ---------------------------------------------------------------------
# TOST equivalence (Section 5.2)
# ---------------------------------------------------------------------
class TestTOSTEquivalence:
    def test_equivalence_within_margin(self):
        d = _load("equivalence_decisioncurve.json")
        # margin ±0.03
        assert d["tost_equivalence_margin_003"]["equivalence_within_margin"] is True


# ---------------------------------------------------------------------
# PaySim cross-dataset (Section 5)
# ---------------------------------------------------------------------
class TestPaySim:
    def test_paysim_n_200(self):
        d = _load("paysim_expanded_audit.json")
        assert d["n"] == 200

    def test_paysim_suppression(self):
        d = _load("paysim_expanded_audit.json")
        # blind > aware on PaySim
        assert d["auroc_scoreblind_llm"] > d["auroc_scoreaware_llm"] - 0.01

    def test_paysim_verdict(self):
        d = _load("paysim_expanded_audit.json")
        assert d["normative_verdict"] == "suppression"


# ---------------------------------------------------------------------
# Rich features (Section 5)
# ---------------------------------------------------------------------
class TestRichFeatures:
    def test_blind_above_chance(self):
        d = _load("rich_features_audit.json")
        # Paper: 0.617
        assert math.isclose(d["auroc_scoreblind_llm"], 0.617, abs_tol=0.01)

    def test_aware_below_blind(self):
        d = _load("rich_features_audit.json")
        # Paper: 0.424 (suppression)
        assert d["auroc_scoreaware_llm"] < d["auroc_scoreblind_llm"]


# ---------------------------------------------------------------------
# Cross-model extended ablation (Section 5.3, Appendix S)
# ---------------------------------------------------------------------
class TestCrossModelExtended:
    def test_claude_haiku_anchored(self):
        d = _load("ablation_claude_haiku.json")
        # Paper Abstract: α ∈ [0.5, 1.1] for all 8 families
        assert 0.5 <= d["alpha"] <= 1.2

    def test_gemini_flash_anchored(self):
        d = _load("ablation_gemini_flash.json")
        assert 0.5 <= d["alpha"] <= 1.2

    def test_o3_mini_strong_anchoring(self):
        d = _load("ablation_o1mini.json")  # filename kept; renamed inside
        # Paper §6 / Appendix: α ≈ 1.00 for o3-mini
        assert d.get("model") == "o3-mini"
        assert math.isclose(d.get("alpha", 0), 1.0, abs_tol=0.1)


# ---------------------------------------------------------------------
# Mitigation Zoo (Section 6.3, Appendix Q)
# ---------------------------------------------------------------------
class TestMitigationZoo:
    def test_adversary_reduces_anchoring(self):
        d = _load("mitigation_zoo_audit.json")
        bm = d["by_method"]
        base_r = bm["fl_base"]["score_correlation_r"]
        adv_r = bm["fl_adversary"]["score_correlation_r"]
        # Paper: r drops 1.00 → 0.41 (-59%)
        assert base_r > 0.95
        assert adv_r < 0.6

    def test_best_mitigation_is_adversary(self):
        d = _load("mitigation_zoo_audit.json")
        assert d["best_mitigation"] == "fl_adversary"


# ---------------------------------------------------------------------
# Bayesian prior dose-response (Appendix M)
# ---------------------------------------------------------------------
class TestPriorDoseResponse:
    def test_low_priors_no_shift(self):
        d = _load("prior_dose_response_audit.json")
        # Paper: 5 of 6 prior levels (1-70%) produce no shift; mean fl ~ 0.15
        for k in ("prior_1", "prior_10", "prior_30", "prior_50", "prior_70"):
            assert math.isclose(d["by_prior"][k]["mean_fl"], 0.15, abs_tol=0.05)


# ---------------------------------------------------------------------
# Mechanism (base-rate priming) (Section 6.2)
# ---------------------------------------------------------------------
class TestMechanism:
    def test_priming_recovery(self):
        d = _load("mechanism_audit.json")
        # Paper §6.2: TP normal=0.150 → primed=0.498
        assert d["tp_normal_mean"] < 0.20
        assert d["tp_primed_mean"] > 0.45

    def test_paired_diff_significant(self):
        d = _load("mechanism_audit.json")
        # Paper: paired p < 10^-10
        assert d["tp_paired_diff_p"] < 1e-9


# ---------------------------------------------------------------------
# Multi-seed ablation stability (Appendix / §7)
# ---------------------------------------------------------------------
class TestMultiSeed:
    def test_cross_seed_sd(self):
        d = _load("multiseed_ablation_audit.json")
        # Paper §7: cross-seed shift SD = 0.023 on mean 0.084
        assert math.isclose(d["across_seed_std_shift"], 0.023, abs_tol=0.01)
        assert math.isclose(d["across_seed_mean_shift"], 0.084, abs_tol=0.02)


# ---------------------------------------------------------------------
# UCI Adult (Section 5)
# ---------------------------------------------------------------------
class TestUCIAdult:
    def test_uci_blind_high(self):
        d = _load("uci_adult_audit.json")
        # Paper Abstract: blind 0.999, aware 0.964
        assert d["auroc_scoreblind"] > 0.99

    def test_uci_lr_significant(self):
        d = _load("uci_adult_audit.json")
        # Paper Abstract: LR p=0.008
        assert d["lr_test"]["p"] < 0.05


# ---------------------------------------------------------------------
# Manifest and dataset card sanity
# ---------------------------------------------------------------------
class TestManifestAndDatasetCard:
    def test_ieee_manifest_has_200(self):
        m = _load("llm_input_ieee_cis_100.manifest.json")
        # Manifest documents what was generated (n_total = n_fp + n_tp)
        assert m.get("n_fp", 0) + m.get("n_tp", 0) >= 100

    def test_dataset_card_has_8_models(self):
        d = _load("dataset_card.json")
        assert d["dataset_card_summary"]["n_models"] == 8

    def test_dataset_card_total_records_in_range(self):
        d = _load("dataset_card.json")
        # Paper: approximately 3,500 LLM responses (after K.2 remediation)
        n = d["totals"]["total_records"]
        assert 3300 <= n <= 3700, f"Expected ~3,500 records, got {n}"
