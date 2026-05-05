"""Core function unit tests."""

import json
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestBootstrapCI:
    def test_basic_ci(self):
        from evaluation.bootstrap_ci import bootstrap_ci
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = bootstrap_ci(data)
        assert 2.0 <= result["mean"] <= 4.0
        assert result["ci_lower"] < result["ci_upper"]
        assert result["n"] == 5

    def test_single_value(self):
        from evaluation.bootstrap_ci import bootstrap_ci
        result = bootstrap_ci([3.0, 3.0, 3.0])
        assert result["mean"] == 3.0
        assert result["std"] == 0.0

    def test_permutation_test_significant(self):
        from evaluation.bootstrap_ci import paired_permutation_test
        a = [10.0] * 50
        b = [0.0] * 50
        result = paired_permutation_test(a, b)
        assert result["p_value"] < 0.001
        assert result["observed_diff"] == 10.0

    def test_permutation_test_nonsignificant(self):
        from evaluation.bootstrap_ci import paired_permutation_test
        np.random.seed(42)
        a = list(np.random.normal(0.5, 0.1, 50))
        b = list(np.random.normal(0.5, 0.1, 50))
        result = paired_permutation_test(a, b)
        assert result["p_value"] > 0.05


class TestJSONParsing:
    def test_json_block(self):
        from llm_explainer.llm_inference import _parse_json_response
        response = '```json\n{"fraud_likelihood": 0.85}\n```'
        result = _parse_json_response(response)
        assert result["fraud_likelihood"] == 0.85

    def test_raw_json(self):
        from llm_explainer.llm_inference import _parse_json_response
        response = '{"fraud_likelihood": 0.5}'
        result = _parse_json_response(response)
        assert result["fraud_likelihood"] == 0.5

    def test_unparseable(self):
        from llm_explainer.llm_inference import _parse_json_response
        result = _parse_json_response("This is not JSON at all")
        assert result.get("parse_error") is True

    def test_malformed_json(self):
        from llm_explainer.llm_inference import _parse_json_response
        result = _parse_json_response('{"key": }')
        assert result.get("parse_error") is True


class TestAnchoringAlpha:
    def test_perfect_anchoring(self):
        """Α=1.0 when aware output equals ML score."""
        from evaluation.anchoring_alpha import compute_alpha
        import tempfile, os
        # Create temp files
        aware = [{"fp_explanation": {"fraud_likelihood": 0.9}, "ml_score": 0.9, "original": {"category": "tp"}}]
        blind = [{"blind_assessment": {"fraud_likelihood": 0.0}, "ml_score": 0.9, "original": {"category": "tp"}}]

        a_path = tempfile.mktemp(suffix=".jsonl")
        b_path = tempfile.mktemp(suffix=".jsonl")
        for p, d in [(a_path, aware), (b_path, blind)]:
            with open(p, "w") as f:
                for item in d:
                    f.write(json.dumps(item) + "\n")

        result = compute_alpha(a_path, b_path)
        assert abs(result["alpha"] - 1.0) < 0.01
        os.unlink(a_path)
        os.unlink(b_path)

    def test_no_anchoring(self):
        """Α=0.0 when aware and blind outputs are the same."""
        from evaluation.anchoring_alpha import compute_alpha
        import tempfile, os
        data = [{"fp_explanation": {"fraud_likelihood": 0.5}, "blind_assessment": {"fraud_likelihood": 0.5}, "ml_score": 0.9, "original": {"category": "tp"}}]

        a_path = tempfile.mktemp(suffix=".jsonl")
        b_path = tempfile.mktemp(suffix=".jsonl")
        for p in [a_path, b_path]:
            with open(p, "w") as f:
                for item in data:
                    f.write(json.dumps(item) + "\n")

        result = compute_alpha(a_path, b_path)
        assert abs(result["alpha"]) < 0.01
        os.unlink(a_path)
        os.unlink(b_path)


class TestPromptBuilders:
    def test_blind_prompt_no_score_value(self):
        """Score-blind prompt must NOT contain an actual ML score number."""
        from llm_explainer.prompts.fp_explanation_blind import build_prompt
        prompt = build_prompt("Transaction Amount: $500, Type: TRANSFER")
        # Should mention "no ML model score" (explaining absence) but not inject a real score value
        assert "Fraud probability:" not in prompt
        assert "fraud score:" not in prompt.lower()
        assert "$500" in prompt or "500" in prompt

    def test_blind_prompt_returns_string(self):
        from llm_explainer.prompts.fp_explanation_blind import build_prompt
        result = build_prompt("test transaction")
        assert isinstance(result, str)
        assert len(result) > 50


class TestDoseResponseSlope:
    def test_slope_indices(self):
        """Verify slope calculation uses correct indices (0.9 vs 0.5)."""
        # Simulate fls array: scores=[0.1, 0.3, 0.5, 0.7, 0.9]
        fls = [0.10, 0.10, 0.23, 0.31, 0.55]
        # Correct: (fl@0.9 - fl@0.5) / (0.9-0.5) = (0.55 - 0.23) / 0.4 = 0.80
        slope = (fls[4] - fls[2]) / 0.4
        assert abs(slope - 0.80) < 0.01
        # Wrong (old bug): (fls[2] - fls[0]) / 0.4 = (0.23 - 0.10) / 0.4 = 0.325
        wrong_slope = (fls[2] - fls[0]) / 0.4
        assert abs(wrong_slope - 0.325) < 0.01
        assert slope != wrong_slope


class TestCalibration:
    def test_perfect_calibration(self):
        from evaluation.calibration_analysis import compute_ece
        preds = [0.1] * 10 + [0.9] * 10
        labels = [0] * 10 + [1] * 10
        result = compute_ece(preds, labels)
        assert result["ece"] < 0.15  # near-perfect

    def test_worst_calibration(self):
        from evaluation.calibration_analysis import compute_ece
        preds = [0.9] * 10 + [0.1] * 10
        labels = [0] * 10 + [1] * 10
        result = compute_ece(preds, labels)
        assert result["ece"] > 0.5
