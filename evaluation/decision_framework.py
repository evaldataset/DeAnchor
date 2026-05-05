"""DeAnchor three-gate pre-deployment decision framework.

Inputs (all AUROCs from 5-fold CV logistic probes):
    B: score-blind LLM AUROC
    A: score-aware LLM AUROC
    M: raw ML AUROC
    F: fusion (M + A) AUROC
    prevalence: operational prevalence (optional)

Three gates (paper Appendix~K):
    G1 — Independent-signal gate:   if B <= M - 0.10  → LLM has < independent signal
    G2 — Suppression gate:          if B - F > 0.05    → score-aware LLM is being suppressed
    G3 — Fusion-value gate:         if |F - M| < 0.02  → fusion adds no measurable lift

Verdict mapping:
    G1 fires alone:                 narrate-only        (use LLM only for narrative)
    G2 fires:                       mitigate            (apply adversary framing)
    G3 fires (without G2):          skip                (do not deploy LLM layer)
    G2 ∨ (G1 ∧ G3):                 mitigate            (G2 takes priority over skip)
    none fire:                      deploy              (LLM adds value)
"""
from __future__ import annotations

from typing import Any


def evaluate_gates(B: float, A: float, M: float, F: float,
                   prevalence: float | None = None,
                   indep_threshold: float = 0.10,
                   suppression_threshold: float = 0.05,
                   fusion_threshold: float = 0.02) -> dict[str, Any]:
    """Apply the 3-gate decision framework to four AUROCs.

    Returns a dict with the gate booleans, the verdict, and a rationale string.
    """
    g1_low_independent_signal = B <= M - indep_threshold
    g2_suppression = (B - F) > suppression_threshold
    g3_fusion_no_lift = abs(F - M) < fusion_threshold

    if g2_suppression:
        verdict = "mitigate"
        rationale = (
            f"Score-aware fusion is suppressed below blind potential "
            f"(B={B:.3f} > F={F:.3f} by {B - F:+.3f} ≥ {suppression_threshold}). "
            f"Apply adversary framing or LLM-only deployment."
        )
    elif g1_low_independent_signal and g3_fusion_no_lift:
        verdict = "narrate-only"
        rationale = (
            f"LLM has < independent signal (B={B:.3f} ≤ M-{indep_threshold}={M - indep_threshold:.3f}) "
            f"and fusion provides no measurable lift (|F-M|={abs(F - M):.3f} < {fusion_threshold}). "
            f"Use LLM for narrative roles only."
        )
    elif g3_fusion_no_lift:
        verdict = "skip"
        rationale = (
            f"Fusion provides no measurable lift over raw ML "
            f"(|F-M|={abs(F - M):.3f} < {fusion_threshold}). "
            f"Do not deploy the LLM layer for discrimination."
        )
    elif g1_low_independent_signal:
        verdict = "narrate-only"
        rationale = (
            f"LLM has materially less independent signal than ML "
            f"(B={B:.3f} ≤ M-{indep_threshold}={M - indep_threshold:.3f}). "
            f"Use LLM for narrative roles only."
        )
    else:
        verdict = "deploy"
        rationale = (
            f"LLM adds incremental signal (F={F:.3f} vs M={M:.3f}, "
            f"B={B:.3f}) and is not score-suppressed."
        )

    return {
        "gates": {
            "G1_low_independent_signal": bool(g1_low_independent_signal),
            "G2_suppression": bool(g2_suppression),
            "G3_fusion_no_lift": bool(g3_fusion_no_lift),
        },
        "thresholds": {
            "indep_threshold": indep_threshold,
            "suppression_threshold": suppression_threshold,
            "fusion_threshold": fusion_threshold,
        },
        "verdict": verdict,
        "rationale": rationale,
        "auroc_summary": {"B": round(B, 4), "A": round(A, 4),
                          "M": round(M, 4), "F": round(F, 4)},
        "prevalence": prevalence,
    }


if __name__ == "__main__":
    # Self-test on the IEEE-CIS reference numbers
    print("Self-test on IEEE-CIS (paper Table 4):")
    r = evaluate_gates(B=0.494, A=0.727, M=0.788, F=0.786)
    print(f"  Gates: {r['gates']}")
    print(f"  Verdict: {r['verdict']}")
    print(f"  Rationale: {r['rationale']}")
    print()
    print("Self-test on enriched-features rich (paper §5):")
    r = evaluate_gates(B=0.617, A=0.424, M=0.700, F=0.513)
    print(f"  Gates: {r['gates']}")
    print(f"  Verdict: {r['verdict']}")
    print(f"  Rationale: {r['rationale']}")
