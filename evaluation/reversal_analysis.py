"""Score-blind 역전 현상 deep dive.

FP인데 높은 fraud_likelihood를 받는 사례 vs
TP인데 낮은 fraud_likelihood를 받는 사례를 비교 분석.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def analyze_reversal(blind_path: str) -> dict:
    with open(blind_path) as f:
        results = [json.loads(line) for line in f]

    fp_high = []  # FP인데 fraud_likelihood 높음 (역전)
    tp_low = []   # TP인데 fraud_likelihood 낮음 (역전)
    fp_low = []   # FP인데 fraud_likelihood 낮음 (정상)
    tp_high = []  # TP인데 fraud_likelihood 높음 (정상)

    for r in results:
        cat = r.get("original", {}).get("category", "")
        assessment = r.get("blind_assessment", {})
        fl = assessment.get("fraud_likelihood")

        if not isinstance(fl, (int, float)):
            continue

        entry = {
            "fraud_likelihood": fl,
            "text": r["original"].get("text", "")[:200],
            "ml_score": r["original"].get("fraud_score", 0),
            "evidence_for": assessment.get("evidence_for_fraud", [])[:2],
            "evidence_against": assessment.get("evidence_against_fraud", [])[:2],
            "recommendation": assessment.get("recommendation", "?"),
        }

        if cat == "false_positive":
            if fl >= 0.5:
                fp_high.append(entry)
            else:
                fp_low.append(entry)
        elif cat == "true_positive":
            if fl < 0.5:
                tp_low.append(entry)
            else:
                tp_high.append(entry)

    print(f"=== Score-Blind Reversal Analysis ===")
    print(f"Total: {len(results)} transactions")
    print(f"")
    print(f"REVERSED (problematic):")
    print(f"  FP with HIGH fraud_likelihood (≥0.5): {len(fp_high)}")
    print(f"  TP with LOW fraud_likelihood (<0.5):  {len(tp_low)}")
    print(f"")
    print(f"CORRECT:")
    print(f"  FP with LOW fraud_likelihood (<0.5):  {len(fp_low)}")
    print(f"  TP with HIGH fraud_likelihood (≥0.5): {len(tp_high)}")

    # Pattern analysis
    if fp_high:
        print(f"\n--- FP cases LLM thinks are fraud (top 3) ---")
        fp_high.sort(key=lambda x: -x["fraud_likelihood"])
        for e in fp_high[:3]:
            print(f"  fl={e['fraud_likelihood']:.2f}, ml={e['ml_score']:.3f}")
            print(f"    Text: {e['text'][:150]}")
            print(f"    FOR fraud: {e['evidence_for']}")
            print(f"    AGAINST:   {e['evidence_against']}")
            print()

    if tp_low:
        print(f"--- TP cases LLM thinks are normal (top 3) ---")
        tp_low.sort(key=lambda x: x["fraud_likelihood"])
        for e in tp_low[:3]:
            print(f"  fl={e['fraud_likelihood']:.2f}, ml={e['ml_score']:.3f}")
            print(f"    Text: {e['text'][:150]}")
            print(f"    FOR fraud: {e['evidence_for']}")
            print(f"    AGAINST:   {e['evidence_against']}")
            print()

    # Transaction text pattern analysis
    print(f"--- Feature Patterns ---")
    for label, group in [("FP_high", fp_high), ("TP_low", tp_low), ("FP_low", fp_low), ("TP_high", tp_high)]:
        if not group:
            continue
        amounts = []
        for e in group:
            text = e["text"]
            if "$" in text:
                import re
                m = re.search(r'\$[\d,.]+', text)
                if m:
                    try:
                        amounts.append(float(m.group().replace("$","").replace(",","")))
                    except ValueError:
                        pass
        if amounts:
            print(f"  {label}: n={len(group)}, mean_amount=${np.mean(amounts):,.0f}, median=${np.median(amounts):,.0f}")

    return {
        "fp_high": len(fp_high), "tp_low": len(tp_low),
        "fp_low": len(fp_low), "tp_high": len(tp_high),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    analyze_reversal(args.input)
