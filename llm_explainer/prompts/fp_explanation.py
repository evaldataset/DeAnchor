"""Task 4: 사기 여부 재평가 (Fraud Re-assessment).

ML이 사기로 플래깅한 거래에 대해 LLM이 독립적으로 재평가.
편향 없이 "사기인가 정상인가"를 양방향으로 분석.
"""

SYSTEM_PROMPT = """You are a senior financial fraud investigator conducting independent reviews of ML-flagged transactions. Your job is to provide an UNBIASED second opinion.

CRITICAL RULES:
- Do NOT assume the ML model is wrong. High-scoring transactions are usually fraud.
- Evaluate BOTH sides: evidence FOR fraud AND evidence AGAINST fraud.
- Only recommend RELEASE if you find STRONG, SPECIFIC evidence of legitimacy.
- When in doubt, recommend HOLD_FOR_REVIEW.
- A transaction with multiple anomalous PCA components (|z|>3) is very likely fraud.
- Late night timing alone is weak evidence, but combined with anomalous features, it strengthens fraud signals."""

FP_EXPLANATION_PROMPT = """An ML fraud detection model flagged this transaction with a high fraud score. Conduct an independent assessment.

## Transaction Details
{transaction_text}

## ML Model Analysis
- Fraud score: {fraud_score:.4f} (threshold: {threshold:.4f})
- This score means the model is {confidence_level} this is fraud

## Task
Provide a BALANCED assessment. You must analyze BOTH possibilities:

1. **Evidence FOR fraud**: What patterns suggest this IS fraudulent?
2. **Evidence AGAINST fraud**: What patterns suggest this is legitimate?
3. **Final verdict**: Based on the weight of evidence, is this more likely fraud or legitimate?

IMPORTANT: Most ML-flagged transactions with scores above 0.99 ARE actual fraud. Only override the ML if you find compelling specific evidence of legitimacy.

Respond in JSON format:
```json
{{
    "evidence_for_fraud": ["specific evidence 1", "specific evidence 2"],
    "evidence_against_fraud": ["specific evidence 1", "specific evidence 2"],
    "fraud_likelihood": 0.0-1.0,
    "assessment": "Balanced analysis of both sides",
    "recommendation": "ESCALATE_AS_FRAUD|HOLD_FOR_REVIEW|RELEASE",
    "confidence": 0.0-1.0,
    "key_deciding_factor": "The single most important factor in your decision"
}}
```"""


def build_prompt(
    transaction_text: str,
    fraud_score: float | None,
    threshold: float = 0.5,
) -> str:
    if fraud_score is None:
        raise ValueError("fraud_score must not be None; use a score-blind prompt instead")
    if fraud_score > 0.99:
        confidence_level = "extremely confident (>99%)"
    elif fraud_score > 0.95:
        confidence_level = "very confident (>95%)"
    elif fraud_score > 0.90:
        confidence_level = "confident (>90%)"
    else:
        confidence_level = f"moderately confident ({fraud_score:.0%})"

    return FP_EXPLANATION_PROMPT.format(
        transaction_text=transaction_text,
        fraud_score=fraud_score,
        threshold=threshold,
        confidence_level=confidence_level,
    )
