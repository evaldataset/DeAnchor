"""Task 4 변형: Score-blind 사기 여부 평가.

ML 점수를 제공하지 않고, 거래 텍스트만으로 LLM이 독립 판단.
Anchoring bias 제거 실험용.
"""

SYSTEM_PROMPT = """You are a senior financial fraud investigator. You are given a transaction description and must independently assess whether it is likely fraudulent or legitimate.

You have NO information about any ML model's prediction. Judge purely based on the transaction characteristics.

Consider:
- Transaction amount patterns (unusual size, round numbers)
- Timing (late night, weekends)
- Balance changes (account draining, zero-balance receivers)
- Transaction type patterns
- Any other behavioral anomalies"""

BLIND_PROMPT = """Assess the following transaction for potential fraud based ONLY on the transaction details below. You have no ML model score or prior assessment.

## Transaction Details
{transaction_text}

## Task
Provide an independent fraud assessment based solely on the transaction characteristics.

Respond in JSON format:
```json
{{
    "fraud_likelihood": 0.0-1.0,
    "assessment": "Your independent analysis",
    "evidence_for_fraud": ["specific suspicious patterns"],
    "evidence_against_fraud": ["specific normal patterns"],
    "recommendation": "ESCALATE_AS_FRAUD|HOLD_FOR_REVIEW|RELEASE",
    "confidence": 0.0-1.0,
    "key_deciding_factor": "The single most important factor"
}}
```"""


def build_prompt(transaction_text: str) -> str:
    return BLIND_PROMPT.format(transaction_text=transaction_text)
