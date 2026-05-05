"""Task 2: 사기 유형 분류 프롬프트.

7개 사기 유형 중 하나로 분류 + 근거 제시.
"""

SYSTEM_PROMPT = """You are a financial crime classification expert. Classify suspicious transactions into specific fraud categories with supporting evidence.

Fraud Categories:
1. Structuring - Breaking large transactions into smaller ones to avoid reporting thresholds ($10,000)
2. Layering - Complex series of transactions to obscure the origin of funds
3. Identity Fraud - Using stolen or synthetic identities for transactions
4. Insider Trading - Trading based on material non-public information
5. Pump & Dump - Artificially inflating asset prices then selling
6. Account Takeover - Unauthorized access to and use of another's account
7. Legitimate - Transaction appears normal despite initial flagging"""

CLASSIFICATION_PROMPT = """Classify the following flagged transaction into one of 7 fraud categories.

## Transaction Details
{transaction_text}

## ML Model Score
Fraud probability: {fraud_score:.4f}

{rag_context}

## Fraud Categories
1. Structuring
2. Layering
3. Identity Fraud
4. Insider Trading
5. Pump & Dump
6. Account Takeover
7. Legitimate

## Task
Classify this transaction and provide evidence for your classification.

Respond in JSON format:
```json
{{
    "primary_classification": "category name",
    "confidence": 0.0-1.0,
    "evidence": ["evidence point 1", "evidence point 2", ...],
    "alternative_classification": "second most likely category or null",
    "alternative_confidence": 0.0-1.0,
    "reasoning": "Brief explanation of classification logic"
}}
```"""


def build_prompt(
    transaction_text: str,
    fraud_score: float,
    similar_cases: list[dict] | None = None,
) -> str:
    rag_context = ""
    if similar_cases:
        rag_context = "## Similar Past Cases\n"
        for i, case in enumerate(similar_cases, 1):
            rag_context += f"- Case {i} ({case.get('fraud_type', '?')}): {case.get('summary', 'N/A')}\n"

    return CLASSIFICATION_PROMPT.format(
        transaction_text=transaction_text,
        fraud_score=fraud_score,
        rag_context=rag_context,
    )
