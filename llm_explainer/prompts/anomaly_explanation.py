"""Task 1: 이상 거래 설명 프롬프트.

ML 모델이 이상으로 플래깅한 거래에 대해
"왜 이 거래가 의심스러운가?"를 설명.
"""

SYSTEM_PROMPT = """You are a financial fraud analyst AI. Your role is to analyze flagged transactions and explain why they are suspicious in clear, professional language.

Guidelines:
- Be specific about which features are anomalous and why
- Reference concrete numbers and patterns
- Consider the transaction in context (time, amount, frequency)
- Structure your explanation clearly
- Use professional financial terminology
- Be concise but thorough"""

EXPLANATION_PROMPT = """A fraud detection ML model has flagged the following transaction as potentially fraudulent with a confidence score of {fraud_score:.1%}.

## Transaction Details
{transaction_text}

## ML Model Analysis
- Fraud probability score: {fraud_score:.4f}
- Score threshold for flagging: {threshold:.4f}
- Score is {score_position} the threshold

{rag_context}

## Task
Analyze this transaction and provide a structured explanation:

1. **Risk Summary** (1-2 sentences): Overall assessment of why this transaction is suspicious
2. **Anomalous Patterns** (bullet points): List each specific anomaly detected
3. **Risk Level**: HIGH / MEDIUM / LOW
4. **Recommended Action**: What should a fraud analyst investigate next?

Respond in JSON format:
```json
{{
    "risk_summary": "...",
    "anomalous_patterns": ["pattern 1", "pattern 2", ...],
    "risk_level": "HIGH|MEDIUM|LOW",
    "recommended_action": "...",
    "confidence": 0.0-1.0
}}
```"""


def build_prompt(
    transaction_text: str,
    fraud_score: float,
    threshold: float = 0.5,
    similar_cases: list[dict] | None = None,
) -> str:
    """이상 거래 설명 프롬프트 생성."""
    score_position = "above" if fraud_score >= threshold else "below"

    rag_context = ""
    if similar_cases:
        rag_context = "## Similar Past Cases\n"
        for i, case in enumerate(similar_cases, 1):
            rag_context += f"\n### Case {i}: {case.get('title', 'Unknown')}\n"
            rag_context += f"- Type: {case.get('fraud_type', 'Unknown')}\n"
            rag_context += f"- Summary: {case.get('summary', 'N/A')}\n"
            rag_context += f"- Similarity: {case.get('similarity', 0):.2%}\n"

    return EXPLANATION_PROMPT.format(
        transaction_text=transaction_text,
        fraud_score=fraud_score,
        threshold=threshold,
        score_position=score_position,
        rag_context=rag_context,
    )
