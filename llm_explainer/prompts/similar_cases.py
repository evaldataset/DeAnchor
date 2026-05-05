"""Task 3: 유사 사기 사례 검색 + 비교 분석 프롬프트.

RAG로 검색된 유사 사기 사례와 현재 거래를 비교 분석.
"""

SYSTEM_PROMPT = """You are a financial crime analyst specializing in case-based reasoning. Compare current suspicious transactions against historical fraud cases to identify patterns and assess risk."""

SIMILAR_CASES_PROMPT = """Analyze the following suspicious transaction by comparing it with similar past fraud cases.

## Current Transaction
{transaction_text}

ML Fraud Score: {fraud_score:.4f}

## Similar Past Cases (from RAG retrieval)
{cases_text}

## Task
Compare the current transaction with the retrieved cases and provide analysis.

Respond in JSON format:
```json
{{
    "case_comparisons": [
        {{
            "case_id": "case identifier",
            "similarity_aspects": ["shared pattern 1", "shared pattern 2"],
            "differences": ["difference 1", "difference 2"],
            "relevance_score": 0.0-1.0
        }}
    ],
    "pattern_summary": "What common fraud pattern does this transaction match?",
    "risk_assessment": "Based on historical cases, how risky is this transaction?",
    "novel_aspects": ["Any aspects not seen in past cases"]
}}
```"""


def build_prompt(
    transaction_text: str,
    fraud_score: float,
    similar_cases: list[dict] | None = None,
) -> str:
    if not similar_cases:
        cases_text = "No similar cases found in the database."
    else:
        parts = []
        for i, case in enumerate(similar_cases, 1):
            parts.append(f"### Case {i}: {case.get('title', 'Unknown')}")
            parts.append(f"- ID: {case.get('id', 'N/A')}")
            parts.append(f"- Type: {case.get('fraud_type', 'Unknown')}")
            parts.append(f"- Summary: {case.get('summary', 'N/A')}")
            if case.get('amounts_mentioned'):
                parts.append(f"- Amounts: {', '.join(case['amounts_mentioned'][:3])}")
            if case.get('regulations'):
                parts.append(f"- Regulations: {', '.join(case['regulations'][:3])}")
            parts.append(f"- Similarity Score: {case.get('similarity', 0):.2%}")
            parts.append("")
        cases_text = "\n".join(parts)

    return SIMILAR_CASES_PROMPT.format(
        transaction_text=transaction_text,
        fraud_score=fraud_score,
        cases_text=cases_text,
    )
