"""Task 5: 사기 조사 보고서 자동 생성 프롬프트.

구조화된 보고서: 거래 요약, 이상 패턴, 유사 사례, 규제 조항, 조사 우선순위.
"""

SYSTEM_PROMPT = """You are a senior financial crime investigator generating formal investigation reports. Your reports are used by compliance officers and regulators. They must be thorough, well-structured, and actionable."""

REPORT_PROMPT = """Generate a comprehensive fraud investigation report for the following flagged transaction.

## Transaction Under Investigation
{transaction_text}

## ML Model Assessment
- Fraud Probability Score: {fraud_score:.4f}
- Model: XGBoost ensemble (5-fold CV)
- Threshold: {threshold:.4f}

## Anomaly Explanation
{anomaly_explanation}

## Fraud Classification
{classification_result}

## Similar Historical Cases
{similar_cases_text}

## Task
Generate a formal investigation report with the following sections:

1. **Executive Summary**: 2-3 sentence overview
2. **Transaction Analysis**: Detailed breakdown of the transaction
3. **Anomaly Details**: Specific anomalous patterns identified
4. **Historical Comparison**: How this compares to past fraud cases
5. **Regulatory Considerations**: Applicable regulations and reporting requirements
6. **Risk Assessment**: Overall risk level with justification
7. **Recommended Actions**: Prioritized list of investigation steps
8. **Priority Level**: CRITICAL / HIGH / MEDIUM / LOW

Respond in JSON format:
```json
{{
    "executive_summary": "...",
    "transaction_analysis": "...",
    "anomaly_details": ["detail 1", "detail 2"],
    "historical_comparison": "...",
    "regulatory_considerations": ["regulation 1", "regulation 2"],
    "risk_assessment": {{
        "level": "CRITICAL|HIGH|MEDIUM|LOW",
        "justification": "..."
    }},
    "recommended_actions": [
        {{"priority": 1, "action": "...", "reason": "..."}},
        {{"priority": 2, "action": "...", "reason": "..."}}
    ],
    "priority_level": "CRITICAL|HIGH|MEDIUM|LOW",
    "estimated_loss_risk": "dollar amount or range if applicable"
}}
```"""


def build_prompt(
    transaction_text: str,
    fraud_score: float,
    threshold: float = 0.5,
    anomaly_explanation: str = "Not yet analyzed",
    classification_result: str = "Not yet classified",
    similar_cases: list[dict] | None = None,
) -> str:
    if not similar_cases:
        similar_cases_text = "No similar cases found."
    else:
        parts = []
        for i, case in enumerate(similar_cases, 1):
            parts.append(f"Case {i}: {case.get('title', 'Unknown')} - {case.get('summary', 'N/A')}")
        similar_cases_text = "\n".join(parts)

    return REPORT_PROMPT.format(
        transaction_text=transaction_text,
        fraud_score=fraud_score,
        threshold=threshold,
        anomaly_explanation=anomaly_explanation,
        classification_result=classification_result,
        similar_cases_text=similar_cases_text,
    )
