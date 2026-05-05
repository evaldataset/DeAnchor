"""Human evaluation MD 파일 생성."""
import json

with open("evaluation/human_eval_items.json") as f:
    items = json.load(f)

lines = []
lines.append("# Human Evaluation for Fraud Explanation Quality\n")
lines.append("## Instructions\n")
lines.append("You are evaluating AI-generated explanations for suspicious financial transactions.")
lines.append("A machine learning model has flagged each transaction as potentially fraudulent.")
lines.append("For each transaction, you will see **three different explanation approaches**.\n")
lines.append("**Rate each explanation on a 1-5 scale:**")
lines.append("- 1 = Very Poor (unhelpful, confusing, or wrong)")
lines.append("- 2 = Poor (some information but major gaps)")
lines.append("- 3 = Adequate (reasonable but generic)")
lines.append("- 4 = Good (specific, useful, minor issues)")
lines.append("- 5 = Excellent (immediately actionable, clear, trustworthy)\n")
lines.append("**Evaluate on 4 dimensions:**")
lines.append("1. **Usefulness** — Would this help you investigate this transaction?")
lines.append("2. **Clarity** — Is it easy to understand?")
lines.append("3. **Specificity** — Does it point to concrete evidence, not just generalities?")
lines.append("4. **Trust** — Do you trust this explanation?\n")
lines.append("After rating all three, **rank them 1st / 2nd / 3rd**.\n")
lines.append("---\n")

SCORE_TABLE = """| Dimension | Score (1-5) |
|-----------|------------|
| Usefulness | ___ |
| Clarity | ___ |
| Specificity | ___ |
| Trust | ___ |"""

RANK_TABLE = """### Ranking

| Rank | Explanation (A / B / C) |
|------|----------------------|
| 1st (Best) | ___ |
| 2nd | ___ |
| 3rd (Worst) | ___ |

**Brief reason for your ranking (optional):** ___"""

for item in items:
    lines.append(f"## {item['id']}: Transaction\n")
    lines.append(f"**ML Fraud Score:** {item['ml_score']:.4f}\n")
    lines.append("```")
    lines.append(item["transaction"])
    lines.append("```\n")
    lines.append("---\n")

    lines.append("### Explanation A: Feature Attribution\n")
    lines.append("```")
    lines.append(item["shap_explanation"])
    lines.append("```\n")
    lines.append(SCORE_TABLE)
    lines.append("\n---\n")

    lines.append("### Explanation B: AI Narrative\n")
    lines.append("```")
    lines.append(item["llm_explanation"][:400])
    lines.append("```\n")
    lines.append(SCORE_TABLE)
    lines.append("\n---\n")

    lines.append("### Explanation C: Combined (Feature Attribution + AI Narrative)\n")
    lines.append("```")
    lines.append(item["combined_explanation"][:500])
    lines.append("```\n")
    lines.append(SCORE_TABLE)
    lines.append("\n---\n")

    lines.append(RANK_TABLE)
    lines.append("\n---\n---\n")

lines.append("## Demographics (Optional)\n")
lines.append("1. Do you have experience with data analysis? (Yes / No / Some)")
lines.append("2. Have you worked in financial services? (Yes / No)")
lines.append("3. How familiar are you with fraud detection? (Not at all / Somewhat / Very)\n")
lines.append("---\n")
lines.append("## Thank you!\n")
lines.append("Your responses will be used for academic research on AI explainability.")
lines.append("All data is anonymized and no personal information is collected.")

md = "\n".join(lines)
with open("evaluation/human_eval_example.md", "w") as f:
    f.write(md)
print(f"Written: {len(md)} chars, {len(items)} items")
