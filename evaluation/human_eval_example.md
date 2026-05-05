# Human Evaluation for Fraud Explanation Quality

## Instructions

You are evaluating AI-generated explanations for suspicious financial transactions.
A machine learning model has flagged each transaction as potentially fraudulent.
For each transaction, you will see **three different explanation approaches**.

**Rate each explanation on a 1-5 scale:**
- 1 = Very Poor (unhelpful, confusing, or wrong)
- 2 = Poor (some information but major gaps)
- 3 = Adequate (reasonable but generic)
- 4 = Good (specific, useful, minor issues)
- 5 = Excellent (immediately actionable, clear, trustworthy)

**Evaluate on 4 dimensions:**
1. **Usefulness** — Would this help you investigate this transaction?
2. **Clarity** — Is it easy to understand?
3. **Specificity** — Does it point to concrete evidence, not just generalities?
4. **Trust** — Do you trust this explanation?

After rating all three, **rank them 1st / 2nd / 3rd**.

---

## Q01: Transaction

**ML Fraud Score:** 0.8384

```
Transaction Amount: $335.00 (typical range), round number
Time: evening (18:00), weekday
Product: W
Email domain: verizon.net
Time deltas: 4 available
```

---

### Explanation A: Feature Attribution

```
ML Fraud Score: 0.8384

FACTORS INCREASING FRAUD RISK:
  - D2 = 0.00 (SHAP: +0.647, increases fraud risk)
  - card6 = 1.00 (SHAP: +0.529, increases fraud risk)
  - TransactionAmt = 335.00 (SHAP: +0.430, increases fraud risk)

FACTORS DECREASING FRAUD RISK:
  - C14 = 3.00 (SHAP: -0.487, decreases fraud risk)
  - D10 = 286.00 (SHAP: -0.439, decreases fraud risk)
  - P_emaildomain = 49.00 (SHAP: -0.399, decreases fraud risk)

Overall: 5 features increase risk, 5 features decrease risk.
```

| Dimension | Score (1-5) |
|-----------|------------|
| Usefulness | ___ |
| Clarity | ___ |
| Specificity | ___ |
| Trust | ___ |

---

### Explanation B: AI Narrative

```
The evidence for fraud is moderate, primarily due to the round number and the high fraud score. However, the timing and typical transaction amount provide some counter-evidence suggesting legitimacy.
Recommendation: HOLD_FOR_REVIEW
Fraud likelihood: 0.8385
```

| Dimension | Score (1-5) |
|-----------|------------|
| Usefulness | ___ |
| Clarity | ___ |
| Specificity | ___ |
| Trust | ___ |

---

### Explanation C: Combined (Feature Attribution + AI Narrative)

```
### Quantitative Evidence (SHAP)
ML Fraud Score: 0.8384

FACTORS INCREASING FRAUD RISK:
  - D2 = 0.00 (SHAP: +0.647, increases fraud risk)
  - card6 = 1.00 (SHAP: +0.529, increases fraud risk)
  - TransactionAmt = 335.00 (SHAP: +0.430, increases fraud risk)

FACTORS DECREASING FRAUD RISK:
  - C14 = 3.00 (SHAP: -0.487, decreases fraud risk)
  - D10 = 286.00 (SHAP: -0.439, decreases fraud risk)
  - P_emaildomain = 49.00 (SHAP: -0.399, decreases fraud risk)

Overall: 5 features increase risk, 5 fea
```

| Dimension | Score (1-5) |
|-----------|------------|
| Usefulness | ___ |
| Clarity | ___ |
| Specificity | ___ |
| Trust | ___ |

---

### Ranking

| Rank | Explanation (A / B / C) |
|------|----------------------|
| 1st (Best) | ___ |
| 2nd | ___ |
| 3rd (Worst) | ___ |

**Brief reason for your ranking (optional):** ___

---
---

## Q02: Transaction

**ML Fraud Score:** 0.9267

```
Transaction Amount: $23.20 (typical range)
Time: late night/early morning (0:00), weekday
Product: C
Email domain: gmail.com
Time deltas: 5 available
```

---

### Explanation A: Feature Attribution

```
ML Fraud Score: 0.9267

FACTORS INCREASING FRAUD RISK:
  - C14 = 0.00 (SHAP: +1.165, increases fraud risk)
  - C13 = 0.00 (SHAP: +0.619, increases fraud risk)
  - card6 = 1.00 (SHAP: +0.435, increases fraud risk)

FACTORS DECREASING FRAUD RISK:
  - TransactionAmt = 23.20 (SHAP: -0.482, decreases fraud risk)
  - C1 = 1.00 (SHAP: -0.380, decreases fraud risk)
  - card1 = 3867.00 (SHAP: -0.149, decreases fraud risk)

Overall: 5 features increase risk, 5 features decrease risk.
```

| Dimension | Score (1-5) |
|-----------|------------|
| Usefulness | ___ |
| Clarity | ___ |
| Specificity | ___ |
| Trust | ___ |

---

### Explanation B: AI Narrative

```
The evidence against fraud is weak and does not outweigh the high fraud score provided by the ML model.
Recommendation: HOLD_FOR_REVIEW
Fraud likelihood: 0.9274
```

| Dimension | Score (1-5) |
|-----------|------------|
| Usefulness | ___ |
| Clarity | ___ |
| Specificity | ___ |
| Trust | ___ |

---

### Explanation C: Combined (Feature Attribution + AI Narrative)

```
### Quantitative Evidence (SHAP)
ML Fraud Score: 0.9267

FACTORS INCREASING FRAUD RISK:
  - C14 = 0.00 (SHAP: +1.165, increases fraud risk)
  - C13 = 0.00 (SHAP: +0.619, increases fraud risk)
  - card6 = 1.00 (SHAP: +0.435, increases fraud risk)

FACTORS DECREASING FRAUD RISK:
  - TransactionAmt = 23.20 (SHAP: -0.482, decreases fraud risk)
  - C1 = 1.00 (SHAP: -0.380, decreases fraud risk)
  - card1 = 3867.00 (SHAP: -0.149, decreases fraud risk)

Overall: 5 features increase risk, 5 features dec
```

| Dimension | Score (1-5) |
|-----------|------------|
| Usefulness | ___ |
| Clarity | ___ |
| Specificity | ___ |
| Trust | ___ |

---

### Ranking

| Rank | Explanation (A / B / C) |
|------|----------------------|
| 1st (Best) | ___ |
| 2nd | ___ |
| 3rd (Worst) | ___ |

**Brief reason for your ranking (optional):** ___

---
---

## Q03: Transaction

**ML Fraud Score:** 0.8799

```
Transaction Amount: $117.00 (typical range), round number
Time: morning (9:00), weekday
Product: W
Email domain: aol.com
Time deltas: 7 available
```

---

### Explanation A: Feature Attribution

```
ML Fraud Score: 0.8799

FACTORS INCREASING FRAUD RISK:
  - card6 = 1.00 (SHAP: +0.643, increases fraud risk)
  - hour = 9.00 (SHAP: +0.640, increases fraud risk)
  - C6 = 3.00 (SHAP: +0.400, increases fraud risk)

FACTORS DECREASING FRAUD RISK:
  - P_emaildomain = 2.00 (SHAP: -0.307, decreases fraud risk)
  - card1 = 15066.00 (SHAP: -0.249, decreases fraud risk)
  - C1 = 1.00 (SHAP: -0.244, decreases fraud risk)

Overall: 5 features increase risk, 5 features decrease risk.
```

| Dimension | Score (1-5) |
|-----------|------------|
| Usefulness | ___ |
| Clarity | ___ |
| Specificity | ___ |
| Trust | ___ |

---

### Explanation B: AI Narrative

```
The evidence against fraud is relatively strong, indicating that this transaction may be legitimate. However, the high fraud score from the ML model suggests caution.
Recommendation: HOLD_FOR_REVIEW
Fraud likelihood: 0.8796
```

| Dimension | Score (1-5) |
|-----------|------------|
| Usefulness | ___ |
| Clarity | ___ |
| Specificity | ___ |
| Trust | ___ |

---

### Explanation C: Combined (Feature Attribution + AI Narrative)

```
### Quantitative Evidence (SHAP)
ML Fraud Score: 0.8799

FACTORS INCREASING FRAUD RISK:
  - card6 = 1.00 (SHAP: +0.643, increases fraud risk)
  - hour = 9.00 (SHAP: +0.640, increases fraud risk)
  - C6 = 3.00 (SHAP: +0.400, increases fraud risk)

FACTORS DECREASING FRAUD RISK:
  - P_emaildomain = 2.00 (SHAP: -0.307, decreases fraud risk)
  - card1 = 15066.00 (SHAP: -0.249, decreases fraud risk)
  - C1 = 1.00 (SHAP: -0.244, decreases fraud risk)

Overall: 5 features increase risk, 5 features decr
```

| Dimension | Score (1-5) |
|-----------|------------|
| Usefulness | ___ |
| Clarity | ___ |
| Specificity | ___ |
| Trust | ___ |

---

### Ranking

| Rank | Explanation (A / B / C) |
|------|----------------------|
| 1st (Best) | ___ |
| 2nd | ___ |
| 3rd (Worst) | ___ |

**Brief reason for your ranking (optional):** ___

---
---

## Q04: Transaction

**ML Fraud Score:** 0.9624

```
Transaction Amount: $22.75 (typical range)
Time: late night/early morning (3:00), weekend
Product: C
Email domain: hotmail.com
Unusual count features (>95th pct): {'C12': 2}
Time deltas: 5 available
```

---

### Explanation A: Feature Attribution

```
ML Fraud Score: 0.9624

FACTORS INCREASING FRAUD RISK:
  - C8 = 2.00 (SHAP: +0.646, increases fraud risk)
  - C13 = 1.00 (SHAP: +0.468, increases fraud risk)
  - card3 = 185.00 (SHAP: +0.334, increases fraud risk)

FACTORS DECREASING FRAUD RISK:
  - TransactionAmt = 22.75 (SHAP: -0.307, decreases fraud risk)
  - P_emaildomain = 19.00 (SHAP: -0.185, decreases fraud risk)
  - card6 = 2.00 (SHAP: -0.167, decreases fraud risk)

Overall: 5 features increase risk, 5 features decrease risk.
```

| Dimension | Score (1-5) |
|-----------|------------|
| Usefulness | ___ |
| Clarity | ___ |
| Specificity | ___ |
| Trust | ___ |

---

### Explanation B: AI Narrative

```
The evidence for fraud is strong due to the high fraud score, late-night timing, and multiple anomalous features. However, the typical transaction amount and common email domain provide some counter-evidence.
Recommendation: HOLD_FOR_REVIEW
Fraud likelihood: 0.95
```

| Dimension | Score (1-5) |
|-----------|------------|
| Usefulness | ___ |
| Clarity | ___ |
| Specificity | ___ |
| Trust | ___ |

---

### Explanation C: Combined (Feature Attribution + AI Narrative)

```
### Quantitative Evidence (SHAP)
ML Fraud Score: 0.9624

FACTORS INCREASING FRAUD RISK:
  - C8 = 2.00 (SHAP: +0.646, increases fraud risk)
  - C13 = 1.00 (SHAP: +0.468, increases fraud risk)
  - card3 = 185.00 (SHAP: +0.334, increases fraud risk)

FACTORS DECREASING FRAUD RISK:
  - TransactionAmt = 22.75 (SHAP: -0.307, decreases fraud risk)
  - P_emaildomain = 19.00 (SHAP: -0.185, decreases fraud risk)
  - card6 = 2.00 (SHAP: -0.167, decreases fraud risk)

Overall: 5 features increase risk, 5 fe
```

| Dimension | Score (1-5) |
|-----------|------------|
| Usefulness | ___ |
| Clarity | ___ |
| Specificity | ___ |
| Trust | ___ |

---

### Ranking

| Rank | Explanation (A / B / C) |
|------|----------------------|
| 1st (Best) | ___ |
| 2nd | ___ |
| 3rd (Worst) | ___ |

**Brief reason for your ranking (optional):** ___

---
---

## Q05: Transaction

**ML Fraud Score:** 0.8674

```
Transaction Amount: $50.00 (typical range)
Time: late night/early morning (22:00), weekday
Product: H
Email domain: hotmail.com
Time deltas: 1 available
```

---

### Explanation A: Feature Attribution

```
ML Fraud Score: 0.8674

FACTORS INCREASING FRAUD RISK:
  - C14 = 0.00 (SHAP: +1.143, increases fraud risk)
  - C13 = 0.00 (SHAP: +0.701, increases fraud risk)
  - card6 = 1.00 (SHAP: +0.385, increases fraud risk)

FACTORS DECREASING FRAUD RISK:
  - C1 = 1.00 (SHAP: -0.442, decreases fraud risk)
  - addr1 = 469.00 (SHAP: -0.261, decreases fraud risk)
  - TransactionAmt = 50.00 (SHAP: -0.203, decreases fraud risk)

Overall: 5 features increase risk, 5 features decrease risk.
```

| Dimension | Score (1-5) |
|-----------|------------|
| Usefulness | ___ |
| Clarity | ___ |
| Specificity | ___ |
| Trust | ___ |

---

### Explanation B: AI Narrative

```
The evidence for fraud is strengthened by the late-night timing and the round number transaction amount. However, the typical range of the transaction amount and the use of a common email domain provide some counter-evidence. Overall, the high fraud score indicates a significant likelihood of fraud.
Recommendation: HOLD_FOR_REVIEW
Fraud likelihood: 0.8674
```

| Dimension | Score (1-5) |
|-----------|------------|
| Usefulness | ___ |
| Clarity | ___ |
| Specificity | ___ |
| Trust | ___ |

---

### Explanation C: Combined (Feature Attribution + AI Narrative)

```
### Quantitative Evidence (SHAP)
ML Fraud Score: 0.8674

FACTORS INCREASING FRAUD RISK:
  - C14 = 0.00 (SHAP: +1.143, increases fraud risk)
  - C13 = 0.00 (SHAP: +0.701, increases fraud risk)
  - card6 = 1.00 (SHAP: +0.385, increases fraud risk)

FACTORS DECREASING FRAUD RISK:
  - C1 = 1.00 (SHAP: -0.442, decreases fraud risk)
  - addr1 = 469.00 (SHAP: -0.261, decreases fraud risk)
  - TransactionAmt = 50.00 (SHAP: -0.203, decreases fraud risk)

Overall: 5 features increase risk, 5 features decr
```

| Dimension | Score (1-5) |
|-----------|------------|
| Usefulness | ___ |
| Clarity | ___ |
| Specificity | ___ |
| Trust | ___ |

---

### Ranking

| Rank | Explanation (A / B / C) |
|------|----------------------|
| 1st (Best) | ___ |
| 2nd | ___ |
| 3rd (Worst) | ___ |

**Brief reason for your ranking (optional):** ___

---
---

## Q06: Transaction

**ML Fraud Score:** 0.8800

```
Transaction Amount: $914.00 (extremely large (>3 std above average)), round number
Time: evening (18:00), weekday
Product: W
Email domain: comcast.net
Time deltas: 8 available
```

---

### Explanation A: Feature Attribution

```
ML Fraud Score: 0.8800

FACTORS INCREASING FRAUD RISK:
  - TransactionAmt = 914.00 (SHAP: +1.267, increases fraud risk)
  - D2 = 2.00 (SHAP: +0.552, increases fraud risk)
  - D3 = 2.00 (SHAP: +0.322, increases fraud risk)

FACTORS DECREASING FRAUD RISK:
  - card6 = 2.00 (SHAP: -0.213, decreases fraud risk)
  - C1 = 1.00 (SHAP: -0.199, decreases fraud risk)
  - C11 = 1.00 (SHAP: -0.158, decreases fraud risk)

Overall: 5 features increase risk, 5 features decrease risk.
```

| Dimension | Score (1-5) |
|-----------|------------|
| Usefulness | ___ |
| Clarity | ___ |
| Specificity | ___ |
| Trust | ___ |

---

### Explanation B: AI Narrative

```
The evidence against fraud is relatively strong, indicating that this transaction may be legitimate. However, the high fraud score from the ML model suggests caution.
Recommendation: HOLD_FOR_REVIEW
Fraud likelihood: 0.8796
```

| Dimension | Score (1-5) |
|-----------|------------|
| Usefulness | ___ |
| Clarity | ___ |
| Specificity | ___ |
| Trust | ___ |

---

### Explanation C: Combined (Feature Attribution + AI Narrative)

```
### Quantitative Evidence (SHAP)
ML Fraud Score: 0.8800

FACTORS INCREASING FRAUD RISK:
  - TransactionAmt = 914.00 (SHAP: +1.267, increases fraud risk)
  - D2 = 2.00 (SHAP: +0.552, increases fraud risk)
  - D3 = 2.00 (SHAP: +0.322, increases fraud risk)

FACTORS DECREASING FRAUD RISK:
  - card6 = 2.00 (SHAP: -0.213, decreases fraud risk)
  - C1 = 1.00 (SHAP: -0.199, decreases fraud risk)
  - C11 = 1.00 (SHAP: -0.158, decreases fraud risk)

Overall: 5 features increase risk, 5 features decrease 
```

| Dimension | Score (1-5) |
|-----------|------------|
| Usefulness | ___ |
| Clarity | ___ |
| Specificity | ___ |
| Trust | ___ |

---

### Ranking

| Rank | Explanation (A / B / C) |
|------|----------------------|
| 1st (Best) | ___ |
| 2nd | ___ |
| 3rd (Worst) | ___ |

**Brief reason for your ranking (optional):** ___

---
---

## Q07: Transaction

**ML Fraud Score:** 0.8305

```
Transaction Amount: $53.05 (typical range)
Time: late night/early morning (23:00), weekday
Product: C
Email domain: gmail.com
Unusual count features (>95th pct): {'C4': 2, 'C7': 2, 'C12': 2}
Time deltas: 5 available
```

---

### Explanation A: Feature Attribution

```
ML Fraud Score: 0.8305

FACTORS INCREASING FRAUD RISK:
  - C13 = 1.00 (SHAP: +0.519, increases fraud risk)
  - card6 = 1.00 (SHAP: +0.485, increases fraud risk)
  - C4 = 2.00 (SHAP: +0.474, increases fraud risk)

FACTORS DECREASING FRAUD RISK:
  - card1 = 16062.00 (SHAP: -0.435, decreases fraud risk)
  - addr1 = 130.00 (SHAP: -0.415, decreases fraud risk)
  - C1 = 2.00 (SHAP: -0.359, decreases fraud risk)

Overall: 5 features increase risk, 5 features decrease risk.
```

| Dimension | Score (1-5) |
|-----------|------------|
| Usefulness | ___ |
| Clarity | ___ |
| Specificity | ___ |
| Trust | ___ |

---

### Explanation B: AI Narrative

```
The evidence suggests a moderate likelihood of fraud due to the combination of late-night timing and multiple anomalous features, despite the typical transaction amount and common email domain.
Recommendation: HOLD_FOR_REVIEW
Fraud likelihood: 0.83
```

| Dimension | Score (1-5) |
|-----------|------------|
| Usefulness | ___ |
| Clarity | ___ |
| Specificity | ___ |
| Trust | ___ |

---

### Explanation C: Combined (Feature Attribution + AI Narrative)

```
### Quantitative Evidence (SHAP)
ML Fraud Score: 0.8305

FACTORS INCREASING FRAUD RISK:
  - C13 = 1.00 (SHAP: +0.519, increases fraud risk)
  - card6 = 1.00 (SHAP: +0.485, increases fraud risk)
  - C4 = 2.00 (SHAP: +0.474, increases fraud risk)

FACTORS DECREASING FRAUD RISK:
  - card1 = 16062.00 (SHAP: -0.435, decreases fraud risk)
  - addr1 = 130.00 (SHAP: -0.415, decreases fraud risk)
  - C1 = 2.00 (SHAP: -0.359, decreases fraud risk)

Overall: 5 features increase risk, 5 features decrease ri
```

| Dimension | Score (1-5) |
|-----------|------------|
| Usefulness | ___ |
| Clarity | ___ |
| Specificity | ___ |
| Trust | ___ |

---

### Ranking

| Rank | Explanation (A / B / C) |
|------|----------------------|
| 1st (Best) | ___ |
| 2nd | ___ |
| 3rd (Worst) | ___ |

**Brief reason for your ranking (optional):** ___

---
---

## Q08: Transaction

**ML Fraud Score:** 0.9523

```
Transaction Amount: $56.55 (typical range)
Time: afternoon (17:00), weekday
Product: C
Email domain: gmail.com
Unusual count features (>95th pct): {'C1': 25, 'C2': 111, 'C4': 19, 'C6': 19, 'C7': 19, 'C8': 4, 'C10': 4, 'C11': 27, 'C12': 27}
Time deltas: 7 available
```

---

### Explanation A: Feature Attribution

```
ML Fraud Score: 0.9523

FACTORS INCREASING FRAUD RISK:
  - C1 = 25.00 (SHAP: +2.131, increases fraud risk)
  - C6 = 19.00 (SHAP: +0.397, increases fraud risk)
  - C12 = 27.00 (SHAP: +0.339, increases fraud risk)

FACTORS DECREASING FRAUD RISK:
  - C14 = 9.00 (SHAP: -0.819, decreases fraud risk)
  - C13 = 17.00 (SHAP: -0.811, decreases fraud risk)
  - card1 = 15885.00 (SHAP: -0.497, decreases fraud risk)

Overall: 5 features increase risk, 5 features decrease risk.
```

| Dimension | Score (1-5) |
|-----------|------------|
| Usefulness | ___ |
| Clarity | ___ |
| Specificity | ___ |
| Trust | ___ |

---

### Explanation B: AI Narrative

```
The evidence for fraud is strong due to the high fraud score and weekend timing, but the typical transaction amount and morning timing provide some counter-evidence.
Recommendation: HOLD_FOR_REVIEW
Fraud likelihood: 0.85
```

| Dimension | Score (1-5) |
|-----------|------------|
| Usefulness | ___ |
| Clarity | ___ |
| Specificity | ___ |
| Trust | ___ |

---

### Explanation C: Combined (Feature Attribution + AI Narrative)

```
### Quantitative Evidence (SHAP)
ML Fraud Score: 0.9523

FACTORS INCREASING FRAUD RISK:
  - C1 = 25.00 (SHAP: +2.131, increases fraud risk)
  - C6 = 19.00 (SHAP: +0.397, increases fraud risk)
  - C12 = 27.00 (SHAP: +0.339, increases fraud risk)

FACTORS DECREASING FRAUD RISK:
  - C14 = 9.00 (SHAP: -0.819, decreases fraud risk)
  - C13 = 17.00 (SHAP: -0.811, decreases fraud risk)
  - card1 = 15885.00 (SHAP: -0.497, decreases fraud risk)

Overall: 5 features increase risk, 5 features decrease risk
```

| Dimension | Score (1-5) |
|-----------|------------|
| Usefulness | ___ |
| Clarity | ___ |
| Specificity | ___ |
| Trust | ___ |

---

### Ranking

| Rank | Explanation (A / B / C) |
|------|----------------------|
| 1st (Best) | ___ |
| 2nd | ___ |
| 3rd (Worst) | ___ |

**Brief reason for your ranking (optional):** ___

---
---

## Q09: Transaction

**ML Fraud Score:** 0.8508

```
Transaction Amount: $60.90 (typical range)
Time: late night/early morning (2:00), weekday
Product: C
Email domain: hotmail.com
Unusual count features (>95th pct): {'C4': 2, 'C7': 2, 'C12': 2}
Time deltas: 4 available
```

---

### Explanation A: Feature Attribution

```
ML Fraud Score: 0.8508

FACTORS INCREASING FRAUD RISK:
  - C4 = 2.00 (SHAP: +0.575, increases fraud risk)
  - card6 = 1.00 (SHAP: +0.518, increases fraud risk)
  - C13 = 1.00 (SHAP: +0.513, increases fraud risk)

FACTORS DECREASING FRAUD RISK:
  - C1 = 2.00 (SHAP: -0.281, decreases fraud risk)
  - P_emaildomain = 19.00 (SHAP: -0.164, decreases fraud risk)
  - C8 = 1.00 (SHAP: -0.146, decreases fraud risk)

Overall: 5 features increase risk, 5 features decrease risk.
```

| Dimension | Score (1-5) |
|-----------|------------|
| Usefulness | ___ |
| Clarity | ___ |
| Specificity | ___ |
| Trust | ___ |

---

### Explanation B: AI Narrative

```
The evidence for fraud is moderate, primarily based on the ML model's score and the round number of the transaction. However, the timing and typical amount provide some counterbalance, suggesting that while there are indicators of potential fraud, they are not overwhelmingly strong.
Recommendation: HOLD_FOR_REVIEW
Fraud likelihood: 0.8514
```

| Dimension | Score (1-5) |
|-----------|------------|
| Usefulness | ___ |
| Clarity | ___ |
| Specificity | ___ |
| Trust | ___ |

---

### Explanation C: Combined (Feature Attribution + AI Narrative)

```
### Quantitative Evidence (SHAP)
ML Fraud Score: 0.8508

FACTORS INCREASING FRAUD RISK:
  - C4 = 2.00 (SHAP: +0.575, increases fraud risk)
  - card6 = 1.00 (SHAP: +0.518, increases fraud risk)
  - C13 = 1.00 (SHAP: +0.513, increases fraud risk)

FACTORS DECREASING FRAUD RISK:
  - C1 = 2.00 (SHAP: -0.281, decreases fraud risk)
  - P_emaildomain = 19.00 (SHAP: -0.164, decreases fraud risk)
  - C8 = 1.00 (SHAP: -0.146, decreases fraud risk)

Overall: 5 features increase risk, 5 features decrease ri
```

| Dimension | Score (1-5) |
|-----------|------------|
| Usefulness | ___ |
| Clarity | ___ |
| Specificity | ___ |
| Trust | ___ |

---

### Ranking

| Rank | Explanation (A / B / C) |
|------|----------------------|
| 1st (Best) | ___ |
| 2nd | ___ |
| 3rd (Worst) | ___ |

**Brief reason for your ranking (optional):** ___

---
---

## Q10: Transaction

**ML Fraud Score:** 0.8899

```
Transaction Amount: $300.00 (typical range), round number
Time: evening (18:00), weekend
Product: H
Email domain: anonymous.com
Time deltas: 1 available
```

---

### Explanation A: Feature Attribution

```
ML Fraud Score: 0.8899

FACTORS INCREASING FRAUD RISK:
  - C14 = 0.00 (SHAP: +1.038, increases fraud risk)
  - TransactionAmt = 300.00 (SHAP: +0.613, increases fraud risk)
  - C13 = 0.00 (SHAP: +0.607, increases fraud risk)

FACTORS DECREASING FRAUD RISK:
  - P_emaildomain = 1.00 (SHAP: -0.590, decreases fraud risk)
  - card1 = 16659.00 (SHAP: -0.422, decreases fraud risk)
  - C1 = 1.00 (SHAP: -0.352, decreases fraud risk)

Overall: 5 features increase risk, 5 features decrease risk.
```

| Dimension | Score (1-5) |
|-----------|------------|
| Usefulness | ___ |
| Clarity | ___ |
| Specificity | ___ |
| Trust | ___ |

---

### Explanation B: AI Narrative

```
The evidence for fraud is strengthened by the late-night timing and the round number amount, but the typical amount and established email domain provide some counter-evidence. However, the overall indicators lean towards fraud.
Recommendation: HOLD_FOR_REVIEW
Fraud likelihood: 0.8897
```

| Dimension | Score (1-5) |
|-----------|------------|
| Usefulness | ___ |
| Clarity | ___ |
| Specificity | ___ |
| Trust | ___ |

---

### Explanation C: Combined (Feature Attribution + AI Narrative)

```
### Quantitative Evidence (SHAP)
ML Fraud Score: 0.8899

FACTORS INCREASING FRAUD RISK:
  - C14 = 0.00 (SHAP: +1.038, increases fraud risk)
  - TransactionAmt = 300.00 (SHAP: +0.613, increases fraud risk)
  - C13 = 0.00 (SHAP: +0.607, increases fraud risk)

FACTORS DECREASING FRAUD RISK:
  - P_emaildomain = 1.00 (SHAP: -0.590, decreases fraud risk)
  - card1 = 16659.00 (SHAP: -0.422, decreases fraud risk)
  - C1 = 1.00 (SHAP: -0.352, decreases fraud risk)

Overall: 5 features increase risk, 5 fe
```

| Dimension | Score (1-5) |
|-----------|------------|
| Usefulness | ___ |
| Clarity | ___ |
| Specificity | ___ |
| Trust | ___ |

---

### Ranking

| Rank | Explanation (A / B / C) |
|------|----------------------|
| 1st (Best) | ___ |
| 2nd | ___ |
| 3rd (Worst) | ___ |

**Brief reason for your ranking (optional):** ___

---
---

## Demographics (Optional)

1. Do you have experience with data analysis? (Yes / No / Some)
2. Have you worked in financial services? (Yes / No)
3. How familiar are you with fraud detection? (Not at all / Somewhat / Very)

---

## Thank you!

Your responses will be used for academic research on AI explainability.
All data is anonymized and no personal information is collected.