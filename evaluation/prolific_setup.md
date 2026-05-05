# Human Evaluation Protocol (Prolific)

## Study Design

- **Platform**: Prolific (prolific.com)
- **Participants**: 30 workers (no fraud expertise required)
- **Task**: Rate fraud explanation quality (blind comparison)
- **Duration**: ~20 minutes per participant
- **Compensation**: $4.00 per participant ($12/hour)
- **Total cost**: ~$120 + Prolific fees (~$150 total)

## Task Structure

### Part 1: SHAP vs LLM (15 pairs)

Each participant sees 15 transaction-explanation pairs.
For each pair, they see two explanations (A and B, randomized order):

- A = SHAP feature attribution + template
- B = LLM narrative

They rate each explanation 1-5 on:

1. **Usefulness**: How helpful would this be for investigating this transaction?
2. **Clarity**: How easy is this to understand?
3. **Trust**: How much do you trust this explanation?
4. **Overall**: Overall quality?

Then select which they prefer (A, B, or No preference).

### Part 2: Combined vs Individual (10 pairs)

Same format, but three explanations:

- A = SHAP only
- B = LLM only
- C = SHAP + LLM combined

Rank 1st/2nd/3rd.

## Materials Location

- `data/benchmarks/human_eval/human_eval_sheet.csv` — 50 evaluation items
- `data/benchmarks/human_eval/evaluation_guidelines.md` — annotator instructions
- `data/benchmarks/human_eval/human_eval_ground_truth.csv` — ground truth (not shown to annotators)

## Prolific Configuration

- **Eligibility**: Fluent English, approval rate > 95%
- **Attention checks**: 2 per participant (known easy/hard cases)
- **Device**: Desktop only (for table readability)
- **Estimated completion**: 3 days

## Analysis Plan

- Inter-annotator agreement: Krippendorff's alpha
- Mean scores per approach (with 95% CI)
- Preference distribution (with chi-squared test)
- Subgroup analysis by self-reported data analysis experience
