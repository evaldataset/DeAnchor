"""Extension battery: Tier 1 + Tier 2 + Tier 3 experiments.

Tier 1 (mitigation + mechanism deepening):
  1. mitigation_zoo: 5 mitigation strategies on paired TP+FP (n=50)
  2. prior_dose_response: Bayesian prior dose-response (6 levels x n=25 TPs)

Tier 2 (cross-model expansion):
  3. claude_gemini_ablation: controlled ablation on Claude-Haiku and Gemini-2.5-flash
  4. reasoning_models: o3-mini and Gemini-2.5-pro (CoT-style) ablation

Tier 3 (robustness):
  5. prompt_paraphrase_robustness: 5 prompt paraphrasings
  6. score_format_robustness: 4 score presentation formats

Total: ~$2-3, ~60-90 min.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm_explainer.llm_inference import _parse_json_response

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "experiments" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

oai = OpenAI()

try:
    import anthropic
    anth = anthropic.Anthropic()
except Exception:
    anth = None
try:
    from google import genai
    gem = genai.Client()
except Exception:
    gem = None


SYS = "You are a financial fraud investigator."


def call_oai(prompt, system=SYS, model="gpt-4o-mini", temperature=0.0, seed=42, max_tokens=400):
    resp = oai.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        temperature=temperature, max_tokens=max_tokens, seed=seed,
    )
    return resp.choices[0].message.content.strip()


def call_oai_reasoning(prompt, system=SYS, model="o3-mini"):
    """Reasoning models do not accept system role; merge into user."""
    full = f"{system}\n\n{prompt}"
    resp = oai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": full}],
        max_completion_tokens=4000,
    )
    return resp.choices[0].message.content.strip()


def call_anth(prompt, system=SYS, model="claude-haiku-4-5-20251001", max_tokens=400):
    msg = anth.messages.create(
        model=model, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


def call_gem(prompt, system=SYS, model="gemini-2.5-flash", thinking=False):
    full = f"{system}\n\n{prompt}"
    cfg = None
    if not thinking:
        from google.genai import types
        try:
            cfg = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        except Exception:
            cfg = None
    if cfg:
        r = gem.models.generate_content(model=model, contents=full, config=cfg)
    else:
        r = gem.models.generate_content(model=model, contents=full)
    return (r.text or "").strip()


def parse_fl(text):
    p = _parse_json_response(text)
    if isinstance(p, dict):
        v = p.get("fraud_likelihood")
        if isinstance(v, (int, float)) and 0 <= v <= 1:
            return float(v)
    return None


def load_paired_items(n_tp=25, n_fp=25):
    """Load TP+FP items from the canonical paired file."""
    with open(RESULTS / "controlled_ablation_ieee_with_score.jsonl") as f:
        items = [json.loads(line) for line in f]
    tps = [it for it in items if it.get("original", {}).get("is_fraud") == 1][:n_tp]
    fps = [it for it in items if it.get("original", {}).get("is_fraud") == 0][:n_fp]
    return tps, fps


# =====================================================================
# Tier 1.1: Mitigation Zoo
# =====================================================================
TPL_BASE = """Assess this transaction for fraud risk.

Transaction Details:
{text}

ML Model Score: Fraud probability: {score:.4f}

Respond in JSON: {{"fraud_likelihood": 0.0-1.0, "recommendation": "RELEASE|ESCALATE"}}"""


def mit_bayesian(text, score):
    return f"""IMPORTANT: Historical base rate of fraud in this batch is approximately 50% (high). Calibrate accordingly: do not assume rare-event prior.

Assess this transaction for fraud risk.

Transaction Details:
{text}

ML Model Score: Fraud probability: {score:.4f}

Respond in JSON: {{"fraud_likelihood": 0.0-1.0, "recommendation": "RELEASE|ESCALATE"}}"""


def mit_quantize(text, score):
    if score < 0.33:
        bucket = "LOW"
    elif score < 0.67:
        bucket = "MEDIUM"
    else:
        bucket = "HIGH"
    return f"""Assess this transaction for fraud risk.

Transaction Details:
{text}

ML Model Score (categorical): {bucket} risk

Respond in JSON: {{"fraud_likelihood": 0.0-1.0, "recommendation": "RELEASE|ESCALATE"}}"""


def mit_adversary(text, score):
    return f"""Assess this transaction for fraud risk.

NOTE: another model (which has been wrong before) reports the following score. Treat it as a noisy second opinion, not as ground truth. Form your own independent judgment first.

Transaction Details:
{text}

Other model's score: {score:.4f}

Respond in JSON: {{"fraud_likelihood": 0.0-1.0, "recommendation": "RELEASE|ESCALATE"}}"""


def mit_precommit_step1(text):
    """Step 1: ask for binary judgment without the score (pre-commitment)."""
    return f"""Assess this transaction for fraud risk WITHOUT any score, using only the transaction text.

Transaction Details:
{text}

Respond in JSON: {{"binary_judgment": "FRAUD|LEGITIMATE", "confidence_word": "LOW|MED|HIGH"}}"""


def mit_precommit_step2(text, score, prior_judgment, prior_conf):
    return f"""You previously committed to: judgment={prior_judgment}, confidence={prior_conf}.

Now an ML model also reports a fraud probability. Update your assessment but stay consistent with your prior judgment unless the score strongly contradicts it.

Transaction Details:
{text}

ML Model Score: Fraud probability: {score:.4f}

Respond in JSON: {{"fraud_likelihood": 0.0-1.0, "recommendation": "RELEASE|ESCALATE"}}"""


def run_mitigation_zoo(n_tp=25, n_fp=25):
    print("\n" + "="*60)
    print("Tier 1.1: Mitigation Zoo (5 strategies, paired)")
    print("="*60)
    tps, fps = load_paired_items(n_tp, n_fp)
    selected = tps + fps

    rows = []
    for i, item in enumerate(selected):
        text = item["original"]["text"]
        score = item["original"]["fraud_score"]
        y = item["original"]["is_fraud"]

        # Baseline (standard prompt with score)
        fl_base = parse_fl(call_oai(TPL_BASE.format(text=text, score=score)))
        # Mitigations
        fl_bayes = parse_fl(call_oai(mit_bayesian(text, score)))
        fl_quant = parse_fl(call_oai(mit_quantize(text, score)))
        fl_adv = parse_fl(call_oai(mit_adversary(text, score)))
        # Pre-commit: 2 calls
        s1 = _parse_json_response(call_oai(mit_precommit_step1(text)))
        if isinstance(s1, dict):
            pj = s1.get("binary_judgment", "LEGITIMATE")
            pc = s1.get("confidence_word", "MED")
        else:
            pj, pc = "LEGITIMATE", "MED"
        fl_pre = parse_fl(call_oai(mit_precommit_step2(text, score, pj, pc)))
        # Multi-LLM ensemble (3 different temperature seeds → averaged)
        fl_e1 = parse_fl(call_oai(TPL_BASE.format(text=text, score=score), temperature=0.7, seed=1))
        fl_e2 = parse_fl(call_oai(TPL_BASE.format(text=text, score=score), temperature=0.7, seed=2))
        fl_e3 = parse_fl(call_oai(TPL_BASE.format(text=text, score=score), temperature=0.7, seed=3))
        ensemble_vals = [v for v in [fl_e1, fl_e2, fl_e3] if v is not None]
        fl_ens = float(np.mean(ensemble_vals)) if ensemble_vals else None

        rows.append({"y": y, "score": score, "fl_base": fl_base, "fl_bayesian": fl_bayes,
                     "fl_quantize": fl_quant, "fl_adversary": fl_adv,
                     "fl_precommit": fl_pre, "fl_ensemble": fl_ens})
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(selected)}]")

    with open(RESULTS / "mitigation_zoo_raw.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    methods = ["fl_base", "fl_bayesian", "fl_quantize", "fl_adversary", "fl_precommit", "fl_ensemble"]
    summary = {"n_tp": n_tp, "n_fp": n_fp, "by_method": {}}
    for m in methods:
        valid = [r for r in rows if r[m] is not None]
        if not valid:
            continue
        tp_v = [r[m] for r in valid if r["y"] == 1]
        fp_v = [r[m] for r in valid if r["y"] == 0]
        scores = np.array([r["score"] for r in valid])
        fls = np.array([r[m] for r in valid])
        # Anchoring coefficient: corr between score and fl
        if len(scores) >= 5 and scores.std() > 0:
            r_pearson, p_pearson = stats.pearsonr(scores, fls)
        else:
            r_pearson, p_pearson = (0.0, 1.0)
        delta = float(np.mean(tp_v) - np.mean(fp_v)) if tp_v and fp_v else 0.0
        # paired t-test against base on TP shift if not baseline
        summary["by_method"][m] = {
            "n_valid": len(valid),
            "tp_mean": round(float(np.mean(tp_v)), 4) if tp_v else None,
            "fp_mean": round(float(np.mean(fp_v)), 4) if fp_v else None,
            "delta_tp_minus_fp": round(delta, 4),
            "score_correlation_r": round(float(r_pearson), 4),
            "score_correlation_p": float(p_pearson),
        }

    # Compute mitigation effect: each method's anchoring vs baseline
    base_r = summary["by_method"].get("fl_base", {}).get("score_correlation_r", 1.0)
    for m in methods:
        if m == "fl_base":
            continue
        r_m = summary["by_method"].get(m, {}).get("score_correlation_r", 1.0)
        summary["by_method"][m]["anchoring_reduction"] = round(base_r - r_m, 4)

    # Decide best mitigation: largest reduction in anchoring while preserving FP/TP delta
    best_m, best_score = None, -1.0
    for m in methods:
        if m == "fl_base":
            continue
        info = summary["by_method"].get(m, {})
        red = info.get("anchoring_reduction", 0)
        delta_kept = info.get("delta_tp_minus_fp", 0)
        # composite: reduction matters, but penalize if discrimination collapses
        if red > best_score and delta_kept > -0.05:
            best_score = red
            best_m = m
    summary["best_mitigation"] = best_m
    summary["best_anchoring_reduction"] = best_score

    with open(RESULTS / "mitigation_zoo_audit.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Best: {best_m} (anchoring reduction Δr={best_score:+.4f})")
    return summary


# =====================================================================
# Tier 1.2: Bayesian prior dose-response
# =====================================================================
def run_prior_dose_response(n_tp=25):
    print("\n" + "="*60)
    print("Tier 1.2: Bayesian prior dose-response")
    print("="*60)
    tps, _ = load_paired_items(n_tp, 0)

    PRIOR_LEVELS = [0.01, 0.10, 0.30, 0.50, 0.70, 0.90]

    def tpl_with_prior(text, score, prior):
        return f"""IMPORTANT: Historical base rate of fraud in this batch is approximately {int(prior*100)}%.

Assess this transaction for fraud risk.

Transaction Details:
{text}

ML Model Score: Fraud probability: {score:.4f}

Respond in JSON: {{"fraud_likelihood": 0.0-1.0, "recommendation": "RELEASE|ESCALATE"}}"""

    rows = []
    fake_score = 0.15  # fixed fake-low score to test prior effect
    for i, item in enumerate(tps):
        text = item["original"]["text"]
        rec = {"idx": i, "text_len": len(text)}
        for p in PRIOR_LEVELS:
            fl = parse_fl(call_oai(tpl_with_prior(text, fake_score, p)))
            rec[f"fl_p{int(p*100)}"] = fl
        rows.append(rec)
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{n_tp}]")

    with open(RESULTS / "prior_dose_response_raw.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    summary = {"n_tp": n_tp, "fake_score": fake_score, "by_prior": {}}
    means_by_p = []
    for p in PRIOR_LEVELS:
        key = f"fl_p{int(p*100)}"
        vals = [r[key] for r in rows if r[key] is not None]
        m = float(np.mean(vals)) if vals else None
        summary["by_prior"][f"prior_{int(p*100)}"] = {"n_valid": len(vals), "mean_fl": round(m, 4) if m else None}
        if m is not None:
            means_by_p.append((p, m))

    # Spearman correlation: prior level vs mean output
    if len(means_by_p) >= 4:
        ps = np.array([x[0] for x in means_by_p])
        ms = np.array([x[1] for x in means_by_p])
        rho, p_rho = stats.spearmanr(ps, ms)
        summary["spearman_prior_vs_fl"] = {"rho": round(float(rho), 4), "p": float(p_rho)}
        summary["range"] = {"min_fl": round(float(min(ms)), 4),
                            "max_fl": round(float(max(ms)), 4),
                            "delta": round(float(max(ms) - min(ms)), 4)}
    summary["interpretation"] = (
        "monotonic_bayesian_supported" if summary.get("spearman_prior_vs_fl", {}).get("rho", 0) > 0.7
        and summary.get("spearman_prior_vs_fl", {}).get("p", 1) < 0.05
        else "weak_prior_response"
    )
    with open(RESULTS / "prior_dose_response_audit.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Spearman ρ={summary.get('spearman_prior_vs_fl',{}).get('rho','-')}, "
          f"range Δ={summary.get('range',{}).get('delta','-')}")
    print(f"  Interpretation: {summary['interpretation']}")
    return summary


# =====================================================================
# Tier 2.3: Cross-model — Claude + Gemini ablation
# =====================================================================
def _ablation_pairs(call_fn, call_label, n_tp=25, n_fp=25):
    tps, fps = load_paired_items(n_tp, n_fp)
    selected = tps + fps
    TPL_AWARE = TPL_BASE
    TPL_BLIND = """Assess this transaction for fraud risk based on transaction details only.

Transaction Details:
{text}

Respond in JSON: {{"fraud_likelihood": 0.0-1.0, "recommendation": "RELEASE|ESCALATE"}}"""

    rows = []
    for i, item in enumerate(selected):
        text = item["original"]["text"]
        score = item["original"]["fraud_score"]
        y = item["original"]["is_fraud"]
        try:
            fl_a = parse_fl(call_fn(TPL_AWARE.format(text=text, score=score)))
        except Exception as e:
            fl_a = None
        try:
            fl_b = parse_fl(call_fn(TPL_BLIND.format(text=text)))
        except Exception:
            fl_b = None
        rows.append({"y": y, "score": score, "fl_aware": fl_a, "fl_blind": fl_b})
        if (i + 1) % 10 == 0:
            print(f"  [{call_label}] [{i+1}/{len(selected)}]")

    valid = [r for r in rows if r["fl_aware"] is not None and r["fl_blind"] is not None]
    if len(valid) < 10:
        return {"model": call_label, "n_valid": len(valid), "error": "insufficient"}
    aware = np.array([r["fl_aware"] for r in valid])
    blind = np.array([r["fl_blind"] for r in valid])
    scores = np.array([r["score"] for r in valid])
    diff = aware - blind
    t, p = stats.ttest_rel(aware, blind)
    if scores.std() > 0:
        r_pearson, _ = stats.pearsonr(scores, aware)
    else:
        r_pearson = 0.0
    alpha_num = float(np.mean(diff))
    alpha_den = float(np.mean(scores) - np.mean(blind))
    alpha = alpha_num / alpha_den if abs(alpha_den) > 1e-6 else None

    return {
        "model": call_label, "n_valid": len(valid),
        "aware_mean": round(float(aware.mean()), 4),
        "blind_mean": round(float(blind.mean()), 4),
        "paired_shift": round(float(diff.mean()), 4),
        "paired_t": round(float(t), 4),
        "paired_p": float(p),
        "alpha": round(alpha, 4) if alpha is not None else None,
        "score_correlation_r": round(float(r_pearson), 4),
    }


def run_claude_gemini_ablation(n_tp=25, n_fp=25):
    print("\n" + "="*60)
    print("Tier 2.3: Cross-model ablation (Claude + Gemini)")
    print("="*60)
    out = {}
    if anth:
        out["claude_haiku_4_5"] = _ablation_pairs(
            lambda p: call_anth(p, model="claude-haiku-4-5-20251001"),
            "claude-haiku-4.5", n_tp, n_fp)
        with open(RESULTS / "ablation_claude_haiku.json", "w") as f:
            json.dump(out["claude_haiku_4_5"], f, indent=2)
    if gem:
        out["gemini_2_5_flash"] = _ablation_pairs(
            lambda p: call_gem(p, model="gemini-2.5-flash"),
            "gemini-2.5-flash", n_tp, n_fp)
        with open(RESULTS / "ablation_gemini_flash.json", "w") as f:
            json.dump(out["gemini_2_5_flash"], f, indent=2)
    with open(RESULTS / "extension_crossmodel_audit.json", "w") as f:
        json.dump(out, f, indent=2)
    for k, v in out.items():
        print(f"  {k}: shift={v.get('paired_shift','-')}, alpha={v.get('alpha','-')}, r={v.get('score_correlation_r','-')}")
    return out


# =====================================================================
# Tier 2.4: Reasoning models (o3-mini + Gemini-2.5-pro thinking)
# =====================================================================
def run_reasoning_models(n_tp=15, n_fp=15):
    print("\n" + "="*60)
    print("Tier 2.4: Reasoning models (o3-mini, gemini-2.5-pro)")
    print("="*60)
    out = {}
    # o3-mini
    try:
        out["o1_mini"] = _ablation_pairs(
            lambda p: call_oai_reasoning(p, model="o3-mini"),
            "o3-mini", n_tp, n_fp)
        with open(RESULTS / "ablation_o1mini.json", "w") as f:
            json.dump(out["o1_mini"], f, indent=2)
    except Exception as e:
        print(f"  o3-mini failed: {e}")
        out["o1_mini"] = {"error": str(e)}
    # Gemini 2.5 Pro with thinking enabled
    if gem:
        try:
            out["gemini_2_5_pro_thinking"] = _ablation_pairs(
                lambda p: call_gem(p, model="gemini-2.5-pro", thinking=True),
                "gemini-2.5-pro-thinking", n_tp, n_fp)
            with open(RESULTS / "ablation_gem_pro_thinking.json", "w") as f:
                json.dump(out["gemini_2_5_pro_thinking"], f, indent=2)
        except Exception as e:
            print(f"  gemini-2.5-pro failed: {e}")
            out["gemini_2_5_pro_thinking"] = {"error": str(e)}
    with open(RESULTS / "extension_reasoning_audit.json", "w") as f:
        json.dump(out, f, indent=2)
    for k, v in out.items():
        if "alpha" in v:
            print(f"  {k}: shift={v.get('paired_shift','-')}, alpha={v.get('alpha','-')}, r={v.get('score_correlation_r','-')}")
    return out


# =====================================================================
# Tier 3.5: Prompt paraphrase robustness (5 paraphrases)
# =====================================================================
PARAPHRASES = [
    # baseline
    """Assess this transaction for fraud risk.

Transaction Details:
{text}

ML Model Score: Fraud probability: {score:.4f}

Respond in JSON: {{"fraud_likelihood": 0.0-1.0, "recommendation": "RELEASE|ESCALATE"}}""",
    # second-person
    """Please review this transaction. Decide whether you suspect it is fraudulent.

Transaction:
{text}

A machine-learning model assigned probability {score:.4f} that the transaction is fraud.

Reply in JSON format: {{"fraud_likelihood": 0.0-1.0, "recommendation": "RELEASE|ESCALATE"}}""",
    # legalese
    """Evaluate the following financial transaction for indicators of fraudulent activity per applicable AML guidelines.

TRANSACTION RECORD:
{text}

AUTOMATED RISK SCORE: {score:.4f}

Issue your determination as JSON: {{"fraud_likelihood": 0.0-1.0, "recommendation": "RELEASE|ESCALATE"}}""",
    # narrative
    """A new transaction just came in. Take a look:

{text}

Our ML pipeline gave it a fraud probability of {score:.4f}. What's your call?

Answer in JSON: {{"fraud_likelihood": 0.0-1.0, "recommendation": "RELEASE|ESCALATE"}}""",
    # checklist
    """Fraud-risk assessment task.

Inputs:
- Transaction: {text}
- ML fraud probability: {score:.4f}

Output (JSON only): {{"fraud_likelihood": 0.0-1.0, "recommendation": "RELEASE|ESCALATE"}}""",
]


def run_prompt_paraphrase_robustness(n_tp=15, n_fp=15):
    print("\n" + "="*60)
    print("Tier 3.5: Prompt paraphrase robustness (5 templates)")
    print("="*60)
    tps, fps = load_paired_items(n_tp, n_fp)
    selected = tps + fps
    rows = []
    for i, item in enumerate(selected):
        text = item["original"]["text"]
        score = item["original"]["fraud_score"]
        y = item["original"]["is_fraud"]
        rec = {"y": y, "score": score}
        for j, tpl in enumerate(PARAPHRASES):
            rec[f"fl_p{j}"] = parse_fl(call_oai(tpl.format(text=text, score=score)))
        rows.append(rec)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(selected)}]")
    with open(RESULTS / "prompt_paraphrase_raw.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    summary = {"n": len(rows), "by_paraphrase": []}
    alphas = []
    deltas = []
    for j in range(len(PARAPHRASES)):
        valid = [r for r in rows if r[f"fl_p{j}"] is not None]
        if not valid:
            continue
        scores = np.array([r["score"] for r in valid])
        fls = np.array([r[f"fl_p{j}"] for r in valid])
        tp_v = [r[f"fl_p{j}"] for r in valid if r["y"] == 1]
        fp_v = [r[f"fl_p{j}"] for r in valid if r["y"] == 0]
        if scores.std() > 0:
            r_pearson, _ = stats.pearsonr(scores, fls)
        else:
            r_pearson = 0.0
        delta = float(np.mean(tp_v) - np.mean(fp_v)) if tp_v and fp_v else 0.0
        # alpha estimate: shift / score
        # use diff against blind impractical here; report Pearson r as anchoring proxy
        info = {"paraphrase_id": j, "n_valid": len(valid),
                "score_correlation_r": round(float(r_pearson), 4),
                "delta_tp_minus_fp": round(delta, 4),
                "fl_mean": round(float(fls.mean()), 4)}
        summary["by_paraphrase"].append(info)
        alphas.append(r_pearson)
        deltas.append(delta)
    if alphas:
        summary["r_mean"] = round(float(np.mean(alphas)), 4)
        summary["r_std"] = round(float(np.std(alphas)), 4)
        summary["r_max_min_spread"] = round(float(max(alphas) - min(alphas)), 4)
        summary["delta_mean"] = round(float(np.mean(deltas)), 4)
        summary["delta_std"] = round(float(np.std(deltas)), 4)
    with open(RESULTS / "prompt_paraphrase_audit.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Anchoring r across paraphrases: mean={summary.get('r_mean','-')}, "
          f"std={summary.get('r_std','-')}, spread={summary.get('r_max_min_spread','-')}")
    return summary


# =====================================================================
# Tier 3.6: Score format robustness (4 formats)
# =====================================================================
SCORE_FORMATS = {
    "decimal": "ML Model Score: Fraud probability: {score:.4f}",
    "percent": "ML Model Score: Fraud probability: {pct:.1f}%",
    "verbal": "ML Model Score: {label} confidence ({pct:.0f}%)",
    "ratio": "ML Model Score: odds {odds_a}:{odds_b} (fraud:legit)",
}


def fmt_score(fmt, score):
    pct = score * 100
    if fmt == "decimal":
        return SCORE_FORMATS[fmt].format(score=score)
    if fmt == "percent":
        return SCORE_FORMATS[fmt].format(pct=pct)
    if fmt == "verbal":
        if score >= 0.85:
            label = "VERY HIGH"
        elif score >= 0.7:
            label = "HIGH"
        elif score >= 0.5:
            label = "MEDIUM"
        elif score >= 0.3:
            label = "LOW"
        else:
            label = "VERY LOW"
        return SCORE_FORMATS[fmt].format(label=label, pct=pct)
    if fmt == "ratio":
        if score <= 0.99 and score >= 0.01:
            a = int(round(score * 100))
            b = 100 - a
        else:
            a, b = (99, 1) if score > 0.5 else (1, 99)
        return SCORE_FORMATS[fmt].format(odds_a=a, odds_b=b)


def run_score_format_robustness(n_tp=15, n_fp=15):
    print("\n" + "="*60)
    print("Tier 3.6: Score format robustness (4 formats)")
    print("="*60)
    tps, fps = load_paired_items(n_tp, n_fp)
    selected = tps + fps

    def tpl(text, score_line):
        return f"""Assess this transaction for fraud risk.

Transaction Details:
{text}

{score_line}

Respond in JSON: {{"fraud_likelihood": 0.0-1.0, "recommendation": "RELEASE|ESCALATE"}}"""

    formats = list(SCORE_FORMATS.keys())
    rows = []
    for i, item in enumerate(selected):
        text = item["original"]["text"]
        score = item["original"]["fraud_score"]
        y = item["original"]["is_fraud"]
        rec = {"y": y, "score": score}
        for fmt in formats:
            line = fmt_score(fmt, score)
            rec[f"fl_{fmt}"] = parse_fl(call_oai(tpl(text, line)))
        rows.append(rec)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(selected)}]")
    with open(RESULTS / "score_format_raw.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    summary = {"n": len(rows), "by_format": {}}
    rs = []
    for fmt in formats:
        valid = [r for r in rows if r[f"fl_{fmt}"] is not None]
        if not valid:
            continue
        scores = np.array([r["score"] for r in valid])
        fls = np.array([r[f"fl_{fmt}"] for r in valid])
        tp_v = [r[f"fl_{fmt}"] for r in valid if r["y"] == 1]
        fp_v = [r[f"fl_{fmt}"] for r in valid if r["y"] == 0]
        if scores.std() > 0:
            r_pearson, _ = stats.pearsonr(scores, fls)
        else:
            r_pearson = 0.0
        delta = float(np.mean(tp_v) - np.mean(fp_v)) if tp_v and fp_v else 0.0
        summary["by_format"][fmt] = {
            "n_valid": len(valid),
            "score_correlation_r": round(float(r_pearson), 4),
            "delta_tp_minus_fp": round(delta, 4),
            "fl_mean": round(float(fls.mean()), 4),
        }
        rs.append(r_pearson)
    if rs:
        summary["r_mean"] = round(float(np.mean(rs)), 4)
        summary["r_max_min_spread"] = round(float(max(rs) - min(rs)), 4)
    with open(RESULTS / "score_format_audit.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Anchoring r by format: {[(k, v['score_correlation_r']) for k,v in summary['by_format'].items()]}")
    return summary


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="all",
                        help="comma-separated subset: mit,prior,cross,reason,prompt,format")
    parser.add_argument("--small", action="store_true", help="reduce n for debug")
    args = parser.parse_args()

    n_tp = 5 if args.small else 25
    n_fp = 5 if args.small else 25

    todo = args.only.split(",") if args.only != "all" else ["mit", "prior", "cross", "reason", "prompt", "format"]
    out = {}
    if "mit" in todo:
        out["mitigation_zoo"] = run_mitigation_zoo(n_tp, n_fp)
    if "prior" in todo:
        out["prior_dose_response"] = run_prior_dose_response(n_tp)
    if "cross" in todo:
        out["claude_gemini"] = run_claude_gemini_ablation(n_tp, n_fp)
    if "reason" in todo:
        out["reasoning"] = run_reasoning_models(min(15, n_tp), min(15, n_fp))
    if "prompt" in todo:
        out["prompt_paraphrase"] = run_prompt_paraphrase_robustness(min(15, n_tp), min(15, n_fp))
    if "format" in todo:
        out["score_format"] = run_score_format_robustness(min(15, n_tp), min(15, n_fp))

    with open(RESULTS / "extension_battery_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n" + "="*60)
    print("ALL TIER 1-3 EXPERIMENTS COMPLETE")
    print("="*60)
    print(f"Summary: experiments/results/extension_battery_summary.json")


if __name__ == "__main__":
    main()
