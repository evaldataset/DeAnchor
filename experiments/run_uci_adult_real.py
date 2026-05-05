"""G.2 — Real UCI Adult re-run.

Replaces the synthetic UCI-Adult-style profiles in run_final_acceptance.py with
real UCI Adult Census data via sklearn.datasets.fetch_openml(adult, version=2).

Outputs:
  experiments/results/uci_adult_real_audit.json
  experiments/results/uci_adult_real_raw.jsonl
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.datasets import fetch_openml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm_explainer.llm_inference import _parse_json_response  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "experiments" / "results"
oai = OpenAI()


def call(prompt, system="", model="gpt-4o-mini"):
    resp = oai.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=400, seed=42,
    )
    return resp.choices[0].message.content.strip()


def parse_fl(text):
    p = _parse_json_response(text)
    if isinstance(p, dict):
        v = p.get("fraud_likelihood") or p.get("likelihood") or p.get("income_likelihood")
        if isinstance(v, (int, float)) and 0 <= v <= 1:
            return float(v)
    return None


def cv_auc(X, y, seed=42):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    preds = np.zeros(len(y))
    for tr, va in skf.split(X, y):
        lr = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
        preds[va] = lr.predict_proba(X[va])[:, 1]
    return float(roc_auc_score(y, preds))


def lr_test(X_full, X_red, y):
    lr_f = LogisticRegression(max_iter=1000, C=1e6).fit(X_full, y)
    lr_r = LogisticRegression(max_iter=1000, C=1e6).fit(X_red, y)
    pf = np.clip(lr_f.predict_proba(X_full)[:, 1], 1e-12, 1 - 1e-12)
    pr = np.clip(lr_r.predict_proba(X_red)[:, 1], 1e-12, 1 - 1e-12)
    ll_f = float(np.sum(y * np.log(pf) + (1 - y) * np.log(1 - pf)))
    ll_r = float(np.sum(y * np.log(pr) + (1 - y) * np.log(1 - pr)))
    chi2 = max(0.0, 2 * (ll_f - ll_r))
    p = float(1 - stats.chi2.cdf(chi2, 1))
    return {"chi2": round(chi2, 4), "p": p}


def profile_to_text(row):
    return (
        f"Person profile:\n"
        f"- Age: {row['age']}\n"
        f"- Education: {row['education']} ({row['education-num']} years)\n"
        f"- Occupation: {row['occupation']}\n"
        f"- Marital status: {row['marital-status']}\n"
        f"- Hours per week: {row['hours-per-week']}\n"
        f"- Capital gain: ${row['capital-gain']}, capital loss: ${row['capital-loss']}\n"
        f"- Workclass: {row['workclass']}, sex: {row['sex']}, race: {row['race']}\n"
        f"- Native country: {row['native-country']}"
    )


SYS = "You are a financial analyst assessing income for credit-risk purposes."

TPL_AWARE = """Predict whether this person earns more than $50K/year.

{text}

ML Model Score: P(income > $50K) = {score:.4f}

Respond in JSON: {{"fraud_likelihood": 0.0-1.0, "recommendation": "ELIGIBLE|NOT_ELIGIBLE"}} where the value is your estimated probability of >$50K income."""

TPL_BLIND = """Predict whether this person earns more than $50K/year.

{text}

Respond in JSON: {{"fraud_likelihood": 0.0-1.0, "recommendation": "ELIGIBLE|NOT_ELIGIBLE"}} where the value is your estimated probability of >$50K income."""


def run():
    print("=" * 60)
    print("G.2: Real UCI Adult re-run (n=100, GPT-4o-mini)")
    print("=" * 60)

    # Load real UCI Adult via sklearn (preserve raw rows for prompt generation)
    print("  Loading UCI Adult...")
    adult = fetch_openml("adult", version=2, as_frame=True, parser="pandas")
    raw_df = adult.data.copy()
    raw_df["target"] = (adult.target == ">50K").astype(int)
    raw_df = raw_df.dropna().reset_index(drop=True)
    print(f"  After dropna: {raw_df.shape}, positive rate {raw_df['target'].mean():.3f}")

    cat_cols = raw_df.select_dtypes(include=["object", "category"]).columns.tolist()
    feature_cols = [c for c in raw_df.columns if c != "target"]

    # Stratified split on row indices (preserves alignment between encoded
    # features and the human-readable rows used for prompt construction).
    idx = np.arange(len(raw_df))
    y_target = raw_df["target"].values
    train_idx, test_idx, y_tr, y_te = train_test_split(
        idx, y_target, test_size=0.3, random_state=42, stratify=y_target
    )

    # Encoders are fit on the train rows only (no test leakage).
    enc = {}
    for c in cat_cols:
        le = LabelEncoder()
        le.fit(raw_df.loc[train_idx, c].astype(str))
        # transform test with seen + unseen handling
        unseen = set(raw_df.loc[test_idx, c].astype(str)) - set(le.classes_)
        if unseen:
            le.classes_ = np.append(le.classes_, sorted(unseen))
        enc[c] = le

    def encode_rows(rows):
        out = rows.copy()
        for c in cat_cols:
            out[c] = enc[c].transform(out[c].astype(str))
        return out[feature_cols].values

    X_tr = encode_rows(raw_df.loc[train_idx])
    X_te = encode_rows(raw_df.loc[test_idx])
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42).fit(X_tr, y_tr)
    test_proba = gb.predict_proba(X_te)[:, 1]
    test_auc = roc_auc_score(y_te, test_proba)
    print(f"  GradientBoosting test AUROC (held-out) = {test_auc:.4f}")

    # Stratified subset of 50 positive + 50 negative from the held-out test
    rng = np.random.default_rng(42)
    pos_local = np.where(y_te == 1)[0]
    neg_local = np.where(y_te == 0)[0]
    rng.shuffle(pos_local)
    rng.shuffle(neg_local)
    sel_local = np.concatenate([pos_local[:50], neg_local[:50]])
    sel_y = y_te[sel_local]
    sel_proba = test_proba[sel_local]
    selected_test_idx = test_idx[sel_local]

    # Human-readable rows are pulled by ABSOLUTE row index — guarantees
    # that profile, label, and ML score all refer to the same person.
    test_rows = raw_df.loc[selected_test_idx].copy().reset_index(drop=True)
    # No need to inverse-transform: raw_df still has the original strings.

    rows = []
    for i, (_, r) in enumerate(test_rows.iterrows()):
        text = profile_to_text(r)
        score = float(sel_proba[i])
        y = int(sel_y[i])
        fla = parse_fl(call(TPL_AWARE.format(text=text, score=score), SYS))
        flb = parse_fl(call(TPL_BLIND.format(text=text), SYS))
        rows.append({"idx": i, "label": y, "ml_score": score,
                     "fl_aware": fla, "fl_blind": flb,
                     "text_preview": text[:80]})
        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/100]")

    valid = [r for r in rows if r["fl_aware"] is not None and r["fl_blind"] is not None]

    aware = np.array([r["fl_aware"] for r in valid])
    blind = np.array([r["fl_blind"] for r in valid])
    ml = np.array([r["ml_score"] for r in valid])
    y_arr = np.array([r["label"] for r in valid])

    auroc_M = cv_auc(ml.reshape(-1, 1), y_arr)
    auroc_A = cv_auc(aware.reshape(-1, 1), y_arr)
    auroc_B = cv_auc(blind.reshape(-1, 1), y_arr)
    auroc_F = cv_auc(np.column_stack([ml, aware]), y_arr)
    lrt = lr_test(np.column_stack([ml, aware]), ml.reshape(-1, 1), y_arr)
    diff = aware - blind
    alpha_num = float(diff.mean())
    alpha_den = float(ml.mean() - blind.mean())
    alpha = alpha_num / alpha_den if abs(alpha_den) > 1e-6 else None

    audit = {
        "domain": "uci_adult_real",
        "model": "gpt-4o-mini",
        "n": len(valid),
        "ml_baseline_auroc": round(test_auc, 4),
        "auroc_raw_ml": round(auroc_M, 4),
        "auroc_scoreaware": round(auroc_A, 4),
        "auroc_scoreblind": round(auroc_B, 4),
        "auroc_fusion": round(auroc_F, 4),
        "lr_test": lrt,
        "alpha": round(alpha, 4) if alpha is not None else None,
        "suppression": round(auroc_B - auroc_A, 4),
        "verdict": ("suppression" if auroc_A < auroc_B - 0.02
                    else ("rational" if auroc_B < 0.55 else "no_suppression")),
    }
    with open(RESULTS / "uci_adult_real_audit.json", "w") as f:
        json.dump(audit, f, indent=2)
    with open(RESULTS / "uci_adult_real_raw.jsonl", "w") as f:
        for r in valid:
            f.write(json.dumps(r) + "\n")
    print(f"\n  blind={auroc_B:.3f}, aware={auroc_A:.3f}, "
          f"shift={alpha_num:+.4f}, alpha={alpha}, "
          f"LR p={lrt['p']:.4f}, verdict={audit['verdict']}")
    return audit


if __name__ == "__main__":
    run()
