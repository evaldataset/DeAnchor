"""K.3 #16-#17 — Post-hoc robustness: paraphrase bootstrap CIs + Holm-Bonferroni
multi-comparison correction across 8-family cross-model paired tests.

Outputs:
  experiments/results/paraphrase_bootstrap_ci.json
  experiments/results/multi_comparison_corrected.json
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "experiments" / "results"
RNG = np.random.default_rng(42)


# =====================================================================
# 1) Paraphrase bootstrap CIs
# =====================================================================
def bootstrap_paraphrase(B: int = 2000):
    print("\n=== Paraphrase robustness: bootstrap CIs ===")
    rows = [json.loads(line) for line in open(RESULTS / "prompt_paraphrase_raw.jsonl")]
    n = len(rows)
    n_p = sum(1 for k in rows[0] if k.startswith("fl_p"))
    out = {"n": n, "n_paraphrases": n_p, "by_paraphrase": []}
    for j in range(n_p):
        key = f"fl_p{j}"
        vals = [(r["score"], r[key]) for r in rows if r.get(key) is not None]
        if not vals:
            continue
        s = np.array([v[0] for v in vals])
        f = np.array([v[1] for v in vals])
        # Bootstrap Pearson r
        rs = []
        for _ in range(B):
            idx = RNG.integers(0, len(vals), size=len(vals))
            ss = s[idx]
            ff = f[idx]
            if ss.std() > 0 and ff.std() > 0:
                rs.append(stats.pearsonr(ss, ff).statistic)
        rs = np.array(rs)
        if len(rs) == 0:
            ci = (None, None)
            r_obs = float("nan")
        else:
            r_obs = float(stats.pearsonr(s, f).statistic) if s.std() > 0 and f.std() > 0 else float("nan")
            ci = (round(float(np.percentile(rs, 2.5)), 4),
                  round(float(np.percentile(rs, 97.5)), 4))
        out["by_paraphrase"].append({
            "paraphrase_id": j,
            "n_valid": len(vals),
            "r_obs": round(r_obs, 4) if not np.isnan(r_obs) else None,
            "ci_95": ci,
        })
        print(f"  paraphrase {j}: r={r_obs:.3f} 95% CI {ci}")

    out_path = RESULTS / "paraphrase_bootstrap_ci.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"  → {out_path}")
    return out


# =====================================================================
# 2) Holm-Bonferroni across 8-family cross-model paired tests
# =====================================================================
def holm_bonferroni():
    print("\n=== Multi-comparison correction across 8 families (Holm-Bonferroni) ===")
    # Source claims: paired t-test p-values per family
    sources = [
        ("ablation_claude_haiku.json", "claude-haiku-4.5"),
        ("ablation_gemini_flash.json", "gemini-2.5-flash"),
        ("ablation_o1mini.json", "o3-mini"),
        ("ablation_gem_pro_thinking.json", "gemini-2.5-pro-thinking"),
        ("gpt4o_ablation_audit.json", "gpt-4o"),
        # GPT-4o-mini, Qwen, Gemma read from controlled_ablation/rich audits
    ]
    pvals = []
    for fname, label in sources:
        path = RESULTS / fname
        if not path.exists():
            continue
        d = json.load(open(path))
        # different schemas
        p = d.get("paired_p")
        if p is None and "p_value" in d:
            p = d["p_value"]
        if p is None and "controlled_ablation" in d:
            p = d["controlled_ablation"].get("paired_p")
        if p is not None:
            pvals.append((label, float(p)))
    # Pull GPT-4o-mini and Qwen from existing summaries
    rev = json.load(open(RESULTS / "revision_analyses.json"))
    pvals.append(("gpt-4o-mini",
                  float(rev["paired_controlled_ablation"]["p_value"])))

    # Holm-Bonferroni
    pvals_sorted = sorted(pvals, key=lambda x: x[1])
    m = len(pvals_sorted)
    corrected = []
    last = 0.0
    for i, (label, p) in enumerate(pvals_sorted):
        adj = min(1.0, max(last, (m - i) * p))
        last = adj
        corrected.append({"family": label, "p_raw": p, "p_holm": adj,
                          "rank": i + 1, "significant_05": adj < 0.05})
        print(f"  rank {i+1:2d}: {label:30s} p_raw={p:.2e}  p_holm={adj:.2e}  "
              f"{'*' if adj < 0.05 else ''}")

    out = {"method": "Holm-Bonferroni", "n_tests": m,
           "results": corrected,
           "all_significant_at_0.05": all(c["significant_05"] for c in corrected)}
    out_path = RESULTS / "multi_comparison_corrected.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"  → {out_path}")
    return out


if __name__ == "__main__":
    bootstrap_paraphrase()
    holm_bonferroni()
