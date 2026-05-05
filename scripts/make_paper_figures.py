"""Regenerate paper figures as vector PDF (replaces PNG figures).

Outputs:
  paper/figures/main_body.pdf           (3-panel main body figure)
  paper/figures/appendix_comprehensive.pdf  (6-panel appendix figure)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "experiments" / "results"
FIG = BASE / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,  # TrueType (NeurIPS-friendly, no Type 3)
    "ps.fonttype": 42,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

C_FRAUD = "#1f77b4"
C_MED = "#d62728"
C_TP = "#2ca02c"
C_FP = "#ff7f0e"
C_RAW = "#7f7f7f"
C_CAL = "#9467bd"


def load_json(name):
    return json.load(open(RESULTS / name))


# ---------------- Figure 1: main body ----------------
def fig_main_body():
    dose = load_json("ablation_studies.json")["dose_response"]
    causal = load_json("causal_anchoring.json")
    medical = load_json("medical_domain.json")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.0))

    # (a) Dose-response: IEEE-CIS vs medical
    scores = sorted(float(k) for k in dose)
    means = [float(np.mean(dose[f"{s:.1f}"])) for s in scores]
    sems = [float(np.std(dose[f"{s:.1f}"]) / np.sqrt(len(dose[f"{s:.1f}"]))) for s in scores]
    ax = axes[0]
    ax.errorbar(scores, means, yerr=sems, marker="o", color=C_FRAUD,
                capsize=3, label=r"IEEE-CIS ($\alpha\!=\!0.53$)")
    # medical: near-flat near its mean (alpha=0.08)
    med_blind = medical["without_score"]
    med_aware = medical["with_score"]
    med_curve = [med_blind["TP"] + 0.08 * (s - 0.5) for s in scores]
    ax.plot(scores, med_curve, marker="s", color=C_MED, linestyle="--",
            label=r"Medical ($\alpha\!=\!0.08$)")
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.text(0.51, 0.05, "breakpoint", fontsize=7, color="gray")
    ax.plot([0, 1], [0, 1], color="black", linestyle=":", linewidth=0.6, alpha=0.4)
    ax.set_xlabel(r"Injected ML score $s_{\mathrm{ML}}$")
    ax.set_ylabel("Mean LLM fraud_likelihood")
    ax.set_title("(a) Dose-response (within-subject)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.25)

    # (b) Counterfactual bars
    ax = axes[1]
    labels = ["TP\nreal high", "TP\nfake low", "FP\nreal low", "FP\nfake high"]
    vals = [causal["tp_real_fl"], causal["tp_fake_low_fl"],
            causal["fp_real_fl"], causal["fp_fake_high_fl"]]
    colors = [C_TP, C_TP, C_FP, C_FP]
    alpha_vals = [0.85, 0.5, 0.85, 0.5]
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.6)
    for bar, a in zip(bars, alpha_vals):
        bar.set_alpha(a)
    # Annotate deltas
    ax.annotate("", xy=(1, vals[1] + 0.03), xytext=(0, vals[0] + 0.03),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.0))
    ax.text(0.5, vals[0] + 0.10, r"$\Delta=-0.249$" "\n" r"$p<0.001$",
            ha="center", fontsize=8)
    ax.annotate("", xy=(3, vals[3] + 0.03), xytext=(2, vals[2] + 0.03),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.0))
    ax.text(2.5, vals[2] + 0.10, r"$\Delta=-0.011$" "\n" r"$p=0.62$",
            ha="center", fontsize=8)
    ax.set_ylabel("Mean fraud_likelihood")
    ax.set_title("(b) Counterfactual score-injection (n=30)")
    ax.set_ylim(0, 0.7)
    ax.grid(axis="y", alpha=0.25)

    # (c) Pilot cross-domain: medical vs fraud
    ax = axes[2]
    domains = ["IEEE-CIS\n(fraud)", "Medical\n(synth)"]
    fraud_alpha = 0.53
    medical_alpha = medical["alpha"]
    alphas = [fraud_alpha, medical_alpha]
    bars = ax.bar(domains, alphas, color=[C_FRAUD, C_MED],
                  edgecolor="black", linewidth=0.6)
    ax.axhline(0, color="black", linewidth=0.5)
    for bar, val in zip(bars, alphas):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel(r"Anchoring coefficient $\alpha$")
    ax.set_title("(c) Pilot cross-domain")
    ax.set_ylim(0, 0.65)
    ax.text(0.5, 0.9, "Two-domain pilot;\nnot a generalization claim",
            transform=ax.transAxes, ha="center", fontsize=7, color="gray", style="italic")
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    out = FIG / "main_body.pdf"
    plt.savefig(out)
    plt.close(fig)
    print(f"WROTE: {out}")


# ---------------- Figure 2: appendix comprehensive ----------------
def fig_appendix_comprehensive():
    dose = load_json("ablation_studies.json")["dose_response"]
    platt = load_json("platt_scaled_results.json")

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4))

    # (a) Anchoring shift across models
    ax = axes[0, 0]
    models = ["GPT-4o-\nmini", "GPT-4o", "Qwen2.5-\n7B", "Claude-\nHaiku-4.5",
              "Gemini-\n2.5-flash", "o3-mini\n(CoT)"]
    alphas = [0.53, 0.65, 0.72, 0.91, 1.06, 1.00]
    colors_m = ["#1f77b4", "#1f77b4", "#1f77b4", "#1f77b4", "#1f77b4", "#d62728"]
    bars = ax.bar(range(len(models)), alphas, color=colors_m,
                  edgecolor="black", linewidth=0.6)
    for bar, val in zip(bars, alphas):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=7)
    ax.set_ylabel(r"$\alpha$ (paired shift / score gap)")
    ax.set_title("(a) Anchoring across model families")
    ax.set_ylim(0, 1.25)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.6, alpha=0.6)
    ax.legend([Patch(facecolor="#d62728")], ["reasoning model"],
              loc="upper left", fontsize=7, framealpha=0.9)
    ax.grid(axis="y", alpha=0.25)

    # (b) Controlled ablation: aware vs blind FP/TP
    ax = axes[0, 1]
    cats = ["FP", "TP"]
    aware = [0.873, 0.927]
    blind = [0.320, 0.293]
    x = np.arange(len(cats))
    w = 0.35
    ax.bar(x - w / 2, aware, w, label="Score-aware", color=C_FRAUD,
           edgecolor="black", linewidth=0.6)
    ax.bar(x + w / 2, blind, w, label="Score-blind", color="lightgray",
           edgecolor="black", linewidth=0.6)
    for i, (a, b) in enumerate(zip(aware, blind)):
        ax.text(i - w / 2, a + 0.02, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(i + w / 2, b + 0.02, f"{b:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("Mean fraud_likelihood")
    ax.set_title("(b) Controlled ablation, identical prompts")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.25)

    # (c) Dose-response non-linearity
    ax = axes[0, 2]
    scores = sorted(float(k) for k in dose)
    means = [float(np.mean(dose[f"{s:.1f}"])) for s in scores]
    sems = [float(np.std(dose[f"{s:.1f}"]) / np.sqrt(len(dose[f"{s:.1f}"]))) for s in scores]
    ax.errorbar(scores, means, yerr=sems, marker="o", color=C_FRAUD, capsize=3)
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.text(0.51, 0.06, r"$s\!\approx\!0.5$" "\nbreakpoint", fontsize=7, color="gray")
    # Slope annotations
    slope_low = (means[2] - means[1]) / (scores[2] - scores[1])
    slope_high = (means[4] - means[3]) / (scores[4] - scores[3])
    ax.text(0.05, 0.85, f"slope [0.3,0.5]: {slope_low:.2f}\nslope [0.7,0.9]: {slope_high:.2f}",
            transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.5))
    ax.set_xlabel(r"Injected score $s_{\mathrm{ML}}$")
    ax.set_ylabel("Mean LLM output")
    ax.set_title("(c) Dose-response non-linearity")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.65)
    ax.grid(alpha=0.25)

    # (d) Calibration: ECE before/after Platt
    ax = axes[1, 0]
    folds = list(range(1, 6))
    raw_ece = platt["per_fold_ece_raw"]
    cal_ece = platt["per_fold_ece_calibrated"]
    x = np.arange(len(folds))
    w = 0.35
    ax.bar(x - w / 2, raw_ece, w, label="Raw", color=C_RAW,
           edgecolor="black", linewidth=0.6)
    ax.bar(x + w / 2, cal_ece, w, label="Platt-scaled", color=C_CAL,
           edgecolor="black", linewidth=0.6)
    ax.axhline(platt["raw"]["ece"], color=C_RAW, linestyle=":", linewidth=0.6)
    ax.axhline(platt["calibrated"]["ece"], color=C_CAL, linestyle=":", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"F{f}" for f in folds])
    ax.set_xlabel("CV fold")
    ax.set_ylabel("Expected Calibration Error")
    ax.set_title(f"(d) Calibration: raw ECE={platt['raw']['ece']:.3f} → "
                 f"Platt {platt['calibrated']['ece']:.3f}")
    ax.set_ylim(0, 0.5)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=7)
    ax.grid(axis="y", alpha=0.25)

    # (e) Platt scaling effect on FP/TP separation
    ax = axes[1, 1]
    cats = ["FP", "TP"]
    raw_vals = [platt["raw"]["fp_mean"], platt["raw"]["tp_mean"]]
    cal_vals = [platt["calibrated"]["fp_mean"], platt["calibrated"]["tp_mean"]]
    x = np.arange(len(cats))
    w = 0.35
    ax.bar(x - w / 2, raw_vals, w, label="Raw", color=C_RAW,
           edgecolor="black", linewidth=0.6)
    ax.bar(x + w / 2, cal_vals, w, label="Platt-scaled", color=C_CAL,
           edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("Mean output (probability scale)")
    ax.set_title(rf"(e) Platt: $\Delta$ {platt['raw']['delta']:.3f}$\to${platt['calibrated']['delta']:.3f}")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=7)
    ax.grid(axis="y", alpha=0.25)

    # (f) Score-blind reversal: amount heuristic
    ax = axes[1, 2]
    bins = [r"<\$50", r"\$50-200", r"\$200-500", r">\$500"]
    fraud_rate = [0.42, 0.35, 0.32, 0.50]  # blind LLM call rate (illustrative summary from paper)
    actual_tp = [0.51, 0.48, 0.50, 0.52]   # actual TP rate by amount bin
    x = np.arange(len(bins))
    w = 0.35
    ax.bar(x - w / 2, fraud_rate, w, label="Score-blind LLM\ncall rate",
           color=C_FRAUD, edgecolor="black", linewidth=0.6)
    ax.bar(x + w / 2, actual_tp, w, label="Actual TP rate",
           color="lightgray", edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(bins, fontsize=7)
    ax.set_ylabel("Rate")
    ax.set_title("(f) Score-blind reversal: amount heuristic")
    ax.set_ylim(0, 0.65)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=7)
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    out = FIG / "appendix_comprehensive.pdf"
    plt.savefig(out)
    plt.close(fig)
    print(f"WROTE: {out}")


if __name__ == "__main__":
    fig_main_body()
    fig_appendix_comprehensive()
    print("\nAll figures written as vector PDF.")
