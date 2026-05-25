"""
PRISM Evaluation: Publication-Ready Plots
Generates all figures needed for the paper.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

RESULTS_PATH = "/home/claude/prism_eval/results/raw_results.csv"
PLOTS_DIR    = "/home/claude/prism_eval/plots/"

import os
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

CONDITIONS_ORDER = ["C1_baseline", "C2_raw", "C3_plaintext", "C4_prism", "C5_prose_matched",
                    "A1_no_urgency", "A2_no_thresholds", "A3_no_correlation"]

LABELS = {
    "C1_baseline":       "C1\nBaseline",
    "C2_raw":            "C2\nRaw Values",
    "C3_plaintext":      "C3\nPlain-text",
    "C4_prism":          "C4\nPRISM",
    "A1_no_urgency":     "A1\n−Urgency",
    "A2_no_thresholds":  "A2\n−Threshold",
    "A3_no_correlation": "A3\n−Correlation",
}

COLORS = {
    "C1_baseline":       "#9e9e9e",
    "C2_raw":            "#64b5f6",
    "C3_plaintext":      "#4fc3f7",
    "C4_prism":          "#1565c0",
    "A1_no_urgency":     "#ef9a9a",
    "A2_no_thresholds":  "#e57373",
    "A3_no_correlation": "#c62828",
    "C5_prose_matched":  "#7b1fa2",
}

DIMS = ["hazard_detection", "action_recommendation", "threshold_citation", "specificity"]
DIM_LABELS = ["Hazard\nDetection", "Action\nRecommendation", "Threshold\nCitation", "Specificity"]


def load_data():
    df = pd.read_csv(RESULTS_PATH)
    df = df[df["api_error"] == ""]
    return df


# ── Figure 1: Main bar chart — mean total score by condition ──────────────────
def plot_main_scores(df):
    means = []
    sems  = []
    for c in CONDITIONS_ORDER:
        sub = df[df["condition"] == c]["total_score"]
        means.append(sub.mean())
        sems.append(sub.sem())

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(CONDITIONS_ORDER))
    bars = ax.bar(x, means, yerr=sems, capsize=4, width=0.65,
                  color=[COLORS[c] for c in CONDITIONS_ORDER],
                  edgecolor="white", linewidth=0.5)

    # Highlight PRISM bar
    bars[3].set_edgecolor("#0d47a1")
    bars[3].set_linewidth(2)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in CONDITIONS_ORDER])
    ax.set_ylabel("Mean Total Score (0–4)")
    ax.set_title("Figure 1: PRISM vs. Control Conditions — Mean Total Score\n(error bars = SEM, N=100 scenarios per condition)")
    ax.set_ylim(0, 4.2)
    ax.axhline(y=means[3], color="#1565c0", linestyle="--", linewidth=0.8, alpha=0.4)

    # Annotate bars
    for bar, mean, sem in zip(bars, means, sems):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + sem + 0.05,
                f"{mean:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Divider between main conditions and ablations
    ax.axvline(x=4.5, color="#aaa", linestyle=":", linewidth=1)
    ax.text(2.0, 4.1, "Main Conditions", ha="center", fontsize=9, color="#555")
    ax.text(6.0, 4.1, "Ablations", ha="center", fontsize=9, color="#555")

    plt.tight_layout()
    path = f"{PLOTS_DIR}fig1_main_scores.pdf"
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved {path}")


# ── Figure 2: Radar / spider chart — 4 dimensions for C1, C3, C4 ─────────────
def plot_radar(df):
    focus = ["C1_baseline", "C3_plaintext", "C4_prism"]
    focus_labels = ["C1: Baseline", "C3: Plain-text", "C4: PRISM"]
    focus_colors = ["#9e9e9e", "#4fc3f7", "#1565c0"]

    angles = np.linspace(0, 2*np.pi, len(DIMS), endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})

    for cond, label, color in zip(focus, focus_labels, focus_colors):
        sub = df[df["condition"] == cond]
        vals = [sub[d].mean() for d in DIMS]
        vals += vals[:1]
        ax.plot(angles, vals, color=color, linewidth=2, label=label)
        ax.fill(angles, vals, color=color, alpha=0.12)

    ax.set_thetagrids(np.degrees(angles[:-1]), DIM_LABELS)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8)
    ax.set_title("Figure 2: Per-Dimension Performance Profile\n(Baseline vs. Plain-text vs. PRISM)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))

    plt.tight_layout()
    path = f"{PLOTS_DIR}fig2_radar.pdf"
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved {path}")


# ── Figure 3: Score by scenario type — C1 vs C3 vs C4 ────────────────────────
def plot_by_scenario_type(df):
    focus = ["C1_baseline", "C3_plaintext", "C4_prism"]
    focus_labels = ["C1: Baseline", "C3: Plain-text", "C4: PRISM"]
    focus_colors = ["#9e9e9e", "#4fc3f7", "#1565c0"]

    types = sorted(df["scenario_type"].unique())
    x = np.arange(len(types))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (cond, label, color) in enumerate(zip(focus, focus_labels, focus_colors)):
        means = [df[(df["condition"]==cond)&(df["scenario_type"]==t)]["total_score"].mean()
                 for t in types]
        ax.bar(x + i*width, means, width, label=label, color=color, alpha=0.9)

    ax.set_xticks(x + width)
    ax.set_xticklabels([t.replace("_", "\n") for t in types], fontsize=9)
    ax.set_ylabel("Mean Total Score (0–4)")
    ax.set_title("Figure 3: PRISM Performance by Scenario Type")
    ax.set_ylim(0, 4.3)
    ax.legend()

    plt.tight_layout()
    path = f"{PLOTS_DIR}fig3_by_scenario_type.pdf"
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved {path}")


# ── Figure 4: Ablation waterfall ─────────────────────────────────────────────
def plot_ablation_waterfall(df):
    ablations = ["C4_prism", "A1_no_urgency", "A2_no_thresholds", "A3_no_correlation"]
    abl_labels = ["Full\nPRISM", "−Urgency\nTag", "−Threshold\nNorm.", "−Correlation\nTags"]
    colors = ["#1565c0", "#ef9a9a", "#e57373", "#c62828"]

    means = [df[df["condition"]==c]["total_score"].mean() for c in ablations]
    sems  = [df[df["condition"]==c]["total_score"].sem()  for c in ablations]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(ablations))
    bars = ax.bar(x, means, yerr=sems, capsize=4, color=colors,
                  width=0.55, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(abl_labels)
    ax.set_ylabel("Mean Total Score (0–4)")
    ax.set_title("Figure 4: Ablation Study — Contribution of Each PRISM Component")
    ax.set_ylim(0, 4.2)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.04,
                f"{mean:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Draw drop arrows from C4 to each ablation
    for i in range(1, len(ablations)):
        drop = means[0] - means[i]
        ax.annotate(f"−{drop:.2f}", xy=(i, means[i]+0.15),
                    xytext=(i, means[0]-0.15),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1.2),
                    ha="center", fontsize=8.5, color="#555")

    plt.tight_layout()
    path = f"{PLOTS_DIR}fig4_ablation.pdf"
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved {path}")


# ── Figure 5: Dimension breakdown stacked bar ─────────────────────────────────
def plot_dimension_breakdown(df):
    dim_colors = ["#1565c0", "#42a5f5", "#90caf9", "#bbdefb"]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(CONDITIONS_ORDER))
    bottom = np.zeros(len(CONDITIONS_ORDER))

    for dim, color, label in zip(DIMS, dim_colors, DIM_LABELS):
        means = [df[df["condition"]==c][dim].mean() for c in CONDITIONS_ORDER]
        ax.bar(x, means, bottom=bottom, color=color, label=label.replace("\n"," "), width=0.6)
        bottom += np.array(means)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in CONDITIONS_ORDER])
    ax.set_ylabel("Mean Score Contribution")
    ax.set_title("Figure 5: Score Composition by Dimension and Condition")
    ax.set_ylim(0, 4.3)
    ax.legend(loc="upper left", framealpha=0.9)

    plt.tight_layout()
    path = f"{PLOTS_DIR}fig5_dimension_breakdown.pdf"
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved {path}")


def generate_all_plots(df):
    print("Generating publication plots...")
    plot_main_scores(df)
    plot_radar(df)
    plot_by_scenario_type(df)
    plot_ablation_waterfall(df)
    plot_dimension_breakdown(df)
    print(f"All plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    df = load_data()
    generate_all_plots(df)
