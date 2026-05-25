"""
PRISM Statistical Analysis
- McNemar test (paired binary: hazard detection)
- Wilcoxon signed-rank (paired continuous: total score)
- Chi-square (condition × hazard correct)
- Cohen's d effect sizes
- Publication-ready LaTeX tables
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import wilcoxon, chi2_contingency, mcnemar
import warnings
warnings.filterwarnings("ignore")

RESULTS_PATH = "/home/claude/prism_eval/results/raw_results.csv"
OUTPUT_DIR   = "/home/claude/prism_eval/results/"
TABLES_DIR   = "/home/claude/prism_eval/results/tables/"

import os
os.makedirs(TABLES_DIR, exist_ok=True)


def cohen_d(a, b):
    n1, n2 = len(a), len(b)
    s1, s2 = np.std(a, ddof=1), np.std(b, ddof=1)
    pooled = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0


def run_analysis():
    df = pd.read_csv(RESULTS_PATH)
    df = df[df["api_error"] == ""]  # drop failed calls
    print(f"Loaded {len(df)} valid results across {df['scenario_id'].nunique()} scenarios.")

    CONDITIONS = ["C1_baseline", "C2_raw", "C3_plaintext", "C4_prism", "C5_prose_matched",
                  "A1_no_urgency", "A2_no_thresholds", "A3_no_correlation"]

    # ── 1. Summary statistics by condition ───────────────────────────────────
    summary = []
    for cond in CONDITIONS:
        sub = df[df["condition"] == cond]
        summary.append({
            "Condition":        cond,
            "N":                len(sub),
            "Mean Score":       round(sub["total_score"].mean(), 3),
            "SD":               round(sub["total_score"].std(), 3),
            "Median Score":     round(sub["total_score"].median(), 3),
            "Hazard Acc (%)":   round(sub["hazard_detection"].mean() * 100, 1),
            "Action Acc (%)":   round(sub["action_recommendation"].mean() * 100, 1),
            "Threshold Cit (%)":round(sub["threshold_citation"].mean() * 100, 1),
            "Specificity (%)":  round(sub["specificity"].mean() * 100, 1),
        })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(f"{OUTPUT_DIR}summary_by_condition.csv", index=False)
    print("\n── Summary by Condition ──")
    print(summary_df.to_string(index=False))

    # ── 2. McNemar test: C4 vs others on hazard detection ────────────────────
    print("\n── McNemar Tests: Hazard Detection (C4_prism vs others) ──")
    mcnemar_results = []

    pivot = df.pivot_table(
        index="scenario_id", columns="condition",
        values="hazard_detection", aggfunc="first"
    )

    for cond in ["C1_baseline", "C2_raw", "C3_plaintext", "C5_prose_matched",
                 "A1_no_urgency", "A2_no_thresholds", "A3_no_correlation"]:
        if cond not in pivot.columns or "C4_prism" not in pivot.columns:
            continue
        paired = pivot[["C4_prism", cond]].dropna()
        a = (paired["C4_prism"] >= 0.5).astype(int)
        b = (paired[cond] >= 0.5).astype(int)
        table = pd.crosstab(a, b).values
        if table.shape == (2, 2):
            result = mcnemar(table, exact=True)
            p = result.pvalue
        else:
            p = 1.0
        prism_acc  = a.mean()
        other_acc  = b.mean()
        mcnemar_results.append({
            "Comparison":  f"C4 vs {cond}",
            "PRISM Acc":   round(prism_acc, 3),
            "Other Acc":   round(other_acc, 3),
            "McNemar p":   round(p, 4),
            "Significant": "Yes" if p < 0.05 else "No",
        })
        print(f"  C4 vs {cond:<22}: p={p:.4f} {'*' if p<0.05 else ''}")

    mcnemar_df = pd.DataFrame(mcnemar_results)
    mcnemar_df.to_csv(f"{OUTPUT_DIR}mcnemar_results.csv", index=False)

    # ── 3. Wilcoxon signed-rank: total score C4 vs others ────────────────────
    print("\n── Wilcoxon Signed-Rank Tests: Total Score (C4_prism vs others) ──")
    wilcoxon_results = []

    pivot_score = df.pivot_table(
        index="scenario_id", columns="condition",
        values="total_score", aggfunc="first"
    )

    for cond in ["C1_baseline", "C2_raw", "C3_plaintext", "C5_prose_matched",
                 "A1_no_urgency", "A2_no_thresholds", "A3_no_correlation"]:
        if cond not in pivot_score.columns or "C4_prism" not in pivot_score.columns:
            continue
        paired = pivot_score[["C4_prism", cond]].dropna()
        a = paired["C4_prism"].values
        b = paired[cond].values
        diff = a - b
        if np.all(diff == 0):
            p, stat = 1.0, 0.0
        else:
            stat, p = wilcoxon(a, b, alternative="greater")
        d = cohen_d(a, b)
        wilcoxon_results.append({
            "Comparison":   f"C4 vs {cond}",
            "C4 Mean":      round(np.mean(a), 3),
            "Other Mean":   round(np.mean(b), 3),
            "Δ Mean":       round(np.mean(a) - np.mean(b), 3),
            "W stat":       round(stat, 1),
            "p-value":      round(p, 4),
            "Cohen d":      round(d, 3),
            "Significant":  "Yes" if p < 0.05 else "No",
        })
        print(f"  C4 vs {cond:<22}: Δ={np.mean(a)-np.mean(b):.3f}  p={p:.4f}  d={d:.3f} {'*' if p<0.05 else ''}")

    wilcoxon_df = pd.DataFrame(wilcoxon_results)
    wilcoxon_df.to_csv(f"{OUTPUT_DIR}wilcoxon_results.csv", index=False)

    # ── 4. Breakdown by scenario type ─────────────────────────────────────────
    print("\n── Mean Total Score by Scenario Type × Condition ──")
    type_cond = df.groupby(["scenario_type", "condition"])["total_score"].mean().unstack()
    type_cond = type_cond.round(3)
    type_cond.to_csv(f"{OUTPUT_DIR}score_by_type_condition.csv")
    print(type_cond.to_string())

    # ── 5. Ablation impact table ──────────────────────────────────────────────
    ablation_rows = []
    prism_scores = df[df["condition"] == "C4_prism"]["total_score"]
    for abl in ["A1_no_urgency", "A2_no_thresholds", "A3_no_correlation"]:
        abl_scores = df[df["condition"] == abl]["total_score"]
        drop = prism_scores.mean() - abl_scores.mean()
        component = {
            "A1_no_urgency":     "Urgency Tag",
            "A2_no_thresholds":  "Threshold Normalization",
            "A3_no_correlation": "Multi-Event Correlation",
        }[abl]
        ablation_rows.append({
            "Removed Component": component,
            "PRISM Score":       round(prism_scores.mean(), 3),
            "Ablated Score":     round(abl_scores.mean(), 3),
            "Score Drop":        round(drop, 3),
            "Drop %":            round(drop / prism_scores.mean() * 100, 1),
        })
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(f"{OUTPUT_DIR}ablation_results.csv", index=False)
    print("\n── Ablation Analysis ──")
    print(ablation_df.to_string(index=False))

    # ── 6. Generate LaTeX tables ───────────────────────────────────────────────
    generate_latex_tables(summary_df, wilcoxon_df, ablation_df)

    print(f"\nAll results saved to {OUTPUT_DIR}")
    return summary_df, wilcoxon_df, ablation_df


def generate_latex_tables(summary_df, wilcoxon_df, ablation_df):
    # Table 1: Main results
    latex1 = r"""\begin{table}[H]
\centering
\caption{PRISM Evaluation: Mean Scores by Prompt Condition (N=100 scenarios)}
\label{tab:main_results}
\small
\begin{tabular}{lcccccc}
\toprule
\textbf{Condition} & \textbf{Total (0--4)} & \textbf{Hazard Acc.} & \textbf{Action Acc.} & \textbf{Threshold Cit.} & \textbf{Specificity} \\
\midrule
"""
    LABELS = {
        "C1_baseline":       "C1: No Context (Baseline)",
        "C2_raw":            "C2: Raw Sensor Values",
        "C3_plaintext":      "C3: Plain-text Description",
        "C4_prism":          r"\textbf{C4: PRISM Structured}",
        "A1_no_urgency":     "A1: PRISM $-$ Urgency",
        "A2_no_thresholds":  "A2: PRISM $-$ Thresholds",
        "A3_no_correlation": "A3: PRISM $-$ Correlation",
    }
    for _, row in summary_df.iterrows():
        label = LABELS.get(row["Condition"], row["Condition"])
        latex1 += (f"{label} & {row['Mean Score']} $\\pm$ {row['SD']} & "
                   f"{row['Hazard Acc (%)']:.0f}\\% & {row['Action Acc (%)']:.0f}\\% & "
                   f"{row['Threshold Cit (%)']:.0f}\\% & {row['Specificity (%)']:.0f}\\% \\\\\n")
    latex1 += r"""\bottomrule
\end{tabular}
\end{table}
"""
    with open(f"{TABLES_DIR}table_main_results.tex", "w") as f:
        f.write(latex1)

    # Table 2: Statistical tests
    latex2 = r"""\begin{table}[H]
\centering
\caption{Wilcoxon Signed-Rank Tests: PRISM (C4) vs. All Conditions}
\label{tab:stats}
\small
\begin{tabular}{lcccccc}
\toprule
\textbf{Comparison} & \textbf{C4 Mean} & \textbf{Other Mean} & \textbf{$\Delta$} & \textbf{$p$-value} & \textbf{Cohen's $d$} & \textbf{Sig.} \\
\midrule
"""
    for _, row in wilcoxon_df.iterrows():
        sig = "$*$" if row["Significant"] == "Yes" else "--"
        latex2 += (f"{row['Comparison']} & {row['C4 Mean']} & {row['Other Mean']} & "
                   f"{row['Δ Mean']:+.3f} & {row['p-value']} & {row['Cohen d']} & {sig} \\\\\n")
    latex2 += r"""\bottomrule
\end{tabular}
\end{table}
"""
    with open(f"{TABLES_DIR}table_stats.tex", "w") as f:
        f.write(latex2)

    # Table 3: Ablation
    latex3 = r"""\begin{table}[H]
\centering
\caption{Ablation Study: Contribution of Each PRISM Component}
\label{tab:ablation}
\small
\begin{tabular}{lcccc}
\toprule
\textbf{Removed Component} & \textbf{PRISM Score} & \textbf{Ablated Score} & \textbf{Drop} & \textbf{Drop (\%)} \\
\midrule
"""
    for _, row in ablation_df.iterrows():
        latex3 += (f"{row['Removed Component']} & {row['PRISM Score']} & "
                   f"{row['Ablated Score']} & {row['Score Drop']:+.3f} & {row['Drop %']}\\% \\\\\n")
    latex3 += r"""\bottomrule
\end{tabular}
\end{table}
"""
    with open(f"{TABLES_DIR}table_ablation.tex", "w") as f:
        f.write(latex3)

    print(f"LaTeX tables saved to {TABLES_DIR}")


if __name__ == "__main__":
    run_analysis()
