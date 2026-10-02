"""
Workstream 1 - Diagnostics & Exploratory Data Analysis
Project 5: Jio Subscriber Churn Intelligence & Revenue Protection

Tasks:
1. Circle churn heatmap & analysis
2. Recharge gap analysis (silent churn threshold)
3. ARPU band price sensitivity
4. Network KPI correlation with 30-day churn

Produces console diagnostics plus a single PDF report with all charts and a
summary findings page at Reports/workstream1_diagnostics.pdf.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; charts are saved to PDF, not displayed

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

sns.set_theme(style="whitegrid")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "Data" / "subscribers.csv"
REPORT_PATH = BASE_DIR / "Reports" / "workstream1_diagnostics.pdf"


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, parse_dates=["join_date", "churn_date"])


def circle_analysis(df: pd.DataFrame, pdf: PdfPages) -> pd.DataFrame:
    """Task 1: circle-level subscriber count, avg ARPU, and 30-day churn rate."""
    circle_stats = (
        df.groupby("circle")
        .agg(
            subscriber_count=("subscriber_id", "count"),
            avg_arpu=("arpu_last_month_inr", "mean"),
            churn_rate_30d=("churn_flag_30d", "mean"),
        )
        .sort_values("churn_rate_30d", ascending=False)
    )
    circle_stats["churn_rate_30d_pct"] = circle_stats["churn_rate_30d"] * 100

    print("\n=== 1. Circle Churn Analysis (sorted by churn concentration) ===")
    print(circle_stats.round(2).to_string())

    # Normalize metrics 0-1 for a comparable heatmap, but annotate with real values
    heat_data = circle_stats[["subscriber_count", "avg_arpu", "churn_rate_30d_pct"]]
    heat_norm = (heat_data - heat_data.min()) / (heat_data.max() - heat_data.min())

    fig, ax = plt.subplots(figsize=(8, max(6, 0.35 * len(heat_norm))))
    sns.heatmap(heat_norm, annot=heat_data.round(1), fmt="", cmap="Reds", ax=ax,
                cbar_kws={"label": "Normalized scale"})
    ax.set_title("Circle-Level Churn Heatmap (Subscribers | Avg ARPU | 30d Churn %)")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=circle_stats["churn_rate_30d_pct"], y=circle_stats.index,
                hue=circle_stats.index, palette="Reds_r", legend=False, ax=ax)
    ax.set_xlabel("30-Day Churn Rate (%)")
    ax.set_ylabel("Circle")
    ax.set_title("30-Day Churn Rate by Circle")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    return circle_stats


def recharge_gap_analysis(df: pd.DataFrame, pdf: PdfPages):
    """Task 2: churn rate vs. days since last recharge to find the silent threshold."""
    bins = [0, 5, 10, 15, 20, 25, 30, 45, 60, 90, df["days_since_last_recharge"].max() + 1]
    labels = ["0-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31-45", "46-60", "61-90", "90+"]
    df["recharge_gap_band"] = pd.cut(df["days_since_last_recharge"], bins=bins,
                                      labels=labels, include_lowest=True)

    gap_stats = (
        df.groupby("recharge_gap_band", observed=True)
        .agg(subscriber_count=("subscriber_id", "count"), churn_rate_30d=("churn_flag_30d", "mean"))
    )
    gap_stats["churn_rate_30d_pct"] = gap_stats["churn_rate_30d"] * 100

    print("\n=== 2. Recharge Gap Analysis ===")
    print(gap_stats.round(2).to_string())

    # Silent threshold = the band with the steepest jump (largest churn-rate increase
    # vs. the prior band), i.e. where churn risk accelerates most sharply.
    churn_pct = gap_stats["churn_rate_30d_pct"]
    silent_threshold = churn_pct.diff().idxmax()
    print(f"Silent churn threshold identified around recharge gap band: {silent_threshold} days")

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(x=gap_stats.index, y=gap_stats["churn_rate_30d_pct"],
                hue=gap_stats.index, palette="Oranges", legend=False, ax=ax)
    ax.set_xlabel("Days Since Last Recharge")
    ax.set_ylabel("30-Day Churn Rate (%)")
    ax.set_title("Churn Rate vs. Recharge Gap (Silent Churn Threshold)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    return gap_stats, silent_threshold


def arpu_band_analysis(df: pd.DataFrame, pdf: PdfPages) -> pd.DataFrame:
    """Task 3: churn rate by ARPU bracket to test price sensitivity."""
    bins = [-float("inf"), 150, 200, 250, float("inf")]
    labels = ["<150", "150-200", "200-250", ">250"]
    df["arpu_band"] = pd.cut(df["arpu_last_month_inr"], bins=bins, labels=labels)

    arpu_stats = (
        df.groupby("arpu_band", observed=True)
        .agg(subscriber_count=("subscriber_id", "count"), churn_rate_30d=("churn_flag_30d", "mean"))
    )
    arpu_stats["churn_rate_30d_pct"] = arpu_stats["churn_rate_30d"] * 100

    print("\n=== 3. ARPU Band Price Sensitivity ===")
    print(arpu_stats.round(2).to_string())

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=arpu_stats.index, y=arpu_stats["churn_rate_30d_pct"],
                hue=arpu_stats.index, palette="Blues", legend=False, ax=ax)
    ax.set_xlabel("ARPU Bracket (INR)")
    ax.set_ylabel("30-Day Churn Rate (%)")
    ax.set_title("Churn Rate by ARPU Bracket (Price Sensitivity)")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    return arpu_stats


def network_kpi_correlation(df: pd.DataFrame, pdf: PdfPages) -> pd.Series:
    """Task 4: correlation between network KPIs and 30-day churn."""
    kpi_cols = ["avg_sinr_db", "drop_call_rate_pct", "site_congestion_score", "churn_flag_30d"]
    corr_matrix = df[kpi_cols].corr()
    churn_corr = corr_matrix["churn_flag_30d"].drop("churn_flag_30d").sort_values(key=abs, ascending=False)

    print("\n=== 4. Network KPI Correlation with 30-Day Churn ===")
    print(churn_corr.round(3).to_string())

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Network KPI vs. Churn Correlation Matrix")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    return churn_corr


def summary_page(pdf: PdfPages, circle_stats, silent_threshold, arpu_stats, churn_corr) -> None:
    """Final PDF page: auto-generated key-findings summary from the computed stats."""
    top_circle = circle_stats.index[0]
    top_circle_rate = circle_stats["churn_rate_30d_pct"].iloc[0]
    worst_arpu_band = arpu_stats["churn_rate_30d_pct"].idxmax()
    worst_arpu_rate = arpu_stats["churn_rate_30d_pct"].max()
    top_kpi = churn_corr.index[0]
    top_kpi_corr = churn_corr.iloc[0]

    lines = [
        f"1. Circle Concentration: '{top_circle}' has the highest 30-day churn rate "
        f"at {top_circle_rate:.1f}%, the top priority circle for retention action.",
        "",
        f"2. Recharge Silent Threshold: Churn risk accelerates sharply once days-since-last-"
        f"recharge reaches the '{silent_threshold}' day band; proactive outreach should "
        f"trigger before this window.",
        "",
        f"3. Price Sensitivity: The '{worst_arpu_band}' ARPU bracket shows the highest churn "
        f"rate at {worst_arpu_rate:.1f}%, highlighting where pricing/value perception is weakest.",
        "",
        f"4. Network Quality Impact: '{top_kpi}' has the strongest correlation with 30-day "
        f"churn (r = {top_kpi_corr:.2f}), signaling network experience as a material churn driver.",
    ]

    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.08, 0.92, "Workstream 1 - Key Findings Summary", fontsize=16, weight="bold")
    y = 0.82
    for line in lines:
        fig.text(0.08, y, line, fontsize=11, wrap=True, va="top")
        y -= 0.09
    plt.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    df = load_data()
    print(f"Loaded {len(df):,} rows and {df.shape[1]} columns from subscribers.csv")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(REPORT_PATH) as pdf:
        circle_stats = circle_analysis(df, pdf)
        _gap_stats, silent_threshold = recharge_gap_analysis(df, pdf)
        arpu_stats = arpu_band_analysis(df, pdf)
        churn_corr = network_kpi_correlation(df, pdf)
        summary_page(pdf, circle_stats, silent_threshold, arpu_stats, churn_corr)

    print(f"\nPDF report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
