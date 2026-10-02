"""
comprehensive_eda.py
Project 5: Jio Subscriber Churn Intelligence & Revenue Protection
Workstream 1 - Comprehensive Variable-by-Variable Exploratory Data Analysis

Purpose
-------
Performs a rigorous, production-grade EDA on every column of subscribers.csv:
  1. Full statistical summaries for numeric, categorical, and boolean variables.
  2. Target variable (churn_flag_30d) distribution, imbalance, and segment risk analysis.
  3. High-resolution charts: distributions, correlation heatmap, boxplots vs. churn,
     categorical churn-rate bars, and dedicated churn risk-factor charts.
  4. Structured console output plus machine-readable (JSON) and human-readable (TXT)
     artifacts consumed by generate_business_report_pdf.py.

Outputs (relative to project root)
-----------------------------------
  Reports/comprehensive_eda_report.txt
  Reports/eda_summary_stats.json
  Reports/figures/00_target_distribution.png
  Reports/figures/01_numeric_distributions.png
  Reports/figures/02_correlation_heatmap.png
  Reports/figures/03_boxplots_vs_churn.png
  Reports/figures/04_categorical_churn_rates.png
  Reports/figures/05_risk_factor_churn_rates.png
"""

from __future__ import annotations

import json
import math
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; charts are saved to disk, not displayed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")

# --------------------------------------------------------------------------- #
# Path management (robust regardless of the current working directory)
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "Data" / "subscribers.csv"
REPORTS_DIR = BASE_DIR / "Reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TXT_REPORT_PATH = REPORTS_DIR / "comprehensive_eda_report.txt"
JSON_STATS_PATH = REPORTS_DIR / "eda_summary_stats.json"

TARGET = "churn_flag_30d"
ID_COL = "subscriber_id"
DATE_COLS = ["join_date", "churn_date"]
PERCENTILES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

KEY_NUMERIC_FOR_BOXPLOTS = [
    "arpu_last_month_inr", "tenure_months", "days_since_last_recharge",
    "avg_recharge_gap_days", "data_gb_last_month", "complaints_6m",
    "payment_failures_6m", "outgoing_to_competitor_pct",
]
CATEGORICAL_FOR_CHARTS = ["circle", "zone", "plan_type", "device_brand", "home_product"]
SEGMENT_COLS_FOR_TARGET = [
    "circle", "zone", "plan_type", "device_brand", "autopay_enabled",
    "family_plan_flag", "is_5g_active", "roaming_user_flag",
]


# --------------------------------------------------------------------------- #
# Data loading & column classification
# --------------------------------------------------------------------------- #
def load_data() -> pd.DataFrame:
    """Load subscribers.csv with robust path resolution and date parsing."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find subscribers.csv at expected location: {DATA_PATH}\n"
            "Ensure the Data/subscribers.csv file exists relative to the project root."
        )
    df = pd.read_csv(DATA_PATH)
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def classify_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """Split columns into identifier / datetime / boolean / numeric / categorical buckets."""
    identifier = [c for c in [ID_COL] if c in df.columns]
    datetime_cols = [c for c in DATE_COLS if c in df.columns]
    boolean_cols = [c for c in df.columns if pd.api.types.is_bool_dtype(df[c])]
    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c not in boolean_cols and c not in identifier
    ]
    categorical_cols = [
        c for c in df.columns
        if c not in identifier + datetime_cols + boolean_cols + numeric_cols
    ]
    return {
        "identifier": identifier,
        "datetime": datetime_cols,
        "boolean": boolean_cols,
        "numeric": numeric_cols,
        "categorical": categorical_cols,
    }


# --------------------------------------------------------------------------- #
# Statistical summaries
# --------------------------------------------------------------------------- #
def numeric_summary(df: pd.DataFrame, numeric_cols: list[str]) -> dict[str, dict[str, Any]]:
    n = len(df)
    stats: dict[str, dict[str, Any]] = {}
    for col in numeric_cols:
        series = df[col]
        missing = int(series.isna().sum())
        desc = series.describe(percentiles=PERCENTILES)
        stats[col] = {
            "count": int(desc["count"]),
            "missing": missing,
            "missing_pct": round(missing / n * 100, 3),
            "mean": round(float(series.mean()), 4),
            "median": round(float(series.median()), 4),
            "std": round(float(series.std()), 4),
            "min": round(float(series.min()), 4),
            "max": round(float(series.max()), 4),
            "skew": round(float(series.skew()), 4),
            "kurtosis": round(float(series.kurtosis()), 4),
            "percentiles": {
                f"p{int(p * 100)}": round(float(desc[f"{int(p * 100)}%"]), 4) for p in PERCENTILES
            },
        }
    return stats


def categorical_summary(df: pd.DataFrame, categorical_cols: list[str]) -> dict[str, dict[str, Any]]:
    n = len(df)
    stats: dict[str, dict[str, Any]] = {}
    for col in categorical_cols:
        series = df[col]
        missing = int(series.isna().sum())
        vc = series.value_counts(dropna=True)
        top_items = vc.head(10)
        stats[col] = {
            "missing": missing,
            "missing_pct": round(missing / n * 100, 3),
            "unique_values": int(series.nunique(dropna=True)),
            "top_value": str(top_items.index[0]) if len(top_items) else None,
            "top_value_pct": round(float(top_items.iloc[0] / n * 100), 3) if len(top_items) else None,
            "top_10_value_counts": {str(k): int(v) for k, v in top_items.items()},
        }
    return stats


def boolean_summary(df: pd.DataFrame, boolean_cols: list[str]) -> dict[str, dict[str, Any]]:
    n = len(df)
    stats: dict[str, dict[str, Any]] = {}
    for col in boolean_cols:
        series = df[col]
        missing = int(series.isna().sum())
        valid = n - missing
        true_count = int(series.sum(skipna=True))
        stats[col] = {
            "missing": missing,
            "missing_pct": round(missing / n * 100, 3),
            "true_count": true_count,
            "true_pct": round(true_count / valid * 100, 3) if valid else 0.0,
            "false_count": int(valid - true_count),
            "false_pct": round((valid - true_count) / valid * 100, 3) if valid else 0.0,
        }
    return stats


def target_analysis(df: pd.DataFrame, col_buckets: dict[str, list[str]]) -> dict[str, Any]:
    """Class balance, segment-level churn rates, and numeric correlations with churn."""
    n = len(df)
    target_series = df[TARGET].astype(bool)
    churn_count = int(target_series.sum())
    non_churn_count = n - churn_count
    churn_rate_pct = round(churn_count / n * 100, 3)
    imbalance_ratio = round(non_churn_count / churn_count, 2) if churn_count else None

    segment_cols = [c for c in SEGMENT_COLS_FOR_TARGET if c in df.columns]
    segment_rates: dict[str, dict[str, Any]] = {}
    for seg_col in segment_cols:
        grouped = (
            df.groupby(seg_col, observed=True)[TARGET]
            .agg(subscriber_count="count", churn_rate="mean")
            .sort_values("churn_rate", ascending=False)
        )
        grouped["churn_rate_pct"] = (grouped["churn_rate"] * 100).round(2)
        segment_rates[seg_col] = {
            str(idx): {
                "subscriber_count": int(row["subscriber_count"]),
                "churn_rate_pct": float(row["churn_rate_pct"]),
            }
            for idx, row in grouped.iterrows()
        }

    numeric_cols = col_buckets["numeric"]
    corr_df = df[numeric_cols].copy()
    corr_df[TARGET] = target_series.astype(int)
    corr_with_target = (
        corr_df.corr()[TARGET].drop(TARGET).sort_values(key=lambda s: s.abs(), ascending=False)
    )
    top_correlations = {k: round(float(v), 4) for k, v in corr_with_target.head(10).items()}

    return {
        "total_subscribers": n,
        "churn_count": churn_count,
        "non_churn_count": non_churn_count,
        "churn_rate_pct": churn_rate_pct,
        "imbalance_ratio_non_churn_to_churn": imbalance_ratio,
        "segment_churn_rates": segment_rates,
        "top_numeric_correlations_with_churn": top_correlations,
    }


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def plot_target_distribution(df: pd.DataFrame, figures_dir: Path) -> None:
    counts = df[TARGET].value_counts().sort_index()
    pct = df[TARGET].value_counts(normalize=True).sort_index() * 100
    labels = ["Retained" if not v else "Churned" for v in counts.index]
    colors_list = ["#1B365D", "#C0392B"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].bar(labels, counts.values, color=colors_list)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=10)
    axes[0].set_title("Subscriber Counts: Churned vs. Retained")
    axes[0].set_ylabel("Number of Subscribers")

    axes[1].pie(
        pct.values, labels=[f"{lab}\n{p:.1f}%" for lab, p in zip(labels, pct.values)],
        colors=colors_list, startangle=90, wedgeprops={"edgecolor": "white"},
    )
    axes[1].set_title("30-Day Churn Rate Share")

    fig.suptitle("Target Variable Distribution: churn_flag_30d", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(figures_dir / "00_target_distribution.png", dpi=200)
    plt.close(fig)


def plot_numeric_distributions(df: pd.DataFrame, numeric_cols: list[str], figures_dir: Path) -> None:
    n_cols_total = len(numeric_cols)
    if n_cols_total == 0:
        return
    ncols = 4
    nrows = math.ceil(n_cols_total / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.2))
    axes = np.array(axes).reshape(-1)
    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color="#1B365D")
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("")
    for j in range(n_cols_total, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Numeric Feature Distributions", fontsize=15, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(figures_dir / "01_numeric_distributions.png", dpi=200)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: list[str], figures_dir: Path) -> None:
    corr_df = df[numeric_cols].copy()
    corr_df[TARGET] = df[TARGET].astype(int)
    corr = corr_df.corr()

    fig, ax = plt.subplots(figsize=(max(10, 0.5 * len(corr)), max(8, 0.5 * len(corr))))
    sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax,
                cbar_kws={"label": "Pearson correlation"})
    ax.set_title("Correlation Heatmap: Numeric Features & Churn (churn_flag_30d)",
                 fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(figures_dir / "02_correlation_heatmap.png", dpi=200)
    plt.close(fig)


def plot_boxplots_vs_churn(df: pd.DataFrame, figures_dir: Path) -> None:
    cols = [c for c in KEY_NUMERIC_FOR_BOXPLOTS if c in df.columns]
    if not cols:
        return
    ncols = 4
    nrows = math.ceil(len(cols) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axes = np.array(axes).reshape(-1)
    churn_label = df[TARGET].map({False: "Retained", True: "Churned"})
    for i, col in enumerate(cols):
        ax = axes[i]
        sns.boxplot(x=churn_label, y=df[col], hue=churn_label,
                    palette={"Retained": "#1B365D", "Churned": "#C0392B"},
                    legend=False, ax=ax)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
    for j in range(len(cols), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Key Numeric Features vs. 30-Day Churn", fontsize=15, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(figures_dir / "03_boxplots_vs_churn.png", dpi=200)
    plt.close(fig)


def plot_categorical_churn_rates(df: pd.DataFrame, figures_dir: Path, top_n: int = 10) -> None:
    cols = [c for c in CATEGORICAL_FOR_CHARTS if c in df.columns]
    if not cols:
        return
    ncols = 2
    nrows = math.ceil(len(cols) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4.5))
    axes = np.array(axes).reshape(-1)
    for i, col in enumerate(cols):
        ax = axes[i]
        grouped = (
            df.groupby(col, observed=True)[TARGET]
            .agg(count="count", churn_rate="mean")
            .sort_values("churn_rate", ascending=False)
        )
        grouped["churn_rate_pct"] = grouped["churn_rate"] * 100
        top = grouped.head(top_n)
        sns.barplot(x=top["churn_rate_pct"], y=top.index, hue=top.index,
                    palette="Reds_r", legend=False, ax=ax)
        ax.set_title(f"Churn Rate by {col}", fontsize=11)
        ax.set_xlabel("30-Day Churn Rate (%)")
        ax.set_ylabel("")
    for j in range(len(cols), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Categorical Segment Churn Rates (Top Categories)", fontsize=15, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(figures_dir / "04_categorical_churn_rates.png", dpi=200)
    plt.close(fig)


def plot_risk_factor_charts(df: pd.DataFrame, figures_dir: Path) -> tuple[pd.Series, pd.Series]:
    """Dedicated churn-rate charts for recharge gap bands and ARPU tiers (used in the PDF brief)."""
    work = df.copy()

    gap_bins = [0, 5, 10, 15, 20, 25, 30, 45, 60, 90, work["days_since_last_recharge"].max() + 1]
    gap_labels = ["0-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31-45", "46-60", "61-90", "90+"]
    work["recharge_gap_band"] = pd.cut(work["days_since_last_recharge"], bins=gap_bins,
                                        labels=gap_labels, include_lowest=True)

    arpu_bins = [-float("inf"), 150, 200, 250, float("inf")]
    arpu_labels = ["<150", "150-200", "200-250", ">250"]
    work["arpu_tier"] = pd.cut(work["arpu_last_month_inr"], bins=arpu_bins, labels=arpu_labels)

    gap_stats = work.groupby("recharge_gap_band", observed=True)[TARGET].mean() * 100
    arpu_stats = work.groupby("arpu_tier", observed=True)[TARGET].mean() * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.barplot(x=gap_stats.index, y=gap_stats.values, hue=gap_stats.index,
                palette="Oranges", legend=False, ax=axes[0])
    axes[0].set_title("Churn Rate by Recharge Gap (Days Since Last Recharge)")
    axes[0].set_xlabel("Days Since Last Recharge")
    axes[0].set_ylabel("30-Day Churn Rate (%)")
    axes[0].tick_params(axis="x", rotation=30)

    sns.barplot(x=arpu_stats.index, y=arpu_stats.values, hue=arpu_stats.index,
                palette="Blues", legend=False, ax=axes[1])
    axes[1].set_title("Churn Rate by ARPU Tier (INR)")
    axes[1].set_xlabel("ARPU Bracket")
    axes[1].set_ylabel("30-Day Churn Rate (%)")

    fig.suptitle("Key Churn Risk Factors", fontsize=15, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(figures_dir / "05_risk_factor_churn_rates.png", dpi=200)
    plt.close(fig)

    return gap_stats, arpu_stats


# --------------------------------------------------------------------------- #
# Key findings & structured outputs
# --------------------------------------------------------------------------- #
def build_key_findings(
    target_stats: dict[str, Any], gap_stats: pd.Series, arpu_stats: pd.Series
) -> list[str]:
    findings = [
        f"Dataset covers {target_stats['total_subscribers']:,} subscribers with a baseline "
        f"30-day churn rate of {target_stats['churn_rate_pct']:.2f}% "
        f"({target_stats['churn_count']:,} churned vs. {target_stats['non_churn_count']:,} retained), "
        f"an imbalance of roughly 1 churner per "
        f"{target_stats['imbalance_ratio_non_churn_to_churn']:.0f} retained subscribers."
    ]

    circle_rates = target_stats["segment_churn_rates"].get("circle")
    if circle_rates:
        top_circle = max(circle_rates.items(), key=lambda kv: kv[1]["churn_rate_pct"])
        findings.append(
            f"'{top_circle[0]}' circle shows the highest churn concentration at "
            f"{top_circle[1]['churn_rate_pct']:.1f}%, making it the top-priority region for "
            "retention campaigns."
        )

    if len(gap_stats) > 1:
        silent_band = gap_stats.diff().idxmax()
        findings.append(
            f"Churn risk accelerates sharply once the recharge gap reaches the '{silent_band}'-day "
            "band, marking the 'silent churn' threshold for proactive outreach."
        )

    if len(arpu_stats):
        worst_band = arpu_stats.idxmax()
        findings.append(
            f"The '{worst_band}' INR ARPU bracket has the highest churn rate at "
            f"{arpu_stats.max():.1f}%, pointing to pricing/value-perception risk in that tier."
        )

    top_corr = target_stats["top_numeric_correlations_with_churn"]
    if top_corr:
        top_feat, top_val = next(iter(top_corr.items()))
        findings.append(
            f"'{top_feat}' is the numeric feature most correlated with churn (r = {top_val:.2f}), "
            "flagging it as a strong candidate signal for the predictive model."
        )

    return findings


def save_text_report(
    df: pd.DataFrame,
    numeric_stats: dict[str, Any],
    categorical_stats: dict[str, Any],
    boolean_stats: dict[str, Any],
    target_stats: dict[str, Any],
    key_findings: list[str],
) -> None:
    n, c = df.shape
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("JIO SUBSCRIBER CHURN INTELLIGENCE - COMPREHENSIVE EDA REPORT")
    lines.append("=" * 70)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Rows: {n:,} | Columns: {c}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("1. TARGET VARIABLE (churn_flag_30d)")
    lines.append("-" * 70)
    lines.append(f"Churned: {target_stats['churn_count']:,} ({target_stats['churn_rate_pct']:.2f}%)")
    lines.append(
        f"Retained: {target_stats['non_churn_count']:,} "
        f"({100 - target_stats['churn_rate_pct']:.2f}%)"
    )
    lines.append(
        "Imbalance ratio (retained:churned): "
        f"{target_stats['imbalance_ratio_non_churn_to_churn']:.1f} : 1"
    )
    lines.append("")

    lines.append("-" * 70)
    lines.append("2. NUMERIC VARIABLE SUMMARY")
    lines.append("-" * 70)
    for col, s in numeric_stats.items():
        lines.append(
            f"{col}: mean={s['mean']}, median={s['median']}, std={s['std']}, "
            f"min={s['min']}, max={s['max']}, missing={s['missing']} ({s['missing_pct']}%)"
        )
    lines.append("")

    lines.append("-" * 70)
    lines.append("3. CATEGORICAL VARIABLE SUMMARY")
    lines.append("-" * 70)
    for col, s in categorical_stats.items():
        lines.append(
            f"{col}: unique={s['unique_values']}, top='{s['top_value']}' "
            f"({s['top_value_pct']}%), missing={s['missing']} ({s['missing_pct']}%)"
        )
    lines.append("")

    lines.append("-" * 70)
    lines.append("4. BOOLEAN VARIABLE SUMMARY")
    lines.append("-" * 70)
    for col, s in boolean_stats.items():
        lines.append(f"{col}: True={s['true_pct']}%, False={s['false_pct']}%, missing={s['missing']}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("5. SEGMENT CHURN RATES (top 8 per segment)")
    lines.append("-" * 70)
    for seg_col, rates in target_stats["segment_churn_rates"].items():
        lines.append(f"[{seg_col}]")
        ranked = sorted(rates.items(), key=lambda kv: -kv[1]["churn_rate_pct"])[:8]
        for cat, info in ranked:
            lines.append(f"   - {cat}: {info['churn_rate_pct']:.2f}% (n={info['subscriber_count']:,})")
        lines.append("")

    lines.append("-" * 70)
    lines.append("6. TOP NUMERIC CORRELATIONS WITH CHURN")
    lines.append("-" * 70)
    for feat, val in target_stats["top_numeric_correlations_with_churn"].items():
        lines.append(f"   - {feat}: r = {val:.3f}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("7. KEY FINDINGS")
    lines.append("-" * 70)
    for i, finding in enumerate(key_findings, start=1):
        lines.append(f"{i}. {finding}")
    lines.append("")

    TXT_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    try:
        print("Loading data...")
        df = load_data()
        print(f"Loaded {len(df):,} rows and {df.shape[1]} columns from {DATA_PATH.name}")

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        col_buckets = classify_columns(df)
        print(
            f"Identified {len(col_buckets['numeric'])} numeric, "
            f"{len(col_buckets['categorical'])} categorical, "
            f"{len(col_buckets['boolean'])} boolean, "
            f"{len(col_buckets['datetime'])} datetime columns."
        )

        print("\nComputing statistical summaries...")
        numeric_stats = numeric_summary(df, col_buckets["numeric"])
        categorical_stats = categorical_summary(df, col_buckets["categorical"])
        boolean_stats = boolean_summary(df, col_buckets["boolean"])
        target_stats = target_analysis(df, col_buckets)

        print("Generating charts...")
        plot_target_distribution(df, FIGURES_DIR)
        plot_numeric_distributions(df, col_buckets["numeric"], FIGURES_DIR)
        plot_correlation_heatmap(df, col_buckets["numeric"], FIGURES_DIR)
        plot_boxplots_vs_churn(df, FIGURES_DIR)
        plot_categorical_churn_rates(df, FIGURES_DIR)
        gap_stats, arpu_stats = plot_risk_factor_charts(df, FIGURES_DIR)

        key_findings = build_key_findings(target_stats, gap_stats, arpu_stats)

        print("\n" + "=" * 70)
        print("TARGET VARIABLE SUMMARY")
        print("=" * 70)
        print(
            f"Churn rate (30d): {target_stats['churn_rate_pct']:.2f}% "
            f"({target_stats['churn_count']:,} of {target_stats['total_subscribers']:,})"
        )

        print("\n" + "=" * 70)
        print("KEY FINDINGS")
        print("=" * 70)
        for i, finding in enumerate(key_findings, start=1):
            print(f"{i}. {finding}")

        print("\nSaving structured outputs...")
        save_text_report(df, numeric_stats, categorical_stats, boolean_stats, target_stats, key_findings)

        master_stats = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "dataset": {
                "n_rows": int(len(df)),
                "n_cols": int(df.shape[1]),
                "n_duplicates": int(df.duplicated().sum()),
                "join_date_min": str(df["join_date"].min().date()) if "join_date" in df.columns else None,
                "join_date_max": str(df["join_date"].max().date()) if "join_date" in df.columns else None,
                "avg_arpu_last_month_inr": (
                    round(float(df["arpu_last_month_inr"].mean()), 2)
                    if "arpu_last_month_inr" in df.columns else None
                ),
            },
            "column_buckets": col_buckets,
            "numeric_stats": numeric_stats,
            "categorical_stats": categorical_stats,
            "boolean_stats": boolean_stats,
            "target_analysis": target_stats,
            "key_findings": key_findings,
        }
        with open(JSON_STATS_PATH, "w", encoding="utf-8") as f:
            json.dump(master_stats, f, indent=2, default=str)

        print(f"\nText report saved to: {TXT_REPORT_PATH}")
        print(f"JSON stats saved to: {JSON_STATS_PATH}")
        print(f"Charts saved to: {FIGURES_DIR}")
        print("\nEDA complete.")

    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception:
        print("\n[ERROR] An unexpected error occurred during EDA execution:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
