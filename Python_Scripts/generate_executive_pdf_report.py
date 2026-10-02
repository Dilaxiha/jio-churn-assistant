"""
generate_executive_pdf_report.py
Project 5: Jio Subscriber Churn Intelligence & Revenue Protection

Purpose
-------
Builds a publication-grade, 3-page executive PDF business report at
    Reports/Jio_Executive_EDA_Summary.pdf
covering geographic churn hotspots, the recharge "silent churn" threshold,
ARPU price sensitivity, and network QoS correlation with 30-day churn, closing
with actionable next steps for Workstream 2 (predictive modeling).

Design system
-------------
  Deep Navy   #1B365D  - headers, titles, primary accents
  Slate Grey  #708090   - sub-headings, captions, chart takeaways
  Margins     0.75 in   - clean, print-ready page layout

Structure
---------
  Page 1: Title header, Executive Summary, Context on Subscriber Churn & Revenue Exposure.
  Page 2: Visual Section 1 - Geographic Churn Hotspots & Recharge Silent Threshold.
  Page 3: Visual Section 2 - ARPU Price Sensitivity & Network QoS Correlation,
          plus Strategic Next Steps for Workstream 2 (Predictive Modeling).
"""

from __future__ import annotations

import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

sns.set_theme(style="whitegrid")

# --------------------------------------------------------------------------- #
# Path management
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "Data" / "subscribers.csv"
REPORTS_DIR = BASE_DIR / "Reports"
FIGURES_DIR = REPORTS_DIR / "figures"
PDF_PATH = REPORTS_DIR / "Jio_Executive_EDA_Summary.pdf"

TARGET = "churn_flag_30d"
TOTAL_PAGES = 3

# Corporate design system
NAVY = colors.HexColor("#1B365D")
SLATE = colors.HexColor("#708090")
LIGHT_BG = colors.HexColor("#F2F4F7")
ACCENT_RED = colors.HexColor("#C0392B")
BORDER_GRAY = colors.HexColor("#D1D5DB")


# --------------------------------------------------------------------------- #
# Data loading & stats
# --------------------------------------------------------------------------- #
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find subscribers.csv at expected location: {DATA_PATH}\n"
            "Ensure the Data/subscribers.csv file exists relative to the project root."
        )
    return pd.read_csv(DATA_PATH, parse_dates=["join_date", "churn_date"])


def compute_circle_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = (
        df.groupby("circle")
        .agg(subscriber_count=("subscriber_id", "count"), churn_rate=(TARGET, "mean"))
        .sort_values("churn_rate", ascending=False)
    )
    stats["churn_rate_pct"] = stats["churn_rate"] * 100
    return stats


def compute_recharge_gap_stats(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    work = df.copy()
    bins = [0, 5, 10, 15, 20, 25, 30, 45, 60, 90, work["days_since_last_recharge"].max() + 1]
    labels = ["0-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31-45", "46-60", "61-90", "90+"]
    work["recharge_gap_band"] = pd.cut(
        work["days_since_last_recharge"], bins=bins, labels=labels, include_lowest=True
    )
    stats = work.groupby("recharge_gap_band", observed=True)[TARGET].agg(
        subscriber_count="count", churn_rate="mean"
    )
    stats["churn_rate_pct"] = stats["churn_rate"] * 100
    silent_threshold = stats["churn_rate_pct"].diff().idxmax()
    return stats, silent_threshold


def compute_arpu_band_stats(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    bins = [-float("inf"), 150, 200, 250, float("inf")]
    labels = ["<150", "150-200", "200-250", ">250"]
    work["arpu_band"] = pd.cut(work["arpu_last_month_inr"], bins=bins, labels=labels)
    stats = work.groupby("arpu_band", observed=True)[TARGET].agg(
        subscriber_count="count", churn_rate="mean"
    )
    stats["churn_rate_pct"] = stats["churn_rate"] * 100
    return stats


def compute_network_kpi_correlation(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    kpi_cols = ["avg_sinr_db", "drop_call_rate_pct", "site_congestion_score"]
    corr_df = df[kpi_cols].copy()
    corr_df[TARGET] = df[TARGET].astype(int)
    corr = corr_df.corr()
    churn_corr = corr[TARGET].drop(TARGET).sort_values(key=lambda s: s.abs(), ascending=False)
    return corr, churn_corr


# --------------------------------------------------------------------------- #
# Chart generation (standalone, high-resolution PNGs for embedding)
# --------------------------------------------------------------------------- #
def plot_circle_risk_chart(circle_stats: pd.DataFrame, top_n: int = 15) -> Path:
    top = circle_stats.head(top_n)
    fig, ax = plt.subplots(figsize=(9, max(5, 0.4 * len(top))))
    sns.barplot(x=top["churn_rate_pct"], y=top.index, hue=top.index, palette="Reds_r",
                legend=False, ax=ax)
    for i, (rate, count) in enumerate(zip(top["churn_rate_pct"], top["subscriber_count"])):
        ax.text(rate, i, f"  {rate:.1f}%  (n={count:,})", va="center", fontsize=8)
    ax.set_xlabel("30-Day Churn Rate (%)")
    ax.set_ylabel("Telecom Circle")
    ax.set_title("Geographic Churn Hotspots by Circle", fontsize=13, weight="bold")
    fig.tight_layout()
    path = FIGURES_DIR / "exec_circle_risk_chart.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_recharge_gap_chart(gap_stats: pd.DataFrame, silent_threshold: str) -> Path:
    bar_colors = ["#C0392B" if band == silent_threshold else "#1B365D" for band in gap_stats.index]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(gap_stats.index.astype(str), gap_stats["churn_rate_pct"], color=bar_colors)
    ax.set_xlabel("Days Since Last Recharge")
    ax.set_ylabel("30-Day Churn Rate (%)")
    ax.set_title("Recharge Silent-Churn Threshold", fontsize=13, weight="bold")
    ax.tick_params(axis="x", rotation=30)
    ax.text(0.99, 0.95, "Red = silent-churn threshold band", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color="#C0392B")
    fig.tight_layout()
    path = FIGURES_DIR / "exec_recharge_gap_chart.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_arpu_band_chart(arpu_stats: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x=arpu_stats.index, y=arpu_stats["churn_rate_pct"], hue=arpu_stats.index,
                palette="Blues", legend=False, ax=ax)
    for i, v in enumerate(arpu_stats["churn_rate_pct"]):
        ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("ARPU Bracket (INR)")
    ax.set_ylabel("30-Day Churn Rate (%)")
    ax.set_title("ARPU Price-Sensitivity Analysis", fontsize=13, weight="bold")
    fig.tight_layout()
    path = FIGURES_DIR / "exec_arpu_band_chart.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_network_kpi_chart(corr_matrix: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Network KPI Correlation with 30-Day Churn", fontsize=12, weight="bold")
    fig.tight_layout()
    path = FIGURES_DIR / "exec_network_kpi_chart.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Business takeaways (auto-generated from computed stats)
# --------------------------------------------------------------------------- #
def build_circle_takeaways(circle_stats: pd.DataFrame) -> list[str]:
    top_circle = circle_stats.index[0]
    top_rate = circle_stats["churn_rate_pct"].iloc[0]
    avg_rate = circle_stats["churn_rate_pct"].mean()
    above_avg = int((circle_stats["churn_rate_pct"] > avg_rate).sum())
    return [
        f"'{top_circle}' is the highest-risk circle at {top_rate:.1f}% 30-day churn, "
        f"vs. a network-wide average of {avg_rate:.1f}% across all circles.",
        f"{above_avg} of {len(circle_stats)} circles sit above the network average churn rate - "
        "prioritize retention budget in the top 3-5 circles shown above for the fastest ROI.",
    ]


def build_gap_takeaways(gap_stats: pd.DataFrame, silent_threshold: str) -> list[str]:
    threshold_rate = gap_stats.loc[silent_threshold, "churn_rate_pct"]
    baseline_rate = gap_stats["churn_rate_pct"].iloc[0]
    return [
        f"Churn risk accelerates sharply at the '{silent_threshold}'-day recharge gap band "
        f"({threshold_rate:.1f}% churn), the clearest 'silent churn' early-warning signal.",
        f"Subscribers recharging within the first 5 days of their cycle show only "
        f"{baseline_rate:.1f}% churn - proactive outreach should trigger before this threshold.",
    ]


def build_arpu_takeaways(arpu_stats: pd.DataFrame) -> list[str]:
    worst_band = arpu_stats["churn_rate_pct"].idxmax()
    worst_rate = arpu_stats["churn_rate_pct"].max()
    best_band = arpu_stats["churn_rate_pct"].idxmin()
    best_rate = arpu_stats["churn_rate_pct"].min()
    return [
        f"The '{worst_band}' INR ARPU bracket shows the highest churn rate at {worst_rate:.1f}%, "
        "signaling a value-for-money gap in that price tier.",
        f"The '{best_band}' INR bracket is the most stable at {best_rate:.1f}% churn - a useful "
        "benchmark for tiered retention offers calibrated by ARPU bracket.",
    ]


def build_kpi_takeaways(churn_corr: pd.Series) -> list[str]:
    top_kpi = churn_corr.index[0]
    top_val = churn_corr.iloc[0]
    direction = "increases" if top_val > 0 else "reduces"
    return [
        f"'{top_kpi}' has the strongest relationship with churn (r = {top_val:.2f}); as it rises, "
        f"churn risk {direction}, confirming network experience as a material churn driver.",
        "Network KPI correlations are directional signals, not standalone predictors - they should "
        "be combined with behavioral and pricing features in the Workstream 2 model.",
    ]


# --------------------------------------------------------------------------- #
# ReportLab styling & page chrome
# --------------------------------------------------------------------------- #
def scaled_image_flowable(path: Path, max_width: float, max_height: float | None = None) -> RLImage:
    """Build a reportlab Image flowable that preserves the source PNG's aspect ratio."""
    with PILImage.open(path) as im:
        px_w, px_h = im.size
    aspect = px_h / px_w
    width = max_width
    height = width * aspect
    if max_height is not None and height > max_height:
        height = max_height
        width = height / aspect
    return RLImage(str(path), width=width, height=height)


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ExecTitle", fontName="Helvetica-Bold", fontSize=23,
        textColor=NAVY, spaceAfter=6, leading=27))
    styles.add(ParagraphStyle(
        name="ExecSubtitle", fontName="Helvetica", fontSize=11,
        textColor=SLATE, spaceAfter=12))
    styles.add(ParagraphStyle(
        name="ExecSectionHeading", fontName="Helvetica-Bold", fontSize=15,
        textColor=NAVY, spaceBefore=8, spaceAfter=6))
    styles.add(ParagraphStyle(
        name="ExecSubHeading", fontName="Helvetica-Bold", fontSize=11.5,
        textColor=SLATE, spaceBefore=6, spaceAfter=3))
    styles.add(ParagraphStyle(
        name="ExecBody", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=9.8, leading=14.2, alignment=4, textColor=colors.HexColor("#1F2937")))
    styles.add(ParagraphStyle(
        name="ExecBullet", parent=styles["ExecBody"], fontSize=9.3, leading=12.6, spaceAfter=2))
    styles.add(ParagraphStyle(
        name="ExecCaption", fontName="Helvetica-Oblique", fontSize=8.5,
        textColor=SLATE, spaceBefore=2, spaceAfter=5))
    return styles


def _header_footer(canvas, doc) -> None:
    canvas.saveState()
    width, height = A4

    canvas.setFillColor(NAVY)
    canvas.rect(0, height - 0.62 * inch, width, 0.62 * inch, stroke=0, fill=1)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawString(0.75 * inch, height - 0.4 * inch, "JIO SUBSCRIBER CHURN INTELLIGENCE")
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(width - 0.75 * inch, height - 0.4 * inch, "Executive EDA Summary | Confidential")

    canvas.setFillColor(SLATE)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(
        0.75 * inch, 0.55 * inch,
        f"Generated {datetime.now().strftime('%Y-%m-%d')} | Project 5: Churn Intelligence & Revenue Protection",
    )
    canvas.drawRightString(width - 0.75 * inch, 0.55 * inch, f"Page {doc.page} of {TOTAL_PAGES}")
    canvas.setStrokeColor(NAVY)
    canvas.setLineWidth(0.6)
    canvas.line(0.75 * inch, 0.72 * inch, width - 0.75 * inch, 0.72 * inch)

    canvas.restoreState()


def kpi_table(rows: list[list[str]]) -> Table:
    table = Table(rows, colWidths=[3.1 * inch, 2.9 * inch], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER_GRAY),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    return table


def bullet_list(items: list[str], styles) -> ListFlowable:
    return ListFlowable(
        [ListItem(Paragraph(item, styles["ExecBullet"]), bulletColor=SLATE) for item in items],
        bulletType="bullet", start="\u25cf", leftIndent=14, bulletFontSize=8,
    )


# --------------------------------------------------------------------------- #
# Page content
# --------------------------------------------------------------------------- #
def page1_elements(df: pd.DataFrame, circle_stats: pd.DataFrame, silent_threshold: str, styles) -> list:
    n = len(df)
    churn_count = int(df[TARGET].sum())
    churn_rate_pct = churn_count / n * 100
    avg_arpu = float(df["arpu_last_month_inr"].mean())
    revenue_at_risk = churn_count * avg_arpu
    top_circle = circle_stats.index[0]
    top_circle_rate = circle_stats["churn_rate_pct"].iloc[0]

    elements: list = [
        Spacer(1, 0.15 * inch),
        Paragraph("Jio Subscriber Churn Intelligence", styles["ExecTitle"]),
        Paragraph("Executive EDA Summary &mdash; Workstream 1 Diagnostics", styles["ExecSubtitle"]),
        HRFlowable(width="100%", thickness=1.1, color=NAVY, spaceAfter=12),
        Paragraph("Executive Summary", styles["ExecSectionHeading"]),
        Paragraph(
            f"This report distills the Workstream 1 diagnostic analysis of {n:,} Jio subscribers "
            f"into an executive brief. The current 30-day churn rate is <b>{churn_rate_pct:.2f}%</b>, "
            f"concentrated most heavily in the '{top_circle}' circle ({top_circle_rate:.1f}% churn). "
            f"A clear 'silent churn' early-warning signal emerges once the recharge gap reaches the "
            f"'{silent_threshold}'-day band. Pricing sensitivity and network quality-of-service both "
            "show measurable relationships with churn, giving the business three concrete levers - "
            "geography, recharge timing, and price/quality perception - to act on immediately.",
            styles["ExecBody"],
        ),
        Spacer(1, 0.18 * inch),
        Paragraph("Context: Subscriber Churn &amp; Revenue Exposure", styles["ExecSectionHeading"]),
        Paragraph(
            "Even a small churn rate translates into material revenue exposure at Jio's subscriber "
            "scale. The table below frames the headline diagnostics in business terms to support "
            "prioritization discussions ahead of the Workstream 2 predictive-modeling build.",
            styles["ExecBody"],
        ),
        Spacer(1, 0.12 * inch),
    ]

    rows = [
        ["Metric", "Value"],
        ["Total Subscribers Analyzed", f"{n:,}"],
        ["Baseline 30-Day Churn Rate", f"{churn_rate_pct:.2f}%"],
        ["Churned Subscribers (30d)", f"{churn_count:,}"],
        ["Average ARPU (Last Month)", f"INR {avg_arpu:,.2f}"],
        ["Est. Monthly Revenue at Risk", f"INR {revenue_at_risk:,.0f}"],
        ["Top-Risk Circle", f"{top_circle} ({top_circle_rate:.1f}%)"],
        ["Silent-Churn Recharge Gap Threshold", f"{silent_threshold} days"],
    ]
    elements.append(kpi_table(rows))
    elements.append(Spacer(1, 0.05 * inch))
    elements.append(Paragraph(
        "Note: revenue at risk is an illustrative estimate (churned subscribers &times; average ARPU) "
        "and excludes indirect effects such as referral or bundle attrition.",
        styles["ExecCaption"],
    ))
    return elements


def page2_elements(circle_path: Path, gap_path: Path, circle_stats: pd.DataFrame,
                    gap_stats: pd.DataFrame, silent_threshold: str, styles) -> list:
    elements: list = [
        Paragraph("Visual Section 1: Geographic Churn Hotspots &amp; Recharge Silent Threshold",
                  styles["ExecSectionHeading"]),
        Paragraph("1.1 Circle-Level Churn Hotspots", styles["ExecSubHeading"]),
    ]
    if circle_path.exists():
        elements.append(scaled_image_flowable(circle_path, max_width=6.3 * inch, max_height=2.5 * inch))
        elements.append(Paragraph(
            "Figure 1: 30-day churn rate by telecom circle, ranked highest to lowest.",
            styles["ExecCaption"],
        ))
    elements.append(bullet_list(build_circle_takeaways(circle_stats), styles))
    elements.append(Spacer(1, 0.06 * inch))

    elements.append(Paragraph("1.2 Recharge Gap &amp; Silent Churn Threshold", styles["ExecSubHeading"]))
    if gap_path.exists():
        elements.append(scaled_image_flowable(gap_path, max_width=6.3 * inch, max_height=2.2 * inch))
        elements.append(Paragraph(
            "Figure 2: 30-day churn rate by days since last recharge; the silent-churn threshold "
            "band is highlighted in red.",
            styles["ExecCaption"],
        ))
    elements.append(bullet_list(build_gap_takeaways(gap_stats, silent_threshold), styles))
    return elements


def page3_elements(arpu_path: Path, kpi_path: Path, arpu_stats: pd.DataFrame,
                    churn_corr: pd.Series, styles) -> list:
    elements: list = [
        Paragraph("Visual Section 2: ARPU Price Sensitivity &amp; Network QoS Correlation",
                  styles["ExecSectionHeading"]),
        Paragraph("2.1 ARPU Price Sensitivity", styles["ExecSubHeading"]),
    ]
    if arpu_path.exists():
        elements.append(scaled_image_flowable(arpu_path, max_width=4.2 * inch, max_height=2.1 * inch))
        elements.append(Paragraph(
            "Figure 3: 30-day churn rate by ARPU bracket (INR).", styles["ExecCaption"],
        ))
    elements.append(bullet_list(build_arpu_takeaways(arpu_stats), styles))
    elements.append(Spacer(1, 0.06 * inch))

    elements.append(Paragraph("2.2 Network Quality-of-Service Correlation", styles["ExecSubHeading"]))
    if kpi_path.exists():
        elements.append(scaled_image_flowable(kpi_path, max_width=3.4 * inch, max_height=2.1 * inch))
        elements.append(Paragraph(
            "Figure 4: Correlation matrix of network KPIs with 30-day churn.", styles["ExecCaption"],
        ))
    elements.append(bullet_list(build_kpi_takeaways(churn_corr), styles))
    elements.append(Spacer(1, 0.08 * inch))

    elements.append(Paragraph(
        "Strategic Next Steps &mdash; Workstream 2 (Predictive Modeling)", styles["ExecSectionHeading"]
    ))
    next_steps = [
        "Engineer a recharge-gap feature set anchored on the silent-churn threshold identified here, "
        "including days-since-last-recharge and recharge cadence trend.",
        "Include ARPU tier and ARPU trend (3-month vs. 6-month average) as pricing-sensitivity features.",
        "Incorporate network KPIs (SINR, drop-call rate, congestion score) as service-quality features.",
        "Add circle/geography as a categorical feature or apply geographic stratified sampling, given "
        "the concentration of risk observed here.",
        "Apply class-imbalance-aware modeling techniques (class weighting, threshold tuning, or "
        "resampling) given the low baseline churn rate, and validate using precision-at-top-decile "
        "and recall rather than raw accuracy.",
    ]
    elements.append(ListFlowable(
        [ListItem(Paragraph(s, styles["ExecBullet"]), bulletColor=NAVY) for s in next_steps],
        bulletType="bullet", start="\u25cf", leftIndent=14, bulletFontSize=8,
    ))
    return elements


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    try:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        print("Loading data...")
        df = load_data()
        print(f"Loaded {len(df):,} rows from {DATA_PATH.name}")

        print("Computing diagnostics...")
        circle_stats = compute_circle_stats(df)
        gap_stats, silent_threshold = compute_recharge_gap_stats(df)
        arpu_stats = compute_arpu_band_stats(df)
        kpi_corr_matrix, churn_corr = compute_network_kpi_correlation(df)

        print("Generating charts...")
        circle_path = plot_circle_risk_chart(circle_stats)
        gap_path = plot_recharge_gap_chart(gap_stats, silent_threshold)
        arpu_path = plot_arpu_band_chart(arpu_stats)
        kpi_path = plot_network_kpi_chart(kpi_corr_matrix)

        print("Building PDF...")
        styles = build_styles()
        doc = SimpleDocTemplate(
            str(PDF_PATH), pagesize=A4,
            leftMargin=0.75 * inch, rightMargin=0.75 * inch,
            topMargin=0.95 * inch, bottomMargin=0.85 * inch,
            title="Jio Subscriber Churn Intelligence - Executive EDA Summary",
            author="Jio Retention Analytics",
        )

        story: list = []
        story.extend(page1_elements(df, circle_stats, silent_threshold, styles))
        story.append(PageBreak())
        story.extend(page2_elements(circle_path, gap_path, circle_stats, gap_stats, silent_threshold, styles))
        story.append(PageBreak())
        story.extend(page3_elements(arpu_path, kpi_path, arpu_stats, churn_corr, styles))

        doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)

        print(f"\nExecutive PDF report saved to: {PDF_PATH}")

    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception:
        print("\n[ERROR] An unexpected error occurred while generating the executive PDF report:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
