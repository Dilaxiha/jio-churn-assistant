"""
generate_business_report_pdf.py
Project 5: Jio Subscriber Churn Intelligence & Revenue Protection
Workstream 1 - Client-Ready Business Brief

Purpose
-------
Compiles the EDA insights, visual outputs, and target-variable analysis produced by
comprehensive_eda.py into a polished 4-page, client-ready PDF business brief:
    Reports/jio_eda_executive_summary.pdf

Design
------
Uses ReportLab with a clean corporate design system (Navy primary #1B365D, clear
typography, proper margins). Falls back to a matplotlib PdfPages-based builder if
ReportLab is not installed, so a PDF can always be produced.

Robustness
----------
If comprehensive_eda.py has not been run yet (missing eda_summary_stats.json or chart
PNGs), this script recomputes the minimum statistics/charts it needs directly from
subscribers.csv so it can still run standalone.

Structure
---------
  Page 1: Executive Summary, Dataset Overview, Key Takeaways.
  Page 2: Target Variable Analysis & Churn Drivers (churn distribution, recharge gap
          and ARPU tier risk factors).
  Page 3: Variable-by-Variable Deep Dive (numeric business meaning + segment risk chart).
  Page 4: Strategic Recommendations & Next Steps for Workstream 2 (predictive modelling).
"""

from __future__ import annotations

import json
import sys
import textwrap
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

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm, inch
    from reportlab.platypus import (
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

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

sns.set_theme(style="whitegrid")

# --------------------------------------------------------------------------- #
# Path management
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "Data" / "subscribers.csv"
REPORTS_DIR = BASE_DIR / "Reports"
FIGURES_DIR = REPORTS_DIR / "figures"
JSON_STATS_PATH = REPORTS_DIR / "eda_summary_stats.json"
PDF_PATH = REPORTS_DIR / "jio_eda_executive_summary.pdf"

TARGET = "churn_flag_30d"

REQUIRED_CHARTS = {
    "target": FIGURES_DIR / "00_target_distribution.png",
    "risk_factors": FIGURES_DIR / "05_risk_factor_churn_rates.png",
    "categorical": FIGURES_DIR / "04_categorical_churn_rates.png",
}

NUMERIC_BUSINESS_NOTES = {
    "arpu_last_month_inr": "Average revenue per user last month - the core measure of subscriber value.",
    "tenure_months": "How long a subscriber has stayed with Jio - early-tenure customers are typically more churn-prone.",
    "days_since_last_recharge": "Days since the last recharge - a direct proxy for engagement and 'silent churn' risk.",
    "avg_recharge_gap_days": "Typical time between recharges - larger gaps signal weakening commitment.",
    "data_gb_last_month": "Data consumption last month - a proxy for how 'sticky' the connection is for daily use.",
    "complaints_6m": "Number of complaints in the last 6 months - a direct dissatisfaction signal.",
    "payment_failures_6m": "Failed payment attempts - often precede voluntary or involuntary churn.",
    "outgoing_to_competitor_pct": "Share of usage directed to competitor networks - an early switching signal.",
}

# Corporate design system
if REPORTLAB_AVAILABLE:
    NAVY = colors.HexColor("#1B365D")
    ACCENT = colors.HexColor("#C0392B")
    LIGHT_GRAY = colors.HexColor("#F2F4F7")
    SLATE_GREY = colors.HexColor("#708090")
    BORDER_GRAY = colors.HexColor("#D1D5DB")


# --------------------------------------------------------------------------- #
# Statistics loading (with standalone fallback)
# --------------------------------------------------------------------------- #
def compute_fallback_stats() -> dict[str, Any]:
    """Recompute the minimum statistics needed for the report directly from the CSV."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"subscribers.csv not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    n = len(df)
    churn_count = int(df[TARGET].sum())
    churn_rate_pct = round(churn_count / n * 100, 3)

    segment_rates: dict[str, Any] = {}
    for seg_col in ("circle", "plan_type", "device_brand"):
        if seg_col in df.columns:
            grouped = df.groupby(seg_col, observed=True)[TARGET].agg(
                subscriber_count="count", churn_rate="mean"
            )
            grouped["churn_rate_pct"] = grouped["churn_rate"] * 100
            segment_rates[seg_col] = {
                str(idx): {
                    "subscriber_count": int(row["subscriber_count"]),
                    "churn_rate_pct": float(row["churn_rate_pct"]),
                }
                for idx, row in grouped.iterrows()
            }

    avg_arpu = round(float(df["arpu_last_month_inr"].mean()), 2) if "arpu_last_month_inr" in df.columns else None

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": {
            "n_rows": n,
            "n_cols": int(df.shape[1]),
            "n_duplicates": int(df.duplicated().sum()),
            "avg_arpu_last_month_inr": avg_arpu,
        },
        "target_analysis": {
            "total_subscribers": n,
            "churn_count": churn_count,
            "non_churn_count": n - churn_count,
            "churn_rate_pct": churn_rate_pct,
            "imbalance_ratio_non_churn_to_churn": round((n - churn_count) / churn_count, 2) if churn_count else None,
            "segment_churn_rates": segment_rates,
            "top_numeric_correlations_with_churn": {},
        },
        "numeric_stats": {},
        "categorical_stats": {},
        "key_findings": [
            f"Baseline 30-day churn rate is {churn_rate_pct:.2f}% across {n:,} subscribers.",
        ],
        "_fallback": True,
    }


def load_stats() -> dict[str, Any]:
    if JSON_STATS_PATH.exists():
        try:
            with open(JSON_STATS_PATH, "r", encoding="utf-8") as f:
                stats = json.load(f)
            print(f"Loaded EDA statistics from {JSON_STATS_PATH.name}")
            return stats
        except (json.JSONDecodeError, OSError) as e:
            print(f"[WARN] Could not read {JSON_STATS_PATH.name} ({e}); recomputing from CSV.", file=sys.stderr)
    else:
        print(
            f"[WARN] {JSON_STATS_PATH.name} not found - run comprehensive_eda.py first for the full "
            "analysis. Recomputing minimal statistics from CSV in the meantime.",
            file=sys.stderr,
        )
    return compute_fallback_stats()


def ensure_charts() -> dict[str, Path]:
    """Verify required chart PNGs exist; regenerate minimal fallback versions if missing."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    missing = [k for k, p in REQUIRED_CHARTS.items() if not p.exists()]
    if not missing:
        return REQUIRED_CHARTS

    print(f"[WARN] Missing chart(s) {missing} - generating fallback charts from CSV.", file=sys.stderr)
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"subscribers.csv not found at {DATA_PATH}; cannot regenerate charts.")
    df = pd.read_csv(DATA_PATH)

    if "target" in missing and TARGET in df.columns:
        counts = df[TARGET].value_counts().sort_index()
        labels = ["Retained" if not v else "Churned" for v in counts.index]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.bar(labels, counts.values, color=["#1B365D", "#C0392B"])
        for i, v in enumerate(counts.values):
            ax.text(i, v, f"{v:,}", ha="center", va="bottom")
        ax.set_title("Churned vs. Retained Subscribers")
        fig.tight_layout()
        fig.savefig(REQUIRED_CHARTS["target"], dpi=200)
        plt.close(fig)

    if "risk_factors" in missing and {"days_since_last_recharge", "arpu_last_month_inr", TARGET}.issubset(df.columns):
        gap_bins = [0, 5, 10, 15, 20, 25, 30, 45, 60, 90, df["days_since_last_recharge"].max() + 1]
        gap_labels = ["0-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31-45", "46-60", "61-90", "90+"]
        df["recharge_gap_band"] = pd.cut(
            df["days_since_last_recharge"], bins=gap_bins, labels=gap_labels, include_lowest=True
        )
        arpu_bins = [-float("inf"), 150, 200, 250, float("inf")]
        arpu_labels = ["<150", "150-200", "200-250", ">250"]
        df["arpu_tier"] = pd.cut(df["arpu_last_month_inr"], bins=arpu_bins, labels=arpu_labels)

        gap_stats = df.groupby("recharge_gap_band", observed=True)[TARGET].mean() * 100
        arpu_stats = df.groupby("arpu_tier", observed=True)[TARGET].mean() * 100

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        sns.barplot(x=gap_stats.index, y=gap_stats.values, hue=gap_stats.index,
                    palette="Oranges", legend=False, ax=axes[0])
        axes[0].set_title("Churn Rate by Recharge Gap")
        axes[0].tick_params(axis="x", rotation=30)
        sns.barplot(x=arpu_stats.index, y=arpu_stats.values, hue=arpu_stats.index,
                    palette="Blues", legend=False, ax=axes[1])
        axes[1].set_title("Churn Rate by ARPU Tier")
        fig.tight_layout()
        fig.savefig(REQUIRED_CHARTS["risk_factors"], dpi=200)
        plt.close(fig)

    if "categorical" in missing and TARGET in df.columns:
        cols = [c for c in ("circle", "plan_type", "device_brand") if c in df.columns]
        if cols:
            fig, axes = plt.subplots(1, len(cols), figsize=(6 * len(cols), 4.5))
            axes = [axes] if len(cols) == 1 else axes
            for ax, col in zip(axes, cols):
                grouped = (
                    df.groupby(col, observed=True)[TARGET].mean().sort_values(ascending=False).head(10) * 100
                )
                sns.barplot(x=grouped.values, y=grouped.index, hue=grouped.index,
                            palette="Reds_r", legend=False, ax=ax)
                ax.set_title(f"Churn Rate by {col}")
                ax.set_xlabel("Churn Rate (%)")
            fig.tight_layout()
            fig.savefig(REQUIRED_CHARTS["categorical"], dpi=200)
            plt.close(fig)

    return REQUIRED_CHARTS


# --------------------------------------------------------------------------- #
# ReportLab PDF builder
# --------------------------------------------------------------------------- #
def scaled_image_flowable(path: Path, max_width: float, max_height: float | None = None) -> "RLImage":
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
        name="ReportTitle", fontName="Helvetica-Bold", fontSize=22,
        textColor=NAVY, spaceAfter=6, leading=26))
    styles.add(ParagraphStyle(
        name="ReportSubtitle", fontName="Helvetica", fontSize=11,
        textColor=SLATE_GREY, spaceAfter=14))
    styles.add(ParagraphStyle(
        name="SectionHeading", fontName="Helvetica-Bold", fontSize=14,
        textColor=NAVY, spaceBefore=10, spaceAfter=8))
    styles.add(ParagraphStyle(
        name="SubHeading", fontName="Helvetica-Bold", fontSize=11.5,
        textColor=NAVY, spaceBefore=8, spaceAfter=4))
    styles.add(ParagraphStyle(
        name="BodyTextJustify", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=9.7, leading=14, alignment=4, textColor=colors.HexColor("#1F2937")))
    styles.add(ParagraphStyle(
        name="BulletText", parent=styles["BodyTextJustify"], spaceAfter=4))
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
    canvas.drawRightString(width - 0.75 * inch, height - 0.4 * inch, "Executive Summary | Confidential")

    canvas.setFillColor(SLATE_GREY)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(
        0.75 * inch, 0.55 * inch,
        f"Generated {datetime.now().strftime('%Y-%m-%d')} | Project 5: Churn Intelligence & Revenue Protection",
    )
    canvas.drawRightString(width - 0.75 * inch, 0.55 * inch, f"Page {doc.page} of 4")
    canvas.setStrokeColor(NAVY)
    canvas.setLineWidth(0.6)
    canvas.line(0.75 * inch, 0.72 * inch, width - 0.75 * inch, 0.72 * inch)

    canvas.restoreState()


def page1_elements(stats: dict[str, Any], styles) -> list:
    ds = stats.get("dataset", {})
    tgt = stats.get("target_analysis", {})
    findings = stats.get("key_findings", [])

    n = ds.get("n_rows", tgt.get("total_subscribers", 0))
    churn_rate = tgt.get("churn_rate_pct", 0.0)
    avg_arpu = ds.get("avg_arpu_last_month_inr")

    elements: list = [
        Spacer(1, 0.6 * cm),
        Paragraph("Jio Subscriber Churn Intelligence", styles["ReportTitle"]),
        Paragraph(
            "Workstream 1 &mdash; Exploratory Data Analysis &amp; Business Diagnostics Executive Brief",
            styles["ReportSubtitle"],
        ),
        Spacer(1, 0.2 * cm),
        Paragraph("Executive Summary", styles["SectionHeading"]),
        Paragraph(
            f"This brief summarizes findings from a comprehensive exploratory analysis of "
            f"{n:,} Jio subscribers. The current 30-day churn rate stands at "
            f"<b>{churn_rate:.2f}%</b>, reflecting a highly imbalanced but business-critical "
            "retention challenge. The analysis identifies the strongest churn drivers across "
            "network experience, pricing, recharge behavior, and customer engagement, providing "
            "a factual foundation for the predictive modelling work planned in Workstream 2.",
            styles["BodyTextJustify"],
        ),
        Spacer(1, 0.35 * cm),
        Paragraph("Dataset Overview", styles["SectionHeading"]),
    ]

    table_data = [
        ["Metric", "Value"],
        ["Total Subscribers", f"{n:,}"],
        ["Total Data Columns", str(ds.get("n_cols", "N/A"))],
        ["Average ARPU (Last Month)", f"INR {avg_arpu:,.2f}" if avg_arpu is not None else "N/A"],
        ["Baseline 30-Day Churn Rate", f"{churn_rate:.2f}%"],
        ["Churned Subscribers", f"{tgt.get('churn_count', 0):,}"],
        ["Retained Subscribers", f"{tgt.get('non_churn_count', 0):,}"],
    ]
    table = Table(table_data, colWidths=[8 * cm, 7 * cm], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER_GRAY),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 0.4 * cm))

    elements.append(Paragraph("Key Takeaways", styles["SectionHeading"]))
    bullet_source = findings[:5] if findings else ["No key findings available."]
    elements.append(ListFlowable(
        [ListItem(Paragraph(f, styles["BulletText"]), bulletColor=NAVY) for f in bullet_source],
        bulletType="bullet", start="\u25cf", leftIndent=14, bulletFontSize=8,
    ))
    return elements


def page2_elements(stats: dict[str, Any], styles, charts: dict[str, Path]) -> list:
    tgt = stats.get("target_analysis", {})
    elements: list = [Paragraph("Target Variable Analysis &amp; Churn Drivers", styles["SectionHeading"])]

    elements.append(Paragraph(
        f"Only <b>{tgt.get('churn_rate_pct', 0):.2f}%</b> of subscribers churned in the last "
        "30 days, but the business impact is concentrated: certain recharge behaviors and "
        "pricing tiers show materially higher risk. The charts below highlight where "
        "retention effort should be focused first.",
        styles["BodyTextJustify"],
    ))
    elements.append(Spacer(1, 0.25 * cm))

    if charts["target"].exists():
        elements.append(scaled_image_flowable(charts["target"], max_width=16.5 * cm, max_height=7 * cm))
        elements.append(Spacer(1, 0.3 * cm))

    elements.append(Paragraph("Recharge Gaps &amp; ARPU Tiers as Leading Risk Signals", styles["SubHeading"]))
    if charts["risk_factors"].exists():
        elements.append(scaled_image_flowable(charts["risk_factors"], max_width=16.5 * cm, max_height=7.5 * cm))
        elements.append(Spacer(1, 0.2 * cm))

    elements.append(Paragraph(
        "Subscribers who go significantly longer without recharging show a sharply higher "
        "chance of churning in the following 30 days &mdash; this 'silent churn' window is the "
        "clearest early-warning signal available today. Similarly, churn is not spread "
        "evenly across ARPU (average revenue per user) tiers: certain price brackets show "
        "elevated risk, suggesting a value-for-money gap that targeted offers can address.",
        styles["BodyTextJustify"],
    ))
    return elements


def page3_elements(stats: dict[str, Any], styles, charts: dict[str, Path]) -> list:
    elements: list = [Paragraph("Variable-by-Variable Deep Dive", styles["SectionHeading"])]
    elements.append(Paragraph(
        "The table below translates the most predictive raw data fields into plain-English "
        "business meaning, alongside their typical (average) values across the subscriber base.",
        styles["BodyTextJustify"],
    ))
    elements.append(Spacer(1, 0.2 * cm))

    numeric_stats = stats.get("numeric_stats", {})
    rows = [["Variable", "Business Meaning", "Average", "Median"]]
    for var, note in NUMERIC_BUSINESS_NOTES.items():
        s = numeric_stats.get(var)
        avg_val = f"{s['mean']:.1f}" if s else "N/A"
        med_val = f"{s['median']:.1f}" if s else "N/A"
        rows.append([var, note, avg_val, med_val])

    table = Table(rows, colWidths=[3.5 * cm, 8.5 * cm, 2.2 * cm, 2.2 * cm], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
        ("GRID", (0, 0), (-1, -1), 0.4, BORDER_GRAY),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 0.35 * cm))

    elements.append(Paragraph("Which Customer Segments Carry the Most Risk?", styles["SubHeading"]))
    if charts["categorical"].exists():
        elements.append(scaled_image_flowable(charts["categorical"], max_width=16.5 * cm, max_height=8 * cm))
        elements.append(Spacer(1, 0.2 * cm))

    elements.append(Paragraph(
        "Churn is not uniform across geography, plan type, or device brand. The chart above "
        "ranks the segments with the highest 30-day churn rates, helping prioritize which "
        "customer groups need retention attention first.",
        styles["BodyTextJustify"],
    ))
    return elements


def page4_elements(stats: dict[str, Any], styles) -> list:
    tgt = stats.get("target_analysis", {})
    elements: list = [Paragraph("Strategic Recommendations &amp; Next Steps", styles["SectionHeading"])]

    elements.append(Paragraph("Immediate Retention Actions", styles["SubHeading"]))
    recs = [
        "Launch proactive, automated outreach (SMS/app nudge/callback offer) to subscribers "
        "approaching the identified 'silent churn' recharge-gap threshold, before they lapse.",
        "Review pricing and value bundles for the ARPU tier(s) showing the highest churn rate, "
        "e.g. via loyalty discounts, data top-ups, or plan right-sizing.",
        "Prioritize retention campaigns in the highest-churn circles/zones and device segments "
        "identified in this analysis, rather than a one-size-fits-all approach.",
        "Fast-track resolution of unresolved complaints &mdash; unresolved issues are a strong, "
        "actionable churn signal that can be fixed operationally.",
        "Monitor payment-failure patterns and offer flexible retry/payment options to reduce "
        "involuntary churn.",
    ]
    elements.append(ListFlowable(
        [ListItem(Paragraph(r, styles["BulletText"]), bulletColor=NAVY) for r in recs],
        bulletType="bullet", start="\u25cf", leftIndent=14, bulletFontSize=8,
    ))
    elements.append(Spacer(1, 0.3 * cm))

    elements.append(Paragraph("Next Steps: Workstream 2 &mdash; Predictive Modelling", styles["SubHeading"]))
    imbalance = tgt.get("imbalance_ratio_non_churn_to_churn")
    imbalance_note = (
        f" With an observed imbalance of roughly {imbalance:.0f}:1 (retained:churned), "
        "class-imbalance-aware techniques (e.g. class weighting, SMOTE, or threshold tuning) "
        "will be essential." if imbalance else ""
    )
    next_steps = [
        "Build a supervised churn-propensity model (e.g. gradient boosting or logistic "
        "regression baseline) using the engineered risk signals surfaced in this EDA "
        "(recharge gap, ARPU tier, complaints, network KPIs)." + imbalance_note,
        "Engineer time-windowed behavioral features (trend in ARPU, recharge cadence "
        "changes, recent complaint velocity) rather than relying on single snapshot values.",
        "Validate the model with business-relevant metrics (precision at top-risk decile, "
        "recall, lift) rather than raw accuracy, given the rarity of churn events.",
        "Establish a monthly scoring pipeline that feeds a prioritized, actionable "
        "retention worklist to the customer-care and marketing teams.",
        "Set up a feedback loop to track intervention outcomes (offer redemption, "
        "post-intervention churn) to continuously improve targeting precision.",
    ]
    elements.append(ListFlowable(
        [ListItem(Paragraph(s, styles["BulletText"]), bulletColor=NAVY) for s in next_steps],
        bulletType="bullet", start="\u25cf", leftIndent=14, bulletFontSize=8,
    ))
    return elements


def build_pdf_reportlab(stats: dict[str, Any], charts: dict[str, Path]) -> None:
    styles = build_styles()
    doc = SimpleDocTemplate(
        str(PDF_PATH), pagesize=A4,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        topMargin=0.95 * inch, bottomMargin=0.85 * inch,
        title="Jio Subscriber Churn Intelligence - Executive Summary",
        author="Jio Retention Analytics",
    )

    story: list = []
    story.extend(page1_elements(stats, styles))
    story.append(PageBreak())
    story.extend(page2_elements(stats, styles, charts))
    story.append(PageBreak())
    story.extend(page3_elements(stats, styles, charts))
    story.append(PageBreak())
    story.extend(page4_elements(stats, styles))

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)


# --------------------------------------------------------------------------- #
# Matplotlib PdfPages fallback (used only if ReportLab is unavailable)
# --------------------------------------------------------------------------- #
def build_pdf_matplotlib(stats: dict[str, Any], charts: dict[str, Path]) -> None:
    from matplotlib.backends.backend_pdf import PdfPages

    tgt = stats.get("target_analysis", {})
    ds = stats.get("dataset", {})
    findings = stats.get("key_findings", [])

    def text_page(title: str, lines: list[str]):
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 in inches
        fig.patch.set_facecolor("white")
        fig.text(0.07, 0.94, title, fontsize=18, weight="bold", color="#1B365D")
        y = 0.87
        for line in lines:
            for wrapped_line in textwrap.wrap(line, width=95) or [""]:
                fig.text(0.07, y, wrapped_line, fontsize=10, va="top")
                y -= 0.028
            y -= 0.012
        plt.axis("off")
        return fig

    with PdfPages(PDF_PATH) as pdf:
        n = ds.get("n_rows", tgt.get("total_subscribers", 0))
        lines1 = [
            f"Total Subscribers: {n:,}",
            f"Average ARPU (last month): INR {ds.get('avg_arpu_last_month_inr', 'N/A')}",
            f"Baseline 30-Day Churn Rate: {tgt.get('churn_rate_pct', 0):.2f}%",
            "",
            "Key Takeaways:",
        ] + [f"  - {f}" for f in findings[:5]]
        pdf.savefig(text_page("Executive Summary", lines1))
        plt.close()

        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.07, 0.95, "Target Variable Analysis & Churn Drivers",
                  fontsize=16, weight="bold", color="#1B365D")
        y = 0.9
        for key in ("target", "risk_factors"):
            path = charts.get(key)
            if path and path.exists():
                img = plt.imread(path)
                ax = fig.add_axes((0.08, y - 0.38, 0.86, 0.36))
                ax.imshow(img)
                ax.axis("off")
                y -= 0.42
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.07, 0.95, "Variable-by-Variable Deep Dive",
                  fontsize=16, weight="bold", color="#1B365D")
        path = charts.get("categorical")
        if path and path.exists():
            img = plt.imread(path)
            ax = fig.add_axes((0.08, 0.45, 0.86, 0.45))
            ax.imshow(img)
            ax.axis("off")
        pdf.savefig(fig)
        plt.close(fig)

        recs = [
            "Launch proactive outreach ahead of the silent-churn recharge-gap threshold.",
            "Review pricing/value bundles for high-churn ARPU tiers.",
            "Prioritize campaigns in the highest-churn circles/zones and device segments.",
            "Fast-track resolution of unresolved complaints.",
            "Build a class-imbalance-aware predictive model for Workstream 2.",
        ]
        lines4 = ["Strategic Recommendations & Next Steps:", ""] + [f"  - {r}" for r in recs]
        pdf.savefig(text_page("Recommendations", lines4))
        plt.close()

    print(f"[INFO] ReportLab not available - built fallback PDF via matplotlib at {PDF_PATH}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    try:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        print("Loading EDA statistics...")
        stats = load_stats()

        print("Verifying/generating chart assets...")
        charts = ensure_charts()

        print("Building PDF business report...")
        if REPORTLAB_AVAILABLE:
            build_pdf_reportlab(stats, charts)
        else:
            build_pdf_matplotlib(stats, charts)

        print(f"\nBusiness report saved to: {PDF_PATH}")

    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception:
        print("\n[ERROR] An unexpected error occurred while generating the PDF report:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
