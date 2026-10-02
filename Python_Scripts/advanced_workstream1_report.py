"""
Advanced Workstream 1 - Executive Business Brief
Project 5: Jio Subscriber Churn Intelligence & Revenue Protection

Primary source : Data/subscribers.csv (64,738 subscribers x 43 columns)
Supplementary  : Data/Jio_Retention_Dataset.xlsx -> 'circle_monthly_kpi' sheet
                 (port-in/port-out volumes for MNP share-transfer context)

Produces a 2-page, publication-grade executive PDF at:
    Reports/advanced_workstream1_business_report.pdf

Sections:
  1. Circle-level risk tiering + revenue (ARR) exposure
  2. Recharge-silence hazard curve (inflection point detection)
  3. ARPU elasticity x device type x tenure cross-tabulation
  4. Network KPI / service-friction correlation with 30-day churn
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; charts are embedded in the PDF only

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Table,
    TableStyle,
)

sns.set_theme(style="whitegrid", font_scale=0.85)

# --------------------------------------------------------------------------- #
# Paths & brand palette
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "Data" / "subscribers.csv"
XLSX_PATH = BASE_DIR / "Data" / "Jio_Retention_Dataset.xlsx"
REPORT_PATH = BASE_DIR / "Reports" / "advanced_workstream1_business_report.pdf"

NAVY = colors.HexColor("#1B365D")
SLATE = colors.HexColor("#5C6B73")
LIGHT_GREY = colors.HexColor("#F2F4F5")
WHITE = colors.white
RISK_RED = colors.HexColor("#B3261E")
RISK_AMBER = colors.HexColor("#B7791F")
RISK_GREEN = colors.HexColor("#2E7D32")

MPL_NAVY = "#1B365D"
MPL_SLATE = "#5C6B73"
RISK_COLORS = {"High-Risk": "#B3261E", "Moderate-Risk": "#B7791F", "Low-Risk": "#2E7D32"}

PAGE_W, PAGE_H = A4


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_subscribers() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH, parse_dates=["join_date", "churn_date"])


def load_circle_kpi() -> pd.DataFrame:
    return pd.read_excel(XLSX_PATH, sheet_name="circle_monthly_kpi")


# --------------------------------------------------------------------------- #
# 1. Circle-level risk tiering & ARR exposure
# --------------------------------------------------------------------------- #
def circle_risk_tiering(df: pd.DataFrame) -> pd.DataFrame:
    stats = (
        df.groupby("circle")
        .agg(
            subscriber_count=("subscriber_id", "count"),
            avg_arpu=("arpu_last_month_inr", "mean"),
            churn_rate=("churn_flag_30d", "mean"),
        )
        .sort_values("churn_rate", ascending=False)
    )
    stats["churn_rate_pct"] = stats["churn_rate"] * 100

    def tier(rate: float) -> str:
        if rate > 0.02:
            return "High-Risk"
        if rate >= 0.015:
            return "Moderate-Risk"
        return "Low-Risk"

    stats["risk_tier"] = stats["churn_rate"].apply(tier)

    # Annualised revenue (ARR) at risk, in INR crores (1 crore = 1e7 INR)
    at_risk_subs = stats["subscriber_count"] * stats["churn_rate"]
    stats["revenue_exposure_cr"] = (at_risk_subs * stats["avg_arpu"] * 12) / 1e7

    return stats


# --------------------------------------------------------------------------- #
# 2. Recharge-silence hazard analysis
# --------------------------------------------------------------------------- #
def recharge_hazard_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    bins = [0, 5, 10, 15, 20, 25, 30, 45, 60, 90, df["days_since_last_recharge"].max() + 1]
    labels = ["0-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31-45", "46-60", "61-90", "90+"]
    band = pd.cut(df["days_since_last_recharge"], bins=bins, labels=labels, include_lowest=True)

    hazard = (
        df.groupby(band, observed=True)
        .agg(subscriber_count=("subscriber_id", "count"), churn_rate=("churn_flag_30d", "mean"))
    )
    hazard["hazard_rate_pct"] = hazard["churn_rate"] * 100

    # Inflection point = band with the steepest churn-rate acceleration vs. the prior band
    inflection_band = hazard["hazard_rate_pct"].diff().idxmax()
    return hazard, str(inflection_band)


# --------------------------------------------------------------------------- #
# 3. ARPU elasticity x device type x tenure
# --------------------------------------------------------------------------- #
def arpu_elasticity(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    arpu_bins = [-float("inf"), 150, 200, 250, float("inf")]
    arpu_labels = ["<150", "150-200", "200-250", ">250"]
    df["arpu_band"] = pd.cut(df["arpu_last_month_inr"], bins=arpu_bins, labels=arpu_labels)

    df["device_type"] = np.where(df["is_5g_active"], "5G Active", "4G / Inactive")

    tenure_bins = [-1, 12, 24, 48, df["tenure_months"].max() + 1]
    tenure_labels = ["0-12m", "13-24m", "25-48m", "48m+"]
    df["tenure_band"] = pd.cut(df["tenure_months"], bins=tenure_bins, labels=tenure_labels)

    device_pivot = pd.pivot_table(
        df, index="arpu_band", columns="device_type", values="churn_flag_30d", aggfunc="mean", observed=True
    ) * 100
    tenure_pivot = pd.pivot_table(
        df, index="arpu_band", columns="tenure_band", values="churn_flag_30d", aggfunc="mean", observed=True
    ) * 100

    return device_pivot, tenure_pivot


# --------------------------------------------------------------------------- #
# 4. Network KPI & service-friction correlation
# --------------------------------------------------------------------------- #
def network_friction_correlation(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    kpi_cols = [
        "drop_call_rate_pct",
        "site_congestion_score",
        "avg_sinr_db",
        "complaints_6m",
        "churn_flag_30d",
    ]
    corr_matrix = df[kpi_cols].corr()

    # Operational friction: how much do unresolved complaints / slow resolution amplify churn?
    no_unresolved = df.loc[df["unresolved_complaints"] == 0, "churn_flag_30d"].mean() * 100
    has_unresolved = df.loc[df["unresolved_complaints"] > 0, "churn_flag_30d"].mean() * 100

    complainants = df[df["complaints_6m"] > 0]
    fast_resolution = complainants.loc[complainants["avg_resolution_days"] <= 3, "churn_flag_30d"].mean() * 100
    slow_resolution = complainants.loc[complainants["avg_resolution_days"] > 3, "churn_flag_30d"].mean() * 100

    friction = {
        "no_unresolved_churn_pct": no_unresolved,
        "has_unresolved_churn_pct": has_unresolved,
        "fast_resolution_churn_pct": fast_resolution,
        "slow_resolution_churn_pct": slow_resolution,
    }
    return corr_matrix, friction


# --------------------------------------------------------------------------- #
# Strategic context: ARR exposure & MNP share-transfer dynamics
# --------------------------------------------------------------------------- #
def share_transfer_context(circle_kpi: pd.DataFrame) -> dict:
    total_port_in = circle_kpi["port_in_requests"].sum()
    total_port_out = circle_kpi["port_out_requests"].sum()
    net_port = total_port_in - total_port_out

    latest_month = circle_kpi["month_end"].max()
    latest_base = circle_kpi.loc[circle_kpi["month_end"] == latest_month, "closing_base"].sum()

    return {
        "total_port_in": int(total_port_in),
        "total_port_out": int(total_port_out),
        "net_port": int(net_port),
        "latest_base": int(latest_base),
    }


# --------------------------------------------------------------------------- #
# Chart builders -> return PNG bytes buffers sized for the PDF layout
# --------------------------------------------------------------------------- #
def _fig_to_buffer(fig) -> BytesIO:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def chart_circle_risk_heatmap(circle_stats: pd.DataFrame) -> BytesIO:
    ordered = circle_stats.sort_values("churn_rate_pct", ascending=False)
    heat_data = ordered[["subscriber_count", "avg_arpu", "churn_rate_pct", "revenue_exposure_cr"]]
    heat_norm = (heat_data - heat_data.min()) / (heat_data.max() - heat_data.min())

    fig, ax = plt.subplots(figsize=(6.6, 7.4))
    sns.heatmap(
        heat_norm, annot=heat_data.round(1), fmt="", cmap="Reds", ax=ax, cbar=False,
        linewidths=0.5, linecolor="white",
    )
    ax.set_title("Circle Risk Heatmap\n(Subscribers | Avg ARPU | Churn % | ARR Exposure Cr)",
                  fontsize=10, color=MPL_NAVY, weight="bold")
    ax.set_xticklabels(["Subs", "Avg ARPU", "Churn %", "ARR Exp. Cr"], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("")

    for label, circle in zip(ax.get_yticklabels(), ordered.index):
        tier = ordered.loc[circle, "risk_tier"]
        label.set_color(RISK_COLORS[tier])
        label.set_fontsize(7.5)

    fig.tight_layout()
    return _fig_to_buffer(fig)


def chart_recharge_hazard_curve(hazard: pd.DataFrame, inflection_band: str) -> BytesIO:
    fig, ax = plt.subplots(figsize=(6.6, 7.4))
    x = np.arange(len(hazard))
    y = hazard["hazard_rate_pct"].values

    ax.plot(x, y, marker="o", color=MPL_NAVY, linewidth=2, markersize=5)
    ax.fill_between(x, y, color=MPL_NAVY, alpha=0.08)

    inflection_idx = list(hazard.index.astype(str)).index(inflection_band)
    ax.axvline(inflection_idx, color=RISK_COLORS["High-Risk"], linestyle="--", linewidth=1.5)
    ax.annotate(
        f"Inflection point\n({inflection_band} days)",
        xy=(inflection_idx, y[inflection_idx]),
        xytext=(max(inflection_idx - 3.2, 0), y[inflection_idx] + (y.max() * 0.18)),
        fontsize=8, color=RISK_COLORS["High-Risk"], weight="bold",
        arrowprops=dict(arrowstyle="->", color=RISK_COLORS["High-Risk"]),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(hazard.index.astype(str), rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Days Since Last Recharge", fontsize=9)
    ax.set_ylabel("30-Day Churn (Hazard) Rate %", fontsize=9)
    ax.set_title("Recharge-Silence Hazard Curve", fontsize=10, color=MPL_NAVY, weight="bold")
    fig.tight_layout()
    return _fig_to_buffer(fig)


def chart_arpu_elasticity(device_pivot: pd.DataFrame, tenure_pivot: pd.DataFrame) -> BytesIO:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.2))

    sns.heatmap(device_pivot, annot=True, fmt=".1f", cmap="Blues", ax=axes[0], cbar=False, linewidths=0.5,
                linecolor="white")
    axes[0].set_title("Churn % by ARPU Band x Device Type", fontsize=10, color=MPL_NAVY, weight="bold")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("ARPU Band (INR)")

    sns.heatmap(tenure_pivot, annot=True, fmt=".1f", cmap="Blues", ax=axes[1], cbar=False, linewidths=0.5,
                linecolor="white")
    axes[1].set_title("Churn % by ARPU Band x Tenure", fontsize=10, color=MPL_NAVY, weight="bold")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")

    fig.tight_layout()
    return _fig_to_buffer(fig)


def chart_network_correlation(corr_matrix: pd.DataFrame) -> BytesIO:
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax,
                linewidths=0.5, linecolor="white")
    ax.set_title("Network KPI & Service Friction\nCorrelation with 30-Day Churn",
                  fontsize=10, color=MPL_NAVY, weight="bold")
    fig.tight_layout()
    return _fig_to_buffer(fig)


# --------------------------------------------------------------------------- #
# ReportLab document assembly
# --------------------------------------------------------------------------- #
def _styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("TitleNavy", parent=base["Title"], textColor=NAVY, fontSize=17,
                                 leading=20, spaceAfter=2),
        "subtitle": ParagraphStyle("Subtitle", parent=base["Normal"], textColor=SLATE, fontSize=10,
                                    spaceAfter=10),
        "h2": ParagraphStyle("H2", parent=base["Heading2"], textColor=NAVY, fontSize=12.5,
                              spaceBefore=8, spaceAfter=5),
        "body": ParagraphStyle("Body", parent=base["BodyText"], fontSize=9.3, leading=13,
                                textColor=colors.HexColor("#262626"), spaceAfter=4),
        "caption": ParagraphStyle("Caption", parent=base["BodyText"], fontSize=7.6, leading=10,
                                   textColor=SLATE, alignment=1),
    }


def _decorate_page(canvas, doc) -> None:
    """Draws the navy header band and footer on every page."""
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, PAGE_H - 1.3 * cm, PAGE_W, 1.3 * cm, fill=1, stroke=0)
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawString(1.8 * cm, PAGE_H - 0.85 * cm, "JIO SUBSCRIBER CHURN INTELLIGENCE & REVENUE PROTECTION")
    canvas.setFont("Helvetica", 7.5)
    canvas.drawRightString(PAGE_W - 1.8 * cm, PAGE_H - 0.85 * cm, "Project 5 | Workstream 1")

    canvas.setStrokeColor(SLATE)
    canvas.setLineWidth(0.5)
    canvas.line(1.8 * cm, 1.3 * cm, PAGE_W - 1.8 * cm, 1.3 * cm)
    canvas.setFillColor(SLATE)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(1.8 * cm, 0.9 * cm, "Confidential - Internal Use Only")
    canvas.drawRightString(PAGE_W - 1.8 * cm, 0.9 * cm, f"Page {doc.page} of 2")
    canvas.restoreState()


def _risk_table(circle_stats: pd.DataFrame, styles: dict) -> Table:
    ordered = circle_stats.sort_values(["risk_tier", "churn_rate_pct"], ascending=[True, False])
    tier_order = {"High-Risk": 0, "Moderate-Risk": 1, "Low-Risk": 2}
    ordered = ordered.iloc[ordered["risk_tier"].map(tier_order).argsort(kind="stable")]

    header = ["Circle", "Risk Tier", "Subscribers", "Avg ARPU (INR)", "Churn %", "ARR Exposure (Cr)"]
    rows = [header]
    tier_text_colors = {"High-Risk": RISK_RED, "Moderate-Risk": RISK_AMBER, "Low-Risk": RISK_GREEN}
    row_tiers = []
    for circle, r in ordered.iterrows():
        rows.append([
            circle,
            r["risk_tier"],
            f"{int(r['subscriber_count']):,}",
            f"{r['avg_arpu']:.0f}",
            f"{r['churn_rate_pct']:.2f}%",
            f"{r['revenue_exposure_cr']:.2f}",
        ])
        row_tiers.append(r["risk_tier"])

    table = Table(rows, colWidths=[3.6 * cm, 2.6 * cm, 2.4 * cm, 2.6 * cm, 1.9 * cm, 2.9 * cm], repeatRows=1)

    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.6),
        ("ALIGN", (2, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#C7CDD1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]
    for i, tier in enumerate(row_tiers, start=1):
        band_color = WHITE if i % 2 == 1 else LIGHT_GREY
        style_cmds.append(("BACKGROUND", (0, i), (-1, i), band_color))
        style_cmds.append(("TEXTCOLOR", (1, i), (1, i), tier_text_colors[tier]))
        style_cmds.append(("FONTNAME", (1, i), (1, i), "Helvetica-Bold"))

    table.setStyle(TableStyle(style_cmds))
    return table


def build_pdf(
    circle_stats: pd.DataFrame,
    hazard: pd.DataFrame,
    inflection_band: str,
    device_pivot: pd.DataFrame,
    tenure_pivot: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    friction: dict,
    context: dict,
    total_subscribers: int,
) -> None:
    styles = _styles()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(REPORT_PATH), pagesize=A4,
        leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=2.1 * cm, bottomMargin=1.8 * cm,
        title="Jio Subscriber Churn Intelligence - Workstream 1 Executive Brief",
    )

    total_arr_exposure = circle_stats["revenue_exposure_cr"].sum()
    high_risk_circles = (circle_stats["risk_tier"] == "High-Risk").sum()
    overall_churn = circle_stats["subscriber_count"].mul(circle_stats["churn_rate_pct"]).sum() / \
        circle_stats["subscriber_count"].sum()
    top_circle = circle_stats.sort_values("churn_rate_pct", ascending=False).index[0]

    elements = []

    elements.append(Paragraph("Workstream 1 - Executive Diagnostics Brief", styles["title"]))
    elements.append(Paragraph(
        f"Project 5: Jio Subscriber Churn Intelligence &amp; Revenue Protection &nbsp;|&nbsp; "
        f"Base: {total_subscribers:,} subscribers", styles["subtitle"]))

    elements.append(Paragraph("Executive Summary", styles["h2"]))
    elements.append(Paragraph(
        f"Portfolio-wide 30-day churn stands at <b>{overall_churn:.2f}%</b> across "
        f"{total_subscribers:,} subscribers, translating to an estimated "
        f"<b>&#8377;{total_arr_exposure:,.1f} crore</b> of annualised revenue (ARR) at risk. "
        f"<b>{high_risk_circles} circle(s)</b> are classified High-Risk (&gt;2% churn), led by "
        f"<b>{top_circle}</b>. Churn accelerates sharply once subscribers cross the "
        f"<b>{inflection_band}-day</b> recharge-silence window, and low-ARPU / low-tenure "
        f"segments show materially higher price sensitivity. Network friction signals "
        f"(congestion, drop-calls, unresolved complaints) further compound retention risk.",
        styles["body"]))

    elements.append(Paragraph("Strategic Context: ARR Exposure &amp; Share Transfer Dynamics", styles["h2"]))
    net_port_direction = "net outflow" if context["net_port"] < 0 else "net inflow"
    elements.append(Paragraph(
        f"Against a current active base of approximately <b>{context['latest_base']:,}</b> subscribers, "
        f"cumulative MNP activity recorded <b>{context['total_port_in']:,}</b> port-in and "
        f"<b>{context['total_port_out']:,}</b> port-out requests over the observed period - a "
        f"<b>{net_port_direction} of {abs(context['net_port']):,} subscribers</b>, indicating "
        f"{'competitive share pressure' if context['net_port'] < 0 else 'net competitive share gain'} "
        f"that compounds the &#8377;{total_arr_exposure:,.1f} crore organic churn exposure quantified above. "
        f"Protecting the {high_risk_circles} High-Risk circle(s) is therefore the highest-leverage lever "
        f"for both organic retention and inorganic (MNP) share defense.",
        styles["body"]))

    elements.append(Paragraph("Section 1: Circle Risk &amp; Recharge Hazard Diagnostics", styles["h2"]))
    img1 = Image(chart_circle_risk_heatmap(circle_stats), width=8.3 * cm, height=9.3 * cm)
    img2 = Image(chart_recharge_hazard_curve(hazard, inflection_band), width=8.3 * cm, height=9.3 * cm)
    chart_row1 = Table([[img1, img2]], colWidths=[8.6 * cm, 8.6 * cm])
    chart_row1.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    elements.append(chart_row1)

    elements.append(PageBreak())

    elements.append(Paragraph("Section 2: ARPU Elasticity &amp; Network QoS Correlations", styles["h2"]))
    img3 = Image(chart_arpu_elasticity(device_pivot, tenure_pivot), width=17.0 * cm, height=8.0 * cm)
    elements.append(img3)
    img4 = Image(chart_network_correlation(corr_matrix), width=8.3 * cm, height=7.8 * cm)
    friction_text = Paragraph(
        f"<b>Operational Friction Insights</b><br/><br/>"
        f"Churn among subscribers with <b>zero unresolved complaints</b>: "
        f"<b>{friction['no_unresolved_churn_pct']:.2f}%</b><br/>"
        f"Churn among subscribers with <b>1+ unresolved complaints</b>: "
        f"<b>{friction['has_unresolved_churn_pct']:.2f}%</b><br/><br/>"
        f"Among complainants, churn with resolution &le;3 days: "
        f"<b>{friction['fast_resolution_churn_pct']:.2f}%</b> vs. &gt;3 days: "
        f"<b>{friction['slow_resolution_churn_pct']:.2f}%</b><br/><br/>"
        f"Unresolved complaints and slow ticket resolution both materially amplify churn "
        f"probability, confirming service friction as an actionable, controllable retention lever.",
        styles["body"])
    chart_row2 = Table([[img4, friction_text]], colWidths=[8.6 * cm, 8.6 * cm])
    chart_row2.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    elements.append(chart_row2)

    elements.append(Paragraph("Circle Risk Tier Summary", styles["h2"]))
    elements.append(_risk_table(circle_stats, styles))

    elements.append(Paragraph("Strategic Recommendations - Workstream 2 (Predictive Modeling &amp; Uplift)",
                               styles["h2"]))
    elements.append(Paragraph(
        "1. Build a subscriber-level churn propensity model (gradient boosting / logistic baseline) using "
        "recharge-gap, ARPU-band, tenure, device type, and network-friction features identified here as the "
        "core signal set.<br/>"
        "2. Prioritise a proactive save-desk trigger at the recharge-silence inflection point, targeted first "
        "at High-Risk circles to defend the largest share of ARR exposure.<br/>"
        "3. Layer an uplift-modeling (treatment-effect) approach on top of the propensity model to target "
        "retention offers only at subscribers who are both high-risk and responsive to intervention, avoiding "
        "margin erosion on already-loyal, low-ARPU segments.<br/>"
        "4. Route unresolved-complaint and SLA-breach cases into the same retention workflow, since service "
        "friction shows a clear, addressable amplification effect on churn probability.",
        styles["body"]))

    doc.build(elements, onFirstPage=_decorate_page, onLaterPages=_decorate_page)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    df = load_subscribers()
    print(f"Loaded {len(df):,} rows and {df.shape[1]} columns from subscribers.csv")

    circle_kpi = load_circle_kpi()
    context = share_transfer_context(circle_kpi)

    circle_stats = circle_risk_tiering(df)
    hazard, inflection_band = recharge_hazard_analysis(df)
    device_pivot, tenure_pivot = arpu_elasticity(df)
    corr_matrix, friction = network_friction_correlation(df)

    print("\n=== Circle Risk Tiers ===")
    print(circle_stats.round(2).to_string())
    print(f"\nRecharge hazard inflection point: {inflection_band} days")
    print("\n=== ARPU x Device Type Churn % ===")
    print(device_pivot.round(2).to_string())
    print("\n=== ARPU x Tenure Churn % ===")
    print(tenure_pivot.round(2).to_string())
    print("\n=== Network KPI Correlation ===")
    print(corr_matrix.round(3).to_string())
    print("\n=== Operational Friction ===")
    print(friction)

    build_pdf(
        circle_stats=circle_stats,
        hazard=hazard,
        inflection_band=inflection_band,
        device_pivot=device_pivot,
        tenure_pivot=tenure_pivot,
        corr_matrix=corr_matrix,
        friction=friction,
        context=context,
        total_subscribers=len(df),
    )
    print(f"\nExecutive PDF report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
