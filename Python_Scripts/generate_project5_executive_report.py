"""
generate_project5_executive_report.py
Project 5: Jio Subscriber Churn Intelligence & Revenue Protection

Purpose
-------
Compiles the Workstream 1 (diagnostics) and Workstream 2 (predictive modeling)
results into a publication-grade, multi-page executive PDF report at:
    Reports/Project_5_Jio_Executive_Churn_Report.pdf

Design system
-------------
  Deep Navy   #1B365D  - headers, titles, primary accents
  Slate Grey  #708090   - sub-headings, captions
  Margins     0.75 in   - clean, print-ready page layout
  Footer      "Page X of Y" on every page (dynamic total via a numbered canvas)

Structure
---------
  Page 1: Executive Summary & Context (the "leaky bucket" business challenge,
          high-level project goals across all three workstreams).
  Page 2: Predictive Modeling Performance - a formatted comparison table
          (ROC-AUC, PR-AUC, Accuracy, Precision, Recall, Overfitting Gap) loaded
          from Reports/comprehensive_model_metrics.csv.
  Page 3: Model Diagnostics & Explainability - confusion matrix findings and the
          SHAP summary plot, with plain-English churn-driver takeaways.
  Page 4: Strategic Recommendations & Next Steps, including the Workstream 3
          conversational-analytics roadmap.

Robustness
----------
Required input (comprehensive_model_metrics.csv) is verified up front with a
clear error if missing. Optional visual assets (confusion matrix / SHAP PNGs,
EDA stats JSON) are verified individually; if any are missing, the report still
builds, substituting a clearly labeled placeholder instead of failing.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas as pdfcanvas
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

import pandas as pd

# --------------------------------------------------------------------------- #
# Path management
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "Reports"
PDF_PATH = REPORTS_DIR / "Project_5_Jio_Executive_Churn_Report.pdf"

METRICS_CSV_PATH = REPORTS_DIR / "comprehensive_model_metrics.csv"
SHAP_PNG_PATH = REPORTS_DIR / "best_model_shap_summary.png"
CONFUSION_MATRIX_PNG_PATH = REPORTS_DIR / "confusion_matrix_xgboost.png"
CONFUSION_MATRIX_JSON_PATH = REPORTS_DIR / "confusion_matrix_xgboost.json"
EDA_STATS_JSON_PATH = REPORTS_DIR / "eda_summary_stats.json"

OVERFIT_ROC_AUC_GAP_THRESHOLD = 0.05

# Corporate design system
NAVY = colors.HexColor("#1B365D")
SLATE = colors.HexColor("#708090")
LIGHT_BG = colors.HexColor("#F2F4F7")
ACCENT_RED = colors.HexColor("#C0392B")
BORDER_GRAY = colors.HexColor("#D1D5DB")


# --------------------------------------------------------------------------- #
# File verification & data loading (robust: hard-fail only on the required CSV)
# --------------------------------------------------------------------------- #
def verify_inputs() -> dict[str, bool]:
    """Check every input this report can use and print a verification summary."""
    checks = {
        "comprehensive_model_metrics.csv (required)": METRICS_CSV_PATH.exists(),
        "best_model_shap_summary.png (optional)": SHAP_PNG_PATH.exists(),
        "confusion_matrix_xgboost.png (optional)": CONFUSION_MATRIX_PNG_PATH.exists(),
        "confusion_matrix_xgboost.json (optional)": CONFUSION_MATRIX_JSON_PATH.exists(),
        "eda_summary_stats.json (optional)": EDA_STATS_JSON_PATH.exists(),
    }
    print("Input file verification:")
    for label, exists in checks.items():
        print(f"   [{'OK' if exists else 'MISSING'}] {label}")

    if not METRICS_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Required input not found: {METRICS_CSV_PATH}\n"
            "Run train_churn_models.py first to generate the model comparison metrics."
        )
    return checks


def load_metrics() -> pd.DataFrame:
    df = pd.read_csv(METRICS_CSV_PATH)
    required_cols = {"Model", "ROC_AUC", "PR_AUC", "Accuracy", "Precision", "Recall", "Overfitting_Gap"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"{METRICS_CSV_PATH.name} is missing expected column(s): {sorted(missing)}. "
            "Re-run train_churn_models.py to regenerate it."
        )
    return df


def load_confusion_counts() -> dict[str, int] | None:
    if not CONFUSION_MATRIX_JSON_PATH.exists():
        return None
    try:
        with open(CONFUSION_MATRIX_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def load_eda_context() -> dict[str, Any] | None:
    if not EDA_STATS_JSON_PATH.exists():
        return None
    try:
        with open(EDA_STATS_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


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


def safe_image(path: Path, max_width: float, max_height: float, styles, missing_note: str):
    """Return a scaled image flowable, or a clearly labeled placeholder if the file is missing."""
    if path.exists():
        return scaled_image_flowable(path, max_width=max_width, max_height=max_height)
    return Paragraph(f"[Chart not available: {missing_note}]", styles["ExecCaption"])


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ExecTitle", fontName="Helvetica-Bold", fontSize=21,
        textColor=NAVY, spaceAfter=6, leading=25))
    styles.add(ParagraphStyle(
        name="ExecSubtitle", fontName="Helvetica", fontSize=11,
        textColor=SLATE, spaceAfter=12))
    styles.add(ParagraphStyle(
        name="ExecSectionHeading", fontName="Helvetica-Bold", fontSize=15,
        textColor=NAVY, spaceBefore=10, spaceAfter=7))
    styles.add(ParagraphStyle(
        name="ExecSubHeading", fontName="Helvetica-Bold", fontSize=11.5,
        textColor=SLATE, spaceBefore=8, spaceAfter=4))
    styles.add(ParagraphStyle(
        name="ExecBody", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=9.8, leading=14.2, alignment=4, textColor=colors.HexColor("#1F2937")))
    styles.add(ParagraphStyle(
        name="ExecBullet", parent=styles["ExecBody"], fontSize=9.4, leading=13, spaceAfter=3))
    styles.add(ParagraphStyle(
        name="ExecCaption", fontName="Helvetica-Oblique", fontSize=8.5,
        textColor=SLATE, spaceBefore=2, spaceAfter=6))
    styles.add(ParagraphStyle(
        name="TableCell", fontName="Helvetica", fontSize=8.5,
        textColor=colors.HexColor("#1F2937"), leading=11))
    styles.add(ParagraphStyle(
        name="TableCellFlag", parent=styles["TableCell"], textColor=ACCENT_RED, fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle(
        name="TableHeader", fontName="Helvetica-Bold", fontSize=8.7, textColor=colors.white, leading=11))
    return styles


class NumberedCanvas(pdfcanvas.Canvas):
    """Canvas that buffers all pages so the footer can show 'Page X of Y' with a true total."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states: list[dict] = []

    def showPage(self) -> None:
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        total_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            _draw_header_footer(self, total_pages)
            super().showPage()
        super().save()


def _draw_header_footer(canvas_obj: pdfcanvas.Canvas, total_pages: int) -> None:
    canvas_obj.saveState()
    width, height = A4

    canvas_obj.setFillColor(NAVY)
    canvas_obj.rect(0, height - 0.62 * inch, width, 0.62 * inch, stroke=0, fill=1)
    canvas_obj.setFillColor(colors.white)
    canvas_obj.setFont("Helvetica-Bold", 10)
    canvas_obj.drawString(0.75 * inch, height - 0.4 * inch, "PROJECT 5: JIO SUBSCRIBER CHURN INTELLIGENCE")
    canvas_obj.setFont("Helvetica", 8)
    canvas_obj.drawRightString(width - 0.75 * inch, height - 0.4 * inch, "Executive Churn Report | Confidential")

    canvas_obj.setFillColor(SLATE)
    canvas_obj.setFont("Helvetica", 7.5)
    canvas_obj.drawString(0.75 * inch, 0.55 * inch, "Revenue Protection & Retention Analytics")
    canvas_obj.drawRightString(
        width - 0.75 * inch, 0.55 * inch, f"Page {canvas_obj.getPageNumber()} of {total_pages}"
    )
    canvas_obj.setStrokeColor(NAVY)
    canvas_obj.setLineWidth(0.6)
    canvas_obj.line(0.75 * inch, 0.72 * inch, width - 0.75 * inch, 0.72 * inch)

    canvas_obj.restoreState()


def bullet_list(items: list[str], styles, bullet_color=SLATE) -> ListFlowable:
    return ListFlowable(
        [ListItem(Paragraph(item, styles["ExecBullet"]), bulletColor=bullet_color) for item in items],
        bulletType="bullet", start="\u25cf", leftIndent=14, bulletFontSize=8,
    )


# --------------------------------------------------------------------------- #
# Page 1: Executive Summary & Context
# --------------------------------------------------------------------------- #
def page1_elements(eda_context: dict[str, Any] | None, metrics_df: pd.DataFrame, styles) -> list:
    ds = (eda_context or {}).get("dataset", {})
    tgt = (eda_context or {}).get("target_analysis", {})
    n_subscribers = ds.get("n_rows", tgt.get("total_subscribers"))
    churn_rate_pct = tgt.get("churn_rate_pct")

    champion_row = metrics_df.loc[metrics_df["Model"].str.contains("XGBoost", case=False, na=False)]
    champion_recall_decile = None
    if not champion_row.empty and "Recall_at_Top_Decile_Pct" in champion_row.columns:
        champion_recall_decile = float(champion_row["Recall_at_Top_Decile_Pct"].iloc[0])

    elements: list = [
        Spacer(1, 0.15 * inch),
        Paragraph("Jio Subscriber Churn Intelligence &amp; Revenue Protection", styles["ExecTitle"]),
        Paragraph("Executive Churn Report &mdash; Diagnostics, Predictive Modeling &amp; Strategic Roadmap",
                   styles["ExecSubtitle"]),
        HRFlowable(width="100%", thickness=1.1, color=NAVY, spaceAfter=12),
        Paragraph("Executive Summary", styles["ExecSectionHeading"]),
        Paragraph(
            "Subscriber growth works like a leaky bucket: new connections pour in at the top through "
            "acquisition, while churn drains value out through the bottom. Even a small monthly churn "
            "rate compounds into substantial lost revenue and acquisition-cost waste over time. This "
            "project builds the analytics and machine-learning capability to spot the leak early - "
            "flagging subscribers likely to churn within the next 30 days - so retention teams can "
            "intervene before the loss becomes permanent.",
            styles["ExecBody"],
        ),
        Spacer(1, 0.15 * inch),
        Paragraph("Project Goals &amp; Scope", styles["ExecSectionHeading"]),
        Paragraph(
            "Project 5 is structured across three workstreams that move from understanding the problem "
            "to acting on it at scale:",
            styles["ExecBody"],
        ),
    ]

    goals = [
        "<b>Workstream 1 - Diagnostics (complete):</b> exploratory data analysis of the full subscriber "
        "base to identify geographic churn hotspots, the recharge 'silent churn' early-warning threshold, "
        "ARPU price sensitivity, and the correlation between network quality-of-service and churn.",
        "<b>Workstream 2 - Predictive Modeling (complete):</b> trained and compared three churn models "
        "(Logistic Regression, XGBoost, Random Forest) on a stratified 70/15/15 train/validation/test "
        "split, with explicit class-imbalance handling and overfitting diagnostics.",
        "<b>Workstream 3 - Conversational Analytics (planned):</b> a chatbot interface for retention and "
        "customer-care teams to query churn risk and drivers in natural language, with strict "
        "'Data not available' guardrails to prevent fabricated answers.",
    ]
    elements.append(bullet_list(goals, styles, bullet_color=NAVY))
    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph("Where We Stand Today", styles["ExecSectionHeading"]))
    rows = [["Metric", "Value"]]
    if n_subscribers:
        rows.append(["Subscribers Analyzed", f"{int(n_subscribers):,}"])
    if churn_rate_pct is not None:
        rows.append(["Baseline 30-Day Churn Rate", f"{churn_rate_pct:.2f}%"])
    if champion_recall_decile is not None:
        rows.append(["Champion Model Recall @ Top 10% Risk", f"{champion_recall_decile:.1f}%"])
    rows.append(["Models Trained &amp; Compared", "3 (Logistic Regression, XGBoost, Random Forest)"])

    if len(rows) > 1:
        table = Table(rows, colWidths=[3.3 * inch, 3.2 * inch], hAlign="LEFT")
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
        elements.append(table)

    if not eda_context:
        elements.append(Spacer(1, 0.1 * inch))
        elements.append(Paragraph(
            "[Note: eda_summary_stats.json not found - run comprehensive_eda.py to populate full dataset "
            "context figures above.]",
            styles["ExecCaption"],
        ))
    return elements


# --------------------------------------------------------------------------- #
# Page 2: Predictive Modeling Performance
# --------------------------------------------------------------------------- #
def metrics_table_flowable(metrics_df: pd.DataFrame, styles) -> Table:
    header = ["Model", "ROC-AUC", "PR-AUC", "Accuracy", "Precision", "Recall", "Overfitting Gap"]
    rows: list = [[Paragraph(h, styles["TableHeader"]) for h in header]]

    for _, r in metrics_df.iterrows():
        gap = float(r["Overfitting_Gap"])
        gap_style = styles["TableCellFlag"] if gap > OVERFIT_ROC_AUC_GAP_THRESHOLD else styles["TableCell"]
        gap_text = f"{gap:+.4f}" + (" (high)" if gap > OVERFIT_ROC_AUC_GAP_THRESHOLD else "")
        rows.append([
            Paragraph(str(r["Model"]), styles["TableCell"]),
            Paragraph(f"{float(r['ROC_AUC']):.4f}", styles["TableCell"]),
            Paragraph(f"{float(r['PR_AUC']):.4f}", styles["TableCell"]),
            Paragraph(f"{float(r['Accuracy']) * 100:.2f}%", styles["TableCell"]),
            Paragraph(f"{float(r['Precision']) * 100:.2f}%", styles["TableCell"]),
            Paragraph(f"{float(r['Recall']) * 100:.2f}%", styles["TableCell"]),
            Paragraph(gap_text, gap_style),
        ])

    col_widths = [1.65 * inch, 0.78 * inch, 0.78 * inch, 0.78 * inch, 0.78 * inch, 0.72 * inch, 1.05 * inch]
    table = Table(rows, colWidths=col_widths, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER_GRAY),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def page2_elements(metrics_df: pd.DataFrame, styles) -> list:
    best_pr_auc_model = metrics_df.sort_values("PR_AUC", ascending=False)["Model"].iloc[0]
    flagged_models = metrics_df.loc[
        metrics_df["Overfitting_Gap"] > OVERFIT_ROC_AUC_GAP_THRESHOLD, "Model"
    ].tolist()

    elements: list = [
        Paragraph("Predictive Modeling Performance", styles["ExecSectionHeading"]),
        Paragraph(
            "Three models were trained on a stratified 70% training split and evaluated on a fully "
            "held-out 15% test set. Because only ~1.7% of subscribers churn in any 30-day window, "
            "PR-AUC (precision-recall area under the curve) is the most reliable ranking metric here - "
            "it is far less forgiving of a model that simply predicts 'no churn' for everyone than "
            "ROC-AUC or raw accuracy would be.",
            styles["ExecBody"],
        ),
        Spacer(1, 0.15 * inch),
        metrics_table_flowable(metrics_df, styles),
        Spacer(1, 0.1 * inch),
        Paragraph(
            "\"Overfitting Gap\" is the difference between a model's training-set ROC-AUC and its "
            "validation-set ROC-AUC; a gap above 0.05 (flagged above) suggests the model is memorizing "
            "training patterns rather than generalizing, and should be regularized or simplified before "
            "production deployment.",
            styles["ExecCaption"],
        ),
        Spacer(1, 0.15 * inch),
    ]

    narrative = (
        f"By PR-AUC, <b>{best_pr_auc_model}</b> ranks highest on this test set. XGBoost remains the "
        "designated champion architecture for production deployment: it offers early-stopping-based "
        "overfitting control, native handling of the severe class imbalance via "
        "<i>scale_pos_weight</i>, and full SHAP-based explainability (Page 3) that regulated retention "
        "workflows require. "
    )
    if flagged_models:
        narrative += (
            f"{', '.join(flagged_models)} exceeded the overfitting-gap threshold and should be "
            "regularized (e.g. shallower trees, stronger L2 penalty, or fewer estimators) before "
            "being considered for production use."
        )
    else:
        narrative += "No model exceeded the overfitting-gap threshold on this run."
    elements.append(Paragraph(narrative, styles["ExecBody"]))
    return elements


# --------------------------------------------------------------------------- #
# Page 3: Model Diagnostics & Explainability
# --------------------------------------------------------------------------- #
def page3_elements(confusion_counts: dict[str, int] | None, styles) -> list:
    elements: list = [
        Paragraph("Model Diagnostics &amp; Explainability", styles["ExecSectionHeading"]),
        Paragraph("Confusion Matrix (XGBoost Champion, Test Set)", styles["ExecSubHeading"]),
        safe_image(
            CONFUSION_MATRIX_PNG_PATH, max_width=3.4 * inch, max_height=2.3 * inch, styles=styles,
            missing_note="run train_churn_models.py to generate confusion_matrix_xgboost.png",
        ),
    ]

    if confusion_counts:
        tn, fp = confusion_counts["true_negatives"], confusion_counts["false_positives"]
        fn, tp = confusion_counts["false_negatives"], confusion_counts["true_positives"]
        total_churners = fn + tp
        cm_bullets = [
            f"The model correctly flagged {tp:,} of {total_churners:,} true churners "
            f"({tp / total_churners * 100:.1f}% recall) on unseen test data.",
            f"It correctly cleared {tn:,} retained subscribers, but also raised {fp:,} false alarms - "
            "the operational cost of casting a wide enough net to catch most real churners.",
            f"Only {fn:,} churners were missed entirely, confirming the model is tuned toward "
            "high sensitivity, appropriate for a low-cost retention outreach (e.g. SMS, app nudge).",
        ]
    else:
        cm_bullets = [
            "[Confusion matrix counts not available - run train_churn_models.py to generate "
            "confusion_matrix_xgboost.json.]",
        ]
    elements.append(bullet_list(cm_bullets, styles))
    elements.append(Spacer(1, 0.1 * inch))

    elements.append(Paragraph("SHAP Explainability - Top Churn Risk Drivers", styles["ExecSubHeading"]))
    elements.append(safe_image(
        SHAP_PNG_PATH, max_width=3.8 * inch, max_height=2.6 * inch, styles=styles,
        missing_note="run train_churn_models.py to generate best_model_shap_summary.png",
    ))
    shap_bullets = [
        "<b>MNP enquiry flag</b> is the single strongest churn signal - a subscriber who has already "
        "checked mobile number portability is actively shopping for alternatives.",
        "<b>Recharge silence</b> (days since last recharge) confirms the Workstream 1 finding: the "
        "longer a subscriber goes without recharging, the higher their churn risk climbs.",
        "<b>ARPU decline</b> (falling average revenue per user) signals a subscriber quietly reducing "
        "usage before formally leaving - an early, actionable warning sign.",
        "<b>Outgoing-to-competitor usage share</b> shows subscribers already routing calls/usage toward "
        "rival networks, a direct behavioral tell of an in-progress switch.",
    ]
    elements.append(bullet_list(shap_bullets, styles))
    return elements


# --------------------------------------------------------------------------- #
# Page 4: Strategic Recommendations & Next Steps
# --------------------------------------------------------------------------- #
def page4_elements(styles) -> list:
    elements: list = [
        Paragraph("Strategic Recommendations &amp; Next Steps", styles["ExecSectionHeading"]),
        Paragraph("Operational Deployment Roadmap", styles["ExecSubHeading"]),
    ]
    roadmap = [
        "Score the full active subscriber base monthly (or weekly for high-value segments) with the "
        "XGBoost champion model and route the top-risk decile to retention campaigns automatically.",
        "Wire the MNP-enquiry flag and recharge-silence threshold into real-time triggers so outreach "
        "fires within days of a risk signal appearing, not at the next monthly batch cycle.",
        "Stand up a champion/challenger monitoring loop: track live PR-AUC and the overfitting gap "
        "monthly, and retrain automatically if either metric drifts beyond the thresholds set here.",
        "Feed intervention outcomes (offer redemption, post-outreach retention) back into the training "
        "data so the model keeps learning which interventions actually work, not just who is at risk.",
    ]
    elements.append(bullet_list(roadmap, styles, bullet_color=NAVY))
    elements.append(Spacer(1, 0.18 * inch))

    elements.append(Paragraph(
        "Workstream 3: Conversational Analytics (Next Phase)", styles["ExecSubHeading"]
    ))
    elements.append(Paragraph(
        "The next phase extends this analysis into a conversational interface so retention and "
        "customer-care teams can ask natural-language questions ('Which circle has the highest churn "
        "risk this week?') without needing to run scripts or read raw dashboards.",
        styles["ExecBody"],
    ))
    chatbot_points = [
        "Answers must be grounded strictly in the underlying data and model outputs produced by "
        "Workstreams 1 and 2 - no answer may be inferred or fabricated beyond what the data supports.",
        "Any question that cannot be answered from available data or model outputs must receive an "
        "explicit <b>'Data not available'</b> response rather than a best-guess answer - a hard "
        "guardrail to preserve trust with business stakeholders.",
        "All chatbot responses referencing risk scores or churn drivers should cite the source metric "
        "(e.g. SHAP value, segment churn rate) so answers remain auditable.",
        "Access and query logs should be retained for model-governance review, consistent with "
        "responsible-AI practices for customer-facing analytics.",
    ]
    elements.append(bullet_list(chatbot_points, styles))
    return elements


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    try:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        print("Verifying input files...")
        verify_inputs()

        print("\nLoading model comparison metrics...")
        metrics_df = load_metrics()
        confusion_counts = load_confusion_counts()
        eda_context = load_eda_context()

        print("Building PDF report...")
        styles = build_styles()
        doc = SimpleDocTemplate(
            str(PDF_PATH), pagesize=A4,
            leftMargin=0.75 * inch, rightMargin=0.75 * inch,
            topMargin=0.95 * inch, bottomMargin=0.85 * inch,
            title="Project 5: Jio Subscriber Churn Intelligence - Executive Report",
            author="Jio Retention Analytics",
        )

        story: list = []
        story.extend(page1_elements(eda_context, metrics_df, styles))
        story.append(PageBreak())
        story.extend(page2_elements(metrics_df, styles))
        story.append(PageBreak())
        story.extend(page3_elements(confusion_counts, styles))
        story.append(PageBreak())
        story.extend(page4_elements(styles))

        doc.build(story, canvasmaker=NumberedCanvas)

        print(f"\nExecutive report saved to: {PDF_PATH}")

    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception:
        print("\n[ERROR] An unexpected error occurred while generating the executive report:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
