"""
generate_app_demo_pdf.py
Project 5: Jio Subscriber Churn Intelligence & Revenue Protection

Purpose
-------
Builds a client-ready "app demonstration" PDF that walks through the three pages of the
Streamlit application (Executive Overview, Subscriber Risk Lookup, Conversational Churn
Assistant) using real screenshots and real example Q&A captured from a live run of the app,
plus the generated EDA/model charts and architecture diagram already in Reports/figures.

Output
------
  Reports/App_Demo_Walkthrough.pdf
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "Reports"
FIGURES_DIR = REPORTS_DIR / "figures"
PDF_PATH = REPORTS_DIR / "App_Demo_Walkthrough.pdf"

NAVY = colors.HexColor("#1B365D")
SLATE = colors.HexColor("#708090")
LIGHT_BG = colors.HexColor("#F2F4F7")
ACCENT_RED = colors.HexColor("#C0392B")
ACCENT_GREEN = colors.HexColor("#1B7A43")
BORDER_GRAY = colors.HexColor("#D1D5DB")

PAGE_W, _ = A4
CONTENT_W = PAGE_W - 1.5 * inch


# --------------------------------------------------------------------------- #
# Screenshot cleanup: the live-app screenshots have excess white margin baked in
# (browser panel sizing), so trim it before laying each one out on the page.
# --------------------------------------------------------------------------- #
def crop_whitespace(src: Path, dst: Path, pad: int = 12) -> Path:
    img = PILImage.open(src).convert("RGB")
    gray = img.convert("L")
    # Anything darker than near-white is considered content.
    bbox = gray.point(lambda p: 0 if p > 248 else 255).getbbox()
    if bbox:
        left, top, right, bottom = bbox
        left = max(0, left - pad)
        top = max(0, top - pad)
        right = min(img.width, right + pad)
        bottom = min(img.height, bottom + pad)
        img = img.crop((left, top, right, bottom))
    img.save(dst)
    return dst


def styled_image(path: Path, max_width: float = CONTENT_W, max_height: float = 3.6 * inch) -> RLImage:
    with PILImage.open(path) as im:
        w, h = im.size
    scale = min(max_width / w, max_height / h, 1.0)
    return RLImage(str(path), width=w * scale, height=h * scale)


def build_styles() -> dict:
    base = getSampleStyleSheet()
    styles = {
        "Title": ParagraphStyle(
            "DemoTitle", parent=base["Title"], fontName="Helvetica-Bold", fontSize=26,
            textColor=NAVY, spaceAfter=6,
        ),
        "Subtitle": ParagraphStyle(
            "DemoSubtitle", parent=base["Normal"], fontName="Helvetica", fontSize=12,
            textColor=SLATE, spaceAfter=18, leading=16,
        ),
        "H1": ParagraphStyle(
            "DemoH1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=18,
            textColor=NAVY, spaceBefore=4, spaceAfter=10,
        ),
        "H2": ParagraphStyle(
            "DemoH2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=13,
            textColor=NAVY, spaceBefore=14, spaceAfter=6,
        ),
        "Body": ParagraphStyle(
            "DemoBody", parent=base["Normal"], fontName="Helvetica", fontSize=10.3,
            textColor=colors.HexColor("#2B2B2B"), leading=15, spaceAfter=8,
        ),
        "Caption": ParagraphStyle(
            "DemoCaption", parent=base["Normal"], fontName="Helvetica-Oblique", fontSize=9,
            textColor=SLATE, spaceBefore=4, spaceAfter=14,
        ),
        "ChatUser": ParagraphStyle(
            "ChatUser", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=10,
            textColor=NAVY, leading=14,
        ),
        "ChatBot": ParagraphStyle(
            "ChatBot", parent=base["Normal"], fontName="Helvetica", fontSize=10,
            textColor=colors.HexColor("#2B2B2B"), leading=14,
        ),
    }
    return styles


def section_rule() -> HRFlowable:
    return HRFlowable(width="100%", thickness=1, color=BORDER_GRAY, spaceBefore=4, spaceAfter=14)


def footer(canvas, doc) -> None:  # noqa: ANN001 - reportlab callback signature
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(SLATE)
    canvas.drawString(0.75 * inch, 0.5 * inch, "Jio Subscriber Churn Intelligence & Revenue Protection")
    canvas.drawRightString(PAGE_W - 0.75 * inch, 0.5 * inch, f"Page {doc.page}")
    canvas.restoreState()


# --------------------------------------------------------------------------- #
# Page content
# --------------------------------------------------------------------------- #
def build_title_page(story: list, styles: dict) -> None:
    story.append(Spacer(1, 0.6 * inch))
    story.append(Paragraph("Jio Subscriber Churn Intelligence &amp; Revenue Protection", styles["Title"]))
    story.append(Paragraph("Application Demonstration Walkthrough", styles["Subtitle"]))
    story.append(section_rule())

    story.append(Paragraph("What this app does", styles["H2"]))
    story.append(Paragraph(
        "A 3-page Streamlit executive application built on top of a SQLite database "
        "(<b>jio_retention.db</b>) and an Optuna-tuned XGBoost champion model. It lets a "
        "business user explore the subscriber base, score any individual subscriber's churn "
        "risk live, and ask free-text questions that are answered with real-time SQL queries "
        "against the database - not canned responses.", styles["Body"],
    ))

    story.append(Paragraph("Headline results", styles["H2"]))
    data = [
        ["Metric", "Value"],
        ["Subscribers analyzed", "64,738"],
        ["Baseline 30-day churn rate", "1.73% (1,122 churned)"],
        ["Champion model", "XGBoost (Optuna-tuned)"],
        ["ROC-AUC", "0.9473"],
        ["PR-AUC", "0.2989"],
        ["Recall @ Top 10% Risk", "87.5%"],
        ["Top churn signal", "MNP (portability) enquiry"],
    ]
    table = Table(data, colWidths=[2.6 * inch, 3.2 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER_GRAY),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(table)

    story.append(Paragraph("The three pages", styles["H2"]))
    for label, desc in [
        ("1. Executive Overview &amp; Metrics", "KPI cards, model comparison table, and 6 diagnostic charts."),
        ("2. Subscriber Risk Lookup", "Search any subscriber_id for a live churn-risk score and plain-English "
                                       "explanation of why they're flagged."),
        ("3. Conversational Churn Assistant", "Ask natural-language questions; every answer is fetched live via "
                                               "SQL from jio_retention.db, with a strict 'Data not available' "
                                               "guardrail for anything out of scope."),
    ]:
        story.append(Paragraph(f"<b>{label}</b> - {desc}", styles["Body"]))

    story.append(PageBreak())


def build_overview_page(story: list, styles: dict, images: dict) -> None:
    story.append(Paragraph("Page 1 - Executive Overview &amp; Metrics", styles["H1"]))
    story.append(Paragraph(
        "A business snapshot of the subscriber base: dataset KPIs, champion model performance "
        "versus the baseline and alternative models, key diagnostic findings, and visual "
        "diagnostics - all computed once and cached for instant loading.", styles["Body"],
    ))
    if "overview" in images:
        story.append(styled_image(images["overview"]))
        story.append(Paragraph(
            "Live screenshot: dataset snapshot KPI cards and predictive model performance cards "
            "(champion XGBoost vs. the model comparison table).", styles["Caption"],
        ))
    story.append(section_rule())

    story.append(Paragraph("Visual diagnostics generated by the EDA pipeline", styles["H2"]))
    chart_pairs = [
        ("exec_circle_risk_chart.png", "Geographic Churn Hotspots"),
        ("exec_recharge_gap_chart.png", "Recharge Silent-Churn Threshold"),
        ("exec_arpu_band_chart.png", "ARPU Band Analysis"),
        ("exec_network_kpi_chart.png", "Network KPI Overview"),
    ]
    rows = []
    for fname, caption in chart_pairs:
        path = FIGURES_DIR / fname
        if path.exists():
            rows.append([styled_image(path, max_width=2.9 * inch, max_height=2.0 * inch), caption])
    for img, caption in rows:
        story.append(img)
        story.append(Paragraph(caption, styles["Caption"]))

    story.append(PageBreak())


def build_risk_lookup_page(story: list, styles: dict, images: dict) -> None:
    story.append(Paragraph("Page 2 - Subscriber Risk Lookup", styles["H1"]))
    story.append(Paragraph(
        "Enter any <b>subscriber_id</b> and get an instant churn-risk score from the persisted "
        "XGBoost champion bundle, plus a plain-English rationale mirroring the model's SHAP "
        "drivers (MNP enquiry, recharge silence, ARPU decline, competitor usage, unresolved "
        "complaints).", styles["Body"],
    ))
    story.append(Paragraph("Live example: subscriber <b>JIO10022692</b>", styles["H2"]))
    if "risk_score" in images:
        story.append(styled_image(images["risk_score"], max_height=1.6 * inch))
        story.append(Paragraph(
            "Churn Risk Score 98.2%, classified High Risk - and the ground-truth label confirms "
            "this subscriber did in fact churn.", styles["Caption"],
        ))
    if "risk_indicator" in images:
        story.append(styled_image(images["risk_indicator"], max_height=1.9 * inch))
        story.append(Paragraph(
            "Key risk indicator: 210 days since last recharge vs. a network average of 42 days - "
            "a textbook 'silent churn' signal.", styles["Caption"],
        ))
    story.append(section_rule())
    story.append(Paragraph("Why this subscriber was flagged (generated live)", styles["H2"]))
    for reason in [
        "<b>MNP enquiry on record</b> - already checked mobile number portability, the strongest "
        "churn signal in the SHAP analysis.",
        "<b>Recharge silence</b> - 210 days since last recharge vs. a 42-day network average.",
        "<b>ARPU decline</b> - last month's spend (INR 100.20) dropped below their own 3-month "
        "average (INR 107.45).",
        "<b>High competitor usage share</b> - 57.3% of outgoing usage already routed to competitor "
        "networks (network average: 28.2%).",
    ]:
        story.append(Paragraph(f"&bull; {reason}", styles["Body"]))

    story.append(PageBreak())


def chat_bubble_table(pairs: list[tuple[str, str]], styles: dict) -> Table:
    rows = []
    for question, answer in pairs:
        rows.append([Paragraph(f"You: {question}", styles["ChatUser"])])
        rows.append([Paragraph(f"Assistant: {answer}", styles["ChatBot"])])
    table = Table(rows, colWidths=[CONTENT_W])
    style = [
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]
    for i in range(len(rows)):
        bg = LIGHT_BG if i % 2 == 0 else colors.white
        style.append(("BACKGROUND", (0, i), (0, i), bg))
    table.setStyle(TableStyle(style))
    return table


def build_chat_assistant_page(story: list, styles: dict) -> None:
    story.append(Paragraph("Page 3 - Conversational Churn Assistant", styles["H1"]))
    story.append(Paragraph(
        "Every answer below was captured live from the running app - no canned responses. Table "
        "and column identifiers used in the generated SQL are always validated against the "
        "database's own schema (sqlite_master / PRAGMA table_info), never built from raw user "
        "text, so the assistant is safe from SQL injection while still answering open-ended "
        "questions across every table in the database.", styles["Body"],
    ))
    story.append(section_rule())

    pairs = [
        ("how many subscribers in Gujarat",
         "There are 3,679 subscribers in Gujarat (5.7% of the 64,738 total)."),
        ("what is the average arpu_last_month_inr",
         "The average arpu_last_month_inr in subscribers is 213.70."),
        ("how many are high risk and from which circle",
         "2,567 subscribers are classified as High Risk (decision threshold 0.84). Note: this "
         "dataset's geographic field is circle (telecom circle), not city - breakdown by circle "
         "(highest: Uttar Pradesh East, 357 of 5,699 subscribers = 6.3%)."),
        ("what is the capital of France",
         "I am restricted to answering questions related to Jio subscriber churn intelligence "
         "and retention metrics. Data not available for out-of-scope queries."),
    ]
    story.append(chat_bubble_table(pairs, styles))
    story.append(Spacer(1, 10))

    story.append(Paragraph("High Risk subscribers by circle (live SQL + model scoring)", styles["H2"]))
    data = [["Circle", "High Risk Subscribers", "Total Subscribers", "% of Circle"]]
    for row in [
        ("Uttar Pradesh East", "357", "5,699", "6.3%"),
        ("Madhya Pradesh & Chhattisgarh", "246", "4,384", "5.6%"),
        ("Bihar & Jharkhand", "245", "4,788", "5.1%"),
        ("Uttar Pradesh West", "232", "3,672", "6.3%"),
        ("Rajasthan", "169", "3,714", "4.6%"),
    ]:
        data.append(list(row))
    table = Table(data, colWidths=[2.1 * inch, 1.6 * inch, 1.4 * inch, 1.0 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER_GRAY),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(table)

    story.append(PageBreak())


def build_architecture_page(story: list, styles: dict) -> None:
    story.append(Paragraph("System Architecture", styles["H1"]))
    diagram = FIGURES_DIR / "chatbot_architecture_diagram.png"
    if diagram.exists():
        story.append(styled_image(diagram, max_height=5.2 * inch))
    story.append(section_rule())

    story.append(Paragraph("Tech stack", styles["H2"]))
    for line in [
        "<b>Data layer:</b> subscribers.csv + Jio_Retention_Dataset.xlsx, compiled into a SQLite "
        "database (jio_retention.db) via build_database.py.",
        "<b>Modeling:</b> Logistic Regression baseline, Optuna-tuned XGBoost champion, Random "
        "Forest alternative - compared on ROC-AUC, PR-AUC, and Recall@Top-Decile.",
        "<b>App:</b> Streamlit, with st.cache_resource/st.cache_data for the DB connection, "
        "schema introspection, and the persisted model bundle.",
        "<b>Assistant:</b> rule-based NL intent resolvers that generate parameterized SQL - "
        "counts, aggregates, unique values, top-N breakdowns, and raw-row samples - against any "
        "table in the database.",
    ]:
        story.append(Paragraph(f"&bull; {line}", styles["Body"]))

    story.append(Paragraph("Running it locally", styles["H2"]))
    story.append(Paragraph(
        "pip install -r requirements.txt &nbsp;&rarr;&nbsp; python build_database.py &nbsp;&rarr;&nbsp; "
        "streamlit run app.py", styles["Body"],
    ))


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    screenshot_specs = {
        "overview": FIGURES_DIR / "app_demo_overview_2.png",
        "risk_score": FIGURES_DIR / "app_demo_risklookup_2.png",
        "risk_indicator": FIGURES_DIR / "app_demo_risklookup_3.png",
    }
    images: dict[str, Path] = {}
    for key, src in screenshot_specs.items():
        if src.exists():
            dst = src.with_name(src.stem + "_cropped.png")
            images[key] = crop_whitespace(src, dst)

    styles = build_styles()
    doc = SimpleDocTemplate(
        str(PDF_PATH), pagesize=A4,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        title="Jio Churn Intelligence - App Demo Walkthrough",
    )

    story: list = []
    build_title_page(story, styles)
    build_overview_page(story, styles, images)
    build_risk_lookup_page(story, styles, images)
    build_chat_assistant_page(story, styles)
    build_architecture_page(story, styles)

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Wrote {PDF_PATH}")


if __name__ == "__main__":
    main()
