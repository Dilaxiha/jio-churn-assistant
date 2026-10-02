"""
generate_architecture_diagram.py
Project 5: Jio Subscriber Churn Intelligence & Revenue Protection

Purpose
-------
Renders the two-part system architecture diagram (offline build pipeline +
chatbot runtime flow) as a single high-resolution PNG with a solid (non-
transparent) white background, using matplotlib shapes/arrows so it matches
the project's existing Navy/Slate design system without any extra dependencies
(no Mermaid/Node/browser renderer required).

Output
------
  Reports/figures/chatbot_architecture_diagram.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BASE_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = BASE_DIR / "Reports" / "figures"
OUTPUT_PATH = FIGURES_DIR / "chatbot_architecture_diagram.png"

# Corporate design system
NAVY = "#1B365D"
SLATE = "#708090"
DATA_FILL = "#E9ECF2"
ARTIFACT_FILL = "#C9D6E3"
PROCESS_FILL = NAVY
DECISION_FILL = "#C0392B"
UI_FILL = SLATE
WHITE = "#FFFFFF"
DARK_TEXT = "#1F2937"


def draw_box(ax, x, y, w, h, text, facecolor, textcolor=WHITE, fontsize=9.5) -> None:
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.4,rounding_size=1.2",
        linewidth=1.1, edgecolor="#00000022", facecolor=facecolor, zorder=2,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
             color=textcolor, weight="bold", zorder=3, linespacing=1.4)


def draw_arrow(ax, start, end, color=SLATE, style="-|>", lw=1.6, ls="solid", label=None) -> None:
    ax.annotate(
        "", xy=end, xytext=start,
        arrowprops=dict(arrowstyle=style, color=color, lw=lw, linestyle=ls, shrinkA=3, shrinkB=3),
        zorder=1,
    )
    if label:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(mx, my + 1.5, label, ha="center", va="bottom", fontsize=7.5,
                 color=DARK_TEXT, style="italic")


def build_diagram() -> None:
    fig, ax = plt.subplots(figsize=(19, 24), dpi=200)
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    fig.suptitle(
        "Jio Subscriber Churn Intelligence \u2014 System &amp; Chatbot Architecture".replace("&amp;", "&"),
        fontsize=20, weight="bold", color=NAVY, y=0.985,
    )

    # ------------------------------------------------------------------ #
    # Panel divider + section titles
    # ------------------------------------------------------------------ #
    ax.plot([2, 98], [51, 51], color="#D1D5DB", lw=1.4, zorder=1)
    ax.text(2, 96, "1. OFFLINE BUILD PIPELINE  (run once / re-run to refresh)", fontsize=13,
             weight="bold", color=NAVY)
    ax.text(2, 47.5, "2. CHATBOT RUNTIME FLOW  (inside app.py, on every user message)", fontsize=13,
             weight="bold", color=NAVY)

    # ------------------------------------------------------------------ #
    # PANEL 1: Offline build pipeline
    # ------------------------------------------------------------------ #
    draw_box(ax, 2, 84, 18, 8, "Data/\nsubscribers.csv", DATA_FILL, DARK_TEXT)
    draw_box(ax, 26, 84, 20, 8, "comprehensive_\neda.py", PROCESS_FILL)
    draw_box(ax, 52, 84, 30, 8, "eda_summary_stats.json\n+ Reports/figures/*.png", ARTIFACT_FILL, DARK_TEXT)

    draw_box(ax, 2, 62, 40, 16,
             "train_churn_models.py\n\nFeature Engineering  \u2192\nColumnTransformer/Pipeline  \u2192\n"
             "Optuna 5-fold CV tuning  \u2192\nXGBoost Champion (early-stopped)  \u2192\nSHAP TreeExplainer",
             PROCESS_FILL, fontsize=9)

    draw_box(ax, 52, 71, 30, 7, "xgboost_champion_bundle.joblib\n(model + preprocessor + threshold)",
             ARTIFACT_FILL, DARK_TEXT, fontsize=8.5)
    draw_box(ax, 52, 62, 30, 7, "comprehensive_model_metrics.csv\n+ confusion_matrix_xgboost.json",
             ARTIFACT_FILL, DARK_TEXT, fontsize=8.5)
    draw_box(ax, 52, 53, 30, 7, "best_model_shap_summary.png", ARTIFACT_FILL, DARK_TEXT, fontsize=8.5)

    draw_arrow(ax, (20, 88), (26, 88))
    draw_arrow(ax, (46, 88), (52, 88))
    draw_arrow(ax, (11, 84), (11, 78))
    draw_arrow(ax, (42, 74.5), (52, 74.5))
    draw_arrow(ax, (42, 65.5), (52, 65.5))
    draw_arrow(ax, (42, 68), (52, 56.5))

    # ------------------------------------------------------------------ #
    # PANEL 2: Chatbot runtime flow
    # ------------------------------------------------------------------ #
    draw_box(ax, 2, 40, 18, 6, "User types\na question", SLATE)
    draw_box(ax, 24, 40, 16, 6, "st.chat_input", SLATE)
    draw_box(ax, 44, 40, 24, 6, "is_in_scope()\nguardrail", DECISION_FILL)

    draw_box(ax, 72, 40, 26, 6, "SAFETY_MESSAGE\n(exact refusal text)", DECISION_FILL, fontsize=8.5)
    draw_box(ax, 2, 30, 36, 6, "answer_query() dispatcher", PROCESS_FILL)

    draw_box(ax, 2, 18, 22, 8, "resolve_category_filter()\nre + difflib\n(circle/zone/plan/device)",
             PROCESS_FILL, fontsize=8)
    draw_box(ax, 27, 18, 22, 8, "score_all_subscribers() +\nclassify_risk_tiers()\n(XGBoost predict_proba)",
             PROCESS_FILL, fontsize=8)
    draw_box(ax, 52, 18, 30, 8,
             "Static lookups: churn rate, drivers,\nchurn_reason, findings, model metrics",
             PROCESS_FILL, fontsize=8)

    draw_box(ax, 24, 8, 24, 6, "Formatted answer", ARTIFACT_FILL, DARK_TEXT)
    draw_box(ax, 54, 8, 22, 6, "st.session_state.\nchat_history", SLATE, fontsize=8.5)
    draw_box(ax, 80, 8, 18, 6, "st.chat_message\n(rendered in UI)", SLATE, fontsize=8.5)

    draw_arrow(ax, (20, 43), (24, 43))
    draw_arrow(ax, (40, 43), (44, 43))
    draw_arrow(ax, (68, 44), (72, 44), label="out of scope")
    draw_arrow(ax, (56, 40), (20, 33), label="in scope")
    draw_arrow(ax, (12, 30), (12, 26))
    draw_arrow(ax, (20, 33), (38, 26))
    draw_arrow(ax, (20, 33), (67, 26))
    draw_arrow(ax, (13, 18), (32, 14))
    draw_arrow(ax, (38, 18), (36, 14))
    draw_arrow(ax, (63, 18), (44, 14))
    draw_arrow(ax, (85, 40), (85, 14), label="")
    draw_arrow(ax, (48, 11), (54, 11))
    draw_arrow(ax, (76, 11), (80, 11))

    # Cross-panel data-feed links (dashed) from offline artifacts into the runtime resolvers.
    draw_arrow(ax, (2, 87.5), (13, 26), color="#9CA3AF", style="-|>", lw=1.1, ls="dashed")
    draw_arrow(ax, (67, 71), (38, 26), color="#9CA3AF", style="-|>", lw=1.1, ls="dashed")
    draw_arrow(ax, (67, 62), (67, 26), color="#9CA3AF", style="-|>", lw=1.1, ls="dashed")
    draw_arrow(ax, (67, 84), (67, 26), color="#9CA3AF", style="-|>", lw=1.1, ls="dashed")

    # ------------------------------------------------------------------ #
    # Legend
    # ------------------------------------------------------------------ #
    legend_items = [
        ("Data / artifact file", DATA_FILL, DARK_TEXT),
        ("Generated artifact", ARTIFACT_FILL, DARK_TEXT),
        ("Script / process", PROCESS_FILL, WHITE),
        ("Decision / guardrail", DECISION_FILL, WHITE),
        ("Streamlit UI / state", SLATE, WHITE),
    ]
    lx = 2
    for label, fill, txt_color in legend_items:
        draw_box(ax, lx, 1, 15, 3.2, label, fill, txt_color, fontsize=7.5)
        lx += 17

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUTPUT_PATH, dpi=200, facecolor=WHITE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    build_diagram()
    print(f"Architecture diagram saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
