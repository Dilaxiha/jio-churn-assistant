"""
app.py
Project 5: Jio Subscriber Churn Intelligence & Revenue Protection

A Streamlit executive application with three views:
  1. Executive Overview & Metrics   - KPI cards, model performance, key charts.
  2. Subscriber Risk Lookup         - search a subscriber and get a live churn-risk score.
  3. Conversational Churn Assistant - Q&A answered by running live SQL queries against the
                                       SQLite database (Data/jio_retention.db), with an
                                       explicit "Data not available" guardrail.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import difflib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# --------------------------------------------------------------------------- #
# Path management
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "Data" / "jio_retention.db"
REPORTS_DIR = BASE_DIR / "Reports"
FIGURES_DIR = REPORTS_DIR / "figures"
CHAMPION_BUNDLE_PATH = REPORTS_DIR / "xgboost_champion_bundle.joblib"

TARGET = "churn_flag_30d"
LEAKAGE_COLS = ["subscriber_id", "churn_flag_90d", "churn_reason", "churn_date", "join_date"]

NAVY = "#1B365D"
SLATE = "#708090"


# --------------------------------------------------------------------------- #
# Page config & styling
# --------------------------------------------------------------------------- #
def configure_page() -> None:
    st.set_page_config(
        page_title="Jio Subscriber Churn Intelligence",
        page_icon="\U0001F4F6",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_custom_css() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{ background-color: #F5F7FA; }}

        [data-testid="stSidebar"] {{ background-color: {NAVY}; }}
        [data-testid="stSidebar"] * {{ color: #F1F3F5 !important; }}
        [data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.15); }}

        .sidebar-brand {{
            font-size: 1.55rem; font-weight: 800; letter-spacing: 0.08em; color: #FFFFFF;
            margin-bottom: 0;
        }}
        .sidebar-subtitle {{
            font-size: 0.8rem; color: #C9D3DC; margin-top: 0.1rem; margin-bottom: 0.6rem;
        }}

        .page-title {{ color: {NAVY}; font-weight: 800; margin-bottom: 0; }}
        .page-subtitle {{
            color: {SLATE}; font-size: 0.95rem; margin-top: 0.15rem; margin-bottom: 1.4rem;
        }}
        .section-heading {{
            color: {NAVY}; font-weight: 700; font-size: 1.15rem;
            margin-top: 1.4rem; margin-bottom: 0.6rem;
        }}

        .metric-card {{
            background: #FFFFFF; border-radius: 10px; padding: 1rem 1.1rem;
            box-shadow: 0 1px 4px rgba(0,0,0,0.08); border-left: 5px solid {NAVY};
            margin-bottom: 0.8rem; min-height: 96px;
        }}
        .metric-icon {{ font-size: 1.2rem; margin-bottom: 0.15rem; }}
        .metric-value {{ font-size: 1.4rem; font-weight: 700; color: {NAVY}; line-height: 1.25; }}
        .metric-label {{
            font-size: 0.76rem; color: {SLATE}; font-weight: 600;
            text-transform: uppercase; letter-spacing: 0.03em; margin-top: 0.15rem;
        }}
        .metric-sublabel {{ font-size: 0.72rem; color: #9CA3AF; margin-top: 0.2rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card_html(icon: str, label: str, value: str, sublabel: str = "", accent: str = NAVY) -> str:
    sub_html = f'<div class="metric-sublabel">{sublabel}</div>' if sublabel else ""
    return f"""
        <div class="metric-card" style="border-left-color:{accent};">
            <div class="metric-icon">{icon}</div>
            <div class="metric-value" style="color:{accent};">{value}</div>
            <div class="metric-label">{label}</div>
            {sub_html}
        </div>
    """


# --------------------------------------------------------------------------- #
# Backend data-loading module (st.cache_resource)
#
# These loaders own the heavyweight, process-lifetime resources - the raw
# subscriber dataset, the model comparison table, and the persisted XGBoost
# champion bundle (model + preprocessor) produced by train_churn_models.py.
# Each loader fails gracefully: missing/corrupt files surface a friendly
# st.warning/st.error instead of crashing the app, and callers always receive
# None so pages can degrade (fallback model, hidden section, etc.) instead of
# raising an unhandled exception.
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_db_connection() -> sqlite3.Connection | None:
    """Shared read-only connection to the SQLite database built by build_database.py from
    subscribers.csv and every sheet of Jio_Retention_Dataset.xlsx."""
    if not DB_PATH.exists():
        st.error(
            f"`jio_retention.db` not found at `{DB_PATH}`. Run `python build_database.py` "
            "(in Python_Scripts/) to build it from subscribers.csv and Jio_Retention_Dataset.xlsx."
        )
        return None
    uri = f"file:{DB_PATH.as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True, check_same_thread=False)


@st.cache_resource(show_spinner="Loading subscriber dataset from the database...")
def load_subscribers_data() -> pd.DataFrame | None:
    conn = get_db_connection()
    if conn is None:
        return None
    try:
        subs = pd.read_sql_query("SELECT * FROM subscribers", conn)
        # SQLite has no native boolean type - flag columns round-trip as 0/1 integers, so
        # restore bool dtype to match the original CSV/UI expectations.
        bool_cols = [
            "autopay_enabled", "is_5g_device", "is_5g_active", "family_plan_flag", "roaming_user_flag",
            "offer_exposed_90d", "offer_redeemed_90d", "mnp_enquiry_flag", "churn_flag_30d", "churn_flag_90d",
        ]
        for col in bool_cols:
            if col in subs.columns:
                subs[col] = subs[col].astype(bool)
        return subs
    except Exception as e:  # noqa: BLE001 - surface any load failure as a friendly warning
        st.warning(f"Could not load subscriber data from the database ({e}). Reload the app.")
        return None


@st.cache_data(show_spinner=False)
def get_db_schema(_conn: sqlite3.Connection) -> dict[str, dict[str, str]]:
    """table name -> {column name: declared SQL type}, read once from sqlite_master/PRAGMA so the
    assistant can validate identifiers before building any SQL (table/column names always come
    from this allowlist, never interpolated directly from user text)."""
    schema: dict[str, dict[str, str]] = {}
    tables = [r[0] for r in _conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    for table in tables:
        cols = _conn.execute(f"PRAGMA table_info('{table}')").fetchall()
        schema[table] = {row[1]: row[2] for row in cols}
    return schema


@st.cache_data(show_spinner=False)
def get_subscriber_dimensions(_conn: sqlite3.Connection) -> dict[str, list[str]]:
    """Distinct circle/circle_code/zone/plan_type/device_brand values from the subscribers table,
    used to match a query's free text against a real category before building a WHERE clause."""
    dims: dict[str, list[str]] = {}
    for col in ["circle", "circle_code", "zone", "plan_type", "device_brand"]:
        rows = _conn.execute(f"SELECT DISTINCT {col} FROM subscribers WHERE {col} IS NOT NULL").fetchall()
        dims[col] = [r[0] for r in rows]
    return dims


def sql_scalar(conn: sqlite3.Connection, query: str, params: tuple | list = ()) -> Any:
    return conn.execute(query, params).fetchone()[0]


def sql_df(conn: sqlite3.Connection, query: str, params: tuple | list = ()) -> pd.DataFrame:
    return pd.read_sql_query(query, conn, params=params)


NUMERIC_SQL_TYPES = {"INTEGER", "REAL", "BIGINT", "FLOAT", "NUMERIC", "BOOLEAN"}


def is_numeric_sql_type(sql_type: str) -> bool:
    return sql_type.upper() in NUMERIC_SQL_TYPES


@st.cache_resource(show_spinner="Loading model comparison metrics...")
def load_comprehensive_metrics() -> pd.DataFrame | None:
    try:
        path = REPORTS_DIR / "comprehensive_model_metrics.csv"
        if not path.exists():
            raise FileNotFoundError(f"comprehensive_model_metrics.csv not found at {path}")
        return pd.read_csv(path)
    except Exception as e:  # noqa: BLE001
        st.warning(f"Could not load model comparison metrics ({e}). Run `train_churn_models.py` to generate it.")
        return None


@st.cache_resource(show_spinner="Loading champion XGBoost model bundle...")
def load_xgboost_artifacts() -> dict[str, Any] | None:
    """Load the persisted champion model + preprocessor saved by train_churn_models.py, if available."""
    try:
        if not CHAMPION_BUNDLE_PATH.exists():
            raise FileNotFoundError(f"xgboost_champion_bundle.joblib not found at {CHAMPION_BUNDLE_PATH}")
        bundle = joblib.load(CHAMPION_BUNDLE_PATH)
        required_keys = {"model", "preprocessor", "numeric_cols", "categorical_cols"}
        if not required_keys.issubset(bundle.keys()):
            raise ValueError("Bundle is missing expected keys (model/preprocessor/feature columns).")
        return bundle
    except Exception as e:  # noqa: BLE001
        st.warning(
            f"Pre-trained XGBoost champion model not available ({e}). Falling back to a lightweight "
            "in-app model for Subscriber Risk Lookup. Run `train_churn_models.py` to generate it."
        )
        return None


# --------------------------------------------------------------------------- #
# Small JSON loaders (lightweight, re-read cheaply - st.cache_data is enough)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_eda_stats() -> dict[str, Any] | None:
    path = REPORTS_DIR / "eda_summary_stats.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_confusion_info() -> dict[str, Any] | None:
    path = REPORTS_DIR / "confusion_matrix_xgboost.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Shared feature engineering (mirrors train_churn_models.py's engineer_features,
# so both the persisted champion bundle and the in-app fallback model score
# subscribers consistently).
# --------------------------------------------------------------------------- #
def engineer_scoring_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.drop(columns=[c for c in LEAKAGE_COLS + [TARGET] if c in df.columns]).copy()

    features["arpu_ratio_1m_to_3m"] = (
        features["arpu_last_month_inr"] / features["arpu_3m_avg_inr"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    features["arpu_decline_flag"] = (features["arpu_last_month_inr"] < features["arpu_3m_avg_inr"]).astype(int)

    safe_gap = features["avg_recharge_gap_days"].replace(0, np.nan)
    features["recharge_overdue_ratio"] = (features["days_since_last_recharge"] / safe_gap).replace(
        [np.inf, -np.inf], np.nan
    )
    features["long_recharge_gap_flag"] = (features["recharge_overdue_ratio"] > 1.5).astype(int)

    features["network_complaint_intensity"] = features["complaints_6m"] + 2 * features["unresolved_complaints"]

    bool_cols = [c for c in features.columns if pd.api.types.is_bool_dtype(features[c])]
    features[bool_cols] = features[bool_cols].astype(int)
    return features


# --------------------------------------------------------------------------- #
# Natural-language query resolvers: geographic/categorical filtering (executed as
# parameterized SQL against the subscribers table) + live risk-tier scoring via the
# XGBoost champion model (used by the Conversational Churn Assistant).
# --------------------------------------------------------------------------- #
def resolve_category_sql_filter(query: str, dims: dict[str, list[str]]) -> tuple[str, list[Any], str] | None:
    """Match a circle/zone/plan-type/device-brand name or code in the query; return a
    parameterized SQL WHERE clause (against the subscribers table) plus params and a
    human-readable label, or None if no match."""
    q = query.lower()

    code_map = {code.upper(): name for code, name in zip(dims["circle_code"], dims["circle"])}
    for word in re.findall(r"[a-zA-Z]+", query.upper()):
        if word in code_map:
            circle_name = code_map[word]
            return "circle = ?", [circle_name], circle_name

    for circle_name in dims["circle"]:
        if circle_name.lower() in q:
            return "circle = ?", [circle_name], circle_name

    if re.search(r"\bup\b", q):  # "UP" is ambiguous between Uttar Pradesh East/West
        return "circle LIKE ?", ["%Uttar Pradesh%"], "Uttar Pradesh (East + West)"

    for zone_name in dims["zone"]:
        if re.search(rf"\b{re.escape(zone_name.lower())}\b", q):
            return "zone = ?", [zone_name], f"{zone_name} zone"

    for plan_name in dims["plan_type"]:
        if plan_name.lower() in q:
            return "plan_type = ?", [plan_name], f"{plan_name} plan"

    for brand in dims["device_brand"]:
        if brand.lower() in q:
            return "device_brand = ?", [brand], f"{brand} device"

    # Fuzzy fallback for minor typos (e.g. "maharastra" -> "Maharashtra & Goa").
    circle_word_map: dict[str, str] = {}
    for circle_name in dims["circle"]:
        for part in re.findall(r"[a-zA-Z]+", circle_name.lower()):
            if len(part) >= 4:
                circle_word_map[part] = circle_name
    for word in re.findall(r"[a-zA-Z]+", q):
        if len(word) < 4:
            continue
        close = difflib.get_close_matches(word, list(circle_word_map.keys()), n=1, cutoff=0.8)
        if close:
            circle_name = circle_word_map[close[0]]
            return "circle = ?", [circle_name], circle_name

    return None


def format_count_response_sql(conn: sqlite3.Connection, where_clause: str, params: list[Any], label: str) -> str:
    total = sql_scalar(conn, "SELECT COUNT(*) FROM subscribers")
    count = sql_scalar(conn, f"SELECT COUNT(*) FROM subscribers WHERE {where_clause}", params)
    pct = (count / total * 100) if total else 0.0
    return f"There are **{count:,}** subscribers in **{label}** ({pct:.1f}% of the {total:,} total)."


@st.cache_resource(show_spinner="Scoring all subscribers for risk-tier queries...")
def score_all_subscribers(_df: pd.DataFrame) -> np.ndarray:
    """Predicted 30-day churn probability for every subscriber, via the champion model
    (or the lightweight fallback model when the persisted bundle is unavailable)."""
    features_full = engineer_scoring_features(_df)
    bundle = load_xgboost_artifacts()
    if bundle is not None:
        expected_cols = bundle["numeric_cols"] + bundle["categorical_cols"]
        X_proc = bundle["preprocessor"].transform(features_full[expected_cols])
        return bundle["model"].predict_proba(X_proc)[:, 1]

    fallback_bundle = build_risk_model(_df)
    X_proc = fallback_bundle["preprocessor"].transform(fallback_bundle["features_template"])
    return fallback_bundle["model"].predict_proba(X_proc)[:, 1]


def get_high_risk_threshold() -> float:
    bundle = load_xgboost_artifacts()
    if bundle is not None and bundle.get("decision_threshold"):
        return float(bundle["decision_threshold"])
    confusion_info = load_confusion_info()
    return float(confusion_info["decision_threshold"]) if confusion_info else 0.5


def classify_risk_tiers(proba: np.ndarray, high_cutoff: float) -> dict[str, int]:
    med_cutoff = high_cutoff / 2
    return {
        "High Risk": int((proba >= high_cutoff).sum()),
        "Medium Risk": int(((proba >= med_cutoff) & (proba < high_cutoff)).sum()),
        "Low Risk": int((proba < med_cutoff).sum()),
    }


def resolve_risk_tier_query(query: str) -> str | None:
    q = query.lower()
    if "top decile" in q or "top 10%" in q or "top-10%" in q:
        return "top_decile"
    if "high risk" in q or "high-risk" in q:
        return "high"
    if "medium risk" in q or "medium-risk" in q:
        return "medium"
    if "low risk" in q or "low-risk" in q:
        return "low"
    if any(k in q for k in ["risk tier", "risk tiers", "risk breakdown", "risk distribution"]):
        return "all"
    return None


def format_risk_tier_response(tier_key: str, df: pd.DataFrame) -> str | tuple[str, pd.DataFrame]:
    proba = score_all_subscribers(df)
    total = len(proba)
    high_cutoff = get_high_risk_threshold()

    if tier_key == "top_decile":
        top_k = max(1, int(np.ceil(total * 0.10)))
        return f"The top 10% highest-risk decile contains **{top_k:,}** subscribers (out of {total:,} total)."

    tier_counts = classify_risk_tiers(proba, high_cutoff)
    if tier_key == "all":
        table_df = pd.DataFrame(
            [(k, v, round(v / total * 100, 1)) for k, v in tier_counts.items()],
            columns=["Risk Tier", "Subscribers", "Percent"],
        )
        return (
            f"Risk tier breakdown across all {total:,} subscribers (High Risk threshold = "
            f"{high_cutoff:.2f} predicted probability):",
            table_df,
        )

    label = {"high": "High Risk", "medium": "Medium Risk", "low": "Low Risk"}[tier_key]
    count = tier_counts[label]
    return (
        f"**{count:,}** subscribers ({count / total * 100:.1f}%) are classified as **{label}** "
        f"using the champion model's optimized decision threshold ({high_cutoff:.2f})."
    )


GEO_BREAKDOWN_KEYWORDS = [
    "city", "cities", "circle", "circles", "region", "regions", "zone", "zones",
    "location", "where", "which city", "which circle", "which region", "which zone",
    "by city", "by circle", "by region", "by zone", "per city", "per circle",
]


def resolve_risk_tier_by_circle_query(query: str, df: pd.DataFrame) -> tuple[str, pd.DataFrame] | None:
    """Combine live ML risk-tier scoring with a geographic breakdown, for questions like
    'how many subscribers are in high risk and from which city'. The dataset's geographic
    field is 'circle' (telecom circle) - used here since there is no literal 'city' column."""
    q = query.lower()
    tier_key = resolve_risk_tier_query(q)
    if tier_key is None or tier_key == "top_decile":
        return None
    if not any(k in q for k in GEO_BREAKDOWN_KEYWORDS):
        return None

    proba = score_all_subscribers(df)
    high_cutoff = get_high_risk_threshold()
    med_cutoff = high_cutoff / 2

    if tier_key == "high":
        mask, label = proba >= high_cutoff, "High Risk"
    elif tier_key == "medium":
        mask, label = (proba >= med_cutoff) & (proba < high_cutoff), "Medium Risk"
    elif tier_key == "low":
        mask, label = proba < med_cutoff, "Low Risk"
    else:  # "all"
        mask, label = np.ones(len(proba), dtype=bool), "All Tiers"

    scored = pd.DataFrame({"Circle": df["circle"].values, "is_match": mask})
    total_by_circle = scored.groupby("Circle").size()
    match_by_circle = scored[scored["is_match"]].groupby("Circle").size()
    count_col = f"{label} Subscribers"

    table_df = pd.DataFrame({"Circle": total_by_circle.index})
    table_df[count_col] = table_df["Circle"].map(match_by_circle).fillna(0).astype(int)
    table_df["Total Subscribers"] = table_df["Circle"].map(total_by_circle)
    table_df["Percent of Circle"] = (table_df[count_col] / table_df["Total Subscribers"] * 100).round(1)
    table_df = table_df.sort_values(count_col, ascending=False).reset_index(drop=True)

    total_match = int(mask.sum())
    top_circle = table_df.iloc[0]["Circle"] if not table_df.empty else "n/a"
    summary = (
        f"**{total_match:,}** subscribers are classified as **{label}** (decision threshold "
        f"{high_cutoff:.2f}). Note: this dataset's geographic field is **circle** (telecom circle), "
        f"not city - here's the breakdown by circle (highest: **{top_circle}**):"
    )
    return summary, table_df


# --------------------------------------------------------------------------- #
# Generic multi-table query resolver - covers every table in jio_retention.db
# (subscribers, service requests, network sites, monthly circle KPIs, circle
# targets, offer catalogue), executed as live SQL, so the assistant isn't limited
# to the handful of hand-written subscriber-specific handlers above.
# --------------------------------------------------------------------------- #
SHEET_ALIASES: dict[str, list[str]] = {
    "subscribers": ["subscriber", "churn"],
    "service_requests": ["service request", "support ticket", "sr category", "csat", "sla breach", "reopened"],
    "network_sites": ["network site", "cell site", "tower", "prb utilisation", "throughput", "backhaul"],
    "circle_monthly_kpi": ["monthly kpi", "circle kpi", "gross add", "net add", "monthly churn", "closing base"],
    "circle_targets": ["circle target", "retention budget", "budget", "churn ceiling", "fy27", "circle owner"],
    "offer_catalogue": ["offer catalogue", "offer catalog", "discount", "bonus data", "uplift", "offer code"],
    "readme": ["readme", "about the dataset", "data source"],
    "data_dictionary": ["data dictionary", "column meaning", "meaning of", "definition of"],
}

AGG_KEYWORDS: dict[str, str] = {
    "average": "mean", "avg": "mean", "mean": "mean",
    "total": "sum", "sum": "sum",
    "maximum": "max", "max": "max", "highest": "max",
    "minimum": "min", "min": "min", "lowest": "min",
    "median": "median",
}

AGG_SQL_FUNCS: dict[str, str] = {"mean": "AVG", "sum": "SUM", "max": "MAX", "min": "MIN"}


def resolve_sheet_mention(query: str, schema: dict[str, dict[str, str]]) -> str | None:
    """Identify which loaded table a query refers to, defaulting to 'subscribers'."""
    q = query.lower()

    # Direct table-name mention takes priority (supports both "circle_monthly_kpi"
    # and "circle monthly kpi" phrasing) - checked before the alias list so an exact
    # table name always wins over a generic alias keyword.
    for actual_name in schema:
        name_l = actual_name.lower()
        if name_l in q or name_l.replace("_", " ") in q:
            return actual_name

    lookup = {name.lower(): name for name in schema}
    for sheet_key, aliases in SHEET_ALIASES.items():
        actual_name = lookup.get(sheet_key)
        if actual_name and any(alias in q for alias in aliases):
            return actual_name

    if "subscribers" in schema:
        return "subscribers"
    return next(iter(schema), None)


def find_column_match(query: str, columns: list[str]) -> str | None:
    """Longest matching column name (or its space-separated form) mentioned in the query."""
    q = query.lower()
    best: str | None = None
    for col in columns:
        col_l = str(col).lower()
        if col_l in q or col_l.replace("_", " ") in q:
            if best is None or len(col_l) > len(best):
                best = col
    return best


def resolve_generic_stat_query_sql(
    query: str, conn: sqlite3.Connection, schema: dict[str, dict[str, str]]
) -> str | tuple[str, pd.DataFrame] | None:
    """Best-effort answer for arbitrary column/aggregate questions, executed as live SQL
    against any table in jio_retention.db. Table/column names are only ever drawn from
    `schema` (derived from sqlite_master/PRAGMA), never built from raw user text."""
    if not schema:
        return None
    q = query.lower()

    # Column-meaning lookup takes priority over aggregation parsing: "what does X mean"
    # would otherwise collide with the "mean" aggregation keyword below.
    if "data_dictionary" in schema and any(k in q for k in ["what does", "meaning of", "definition of"]):
        dict_columns = [r[0] for r in conn.execute("SELECT DISTINCT column FROM data_dictionary").fetchall()]
        candidate = find_column_match(query, dict_columns)
        if candidate:
            match = sql_df(conn, "SELECT sheet, dtype, note FROM data_dictionary WHERE column = ? LIMIT 1", [candidate])
            if not match.empty:
                sheet_val, dtype_val, note = match.iloc[0][["sheet", "dtype", "note"]]
                note_text = str(note) if pd.notna(note) else "no description available"
                return f"**{candidate}** (sheet: {sheet_val or 'n/a'}, dtype: {dtype_val or 'n/a'}): {note_text}"

    table_name = resolve_sheet_mention(query, schema)
    if not table_name:
        return None
    columns = list(schema[table_name].keys())

    if any(k in q for k in ["how many rows", "how many records", "how many entries", "rows in", "size of"]):
        n_rows = sql_scalar(conn, f"SELECT COUNT(*) FROM {table_name}")
        return f"**{table_name}** has **{n_rows:,}** rows and {len(columns)} columns."

    if any(k in q for k in ["what columns", "which columns", "column list", "columns does", "fields does"]):
        table_df = pd.DataFrame([(c, schema[table_name][c]) for c in columns], columns=["column", "type"])
        return f"**{table_name}** has {len(columns)} columns:", table_df

    agg_key = next((agg for kw, agg in AGG_KEYWORDS.items() if kw in q), None)
    column = find_column_match(query, columns)

    # Cross-table fallback: if the resolved table doesn't have a matching column, search
    # every table and keep the longest (most specific) match, to avoid a short, generic
    # column name (e.g. "value") winning over a more specific one elsewhere.
    if column is None:
        best_match: tuple[str, str] | None = None
        for candidate_table, candidate_cols in schema.items():
            candidate_col = find_column_match(query, list(candidate_cols.keys()))
            if candidate_col and (best_match is None or len(candidate_col) > len(best_match[0])):
                best_match = (candidate_col, candidate_table)
        if best_match:
            column, table_name = best_match

    if column:
        is_numeric = is_numeric_sql_type(schema[table_name][column])

        if agg_key and is_numeric:
            if agg_key == "median":
                values = sql_df(conn, f"SELECT {column} FROM {table_name} WHERE {column} IS NOT NULL")[column]
                value = float(values.median())
            else:
                value = sql_scalar(
                    conn, f"SELECT {AGG_SQL_FUNCS[agg_key]}({column}) FROM {table_name} WHERE {column} IS NOT NULL"
                )
            agg_label = "average" if agg_key == "mean" else agg_key
            return f"The {agg_label} **{column}** in **{table_name}** is **{value:,.2f}**."

        if any(k in q for k in ["unique", "distinct", "list", "what are the", "options", "categories"]):
            n_unique = sql_scalar(
                conn, f"SELECT COUNT(DISTINCT {column}) FROM {table_name} WHERE {column} IS NOT NULL"
            )
            rows = conn.execute(
                f"SELECT DISTINCT {column} FROM {table_name} WHERE {column} IS NOT NULL ORDER BY {column} LIMIT 50"
            ).fetchall()
            table_df = pd.DataFrame(rows, columns=[column])
            more = " (showing first 50)" if n_unique > 50 else ""
            return f"**{column}** in **{table_name}** has **{n_unique}** unique values{more}:", table_df

        if is_numeric:
            mean_v, min_v, max_v, count_v = conn.execute(
                f"SELECT AVG({column}), MIN({column}), MAX({column}), COUNT({column}) FROM {table_name}"
            ).fetchone()
            return (
                f"**{column}** in **{table_name}**: mean={mean_v:,.2f}, min={min_v:,.2f}, "
                f"max={max_v:,.2f}, count={count_v:,}."
            )

        top_rows = conn.execute(
            f"SELECT {column}, COUNT(*) AS cnt FROM {table_name} GROUP BY {column} ORDER BY cnt DESC LIMIT 10"
        ).fetchall()
        table_df = pd.DataFrame(top_rows, columns=[column, "count"])
        return f"Top values for **{column}** in **{table_name}**:", table_df

    return None


ROW_SAMPLE_KEYWORDS = [
    "show", "list", "display", "sample", "raw data", "original data", "actual data", "records",
]
# Aggregate/count/unique-value phrasing is already handled by other resolvers above - excluded
# here so e.g. "list the unique plan types" still returns a unique-value table, not raw rows.
ROW_SAMPLE_EXCLUDE_KEYWORDS = [
    "how many", "count", "risk tier", "risk tiers", "high risk", "medium risk", "low risk",
    "top decile", "top 10%", "average", "avg", "mean", "total", "sum", "maximum", "max",
    "minimum", "min", "highest", "lowest", "median", "unique", "distinct", "what columns",
    "which columns", "column list", "columns does", "fields does", "how many rows",
    "what does", "meaning of", "definition of",
]

ROW_SAMPLE_LIMIT = 100


def resolve_row_sample_query_sql(
    query: str,
    conn: sqlite3.Connection,
    schema: dict[str, dict[str, str]],
    dims: dict[str, list[str]] | None,
) -> tuple[str, pd.DataFrame] | None:
    """Return a literal sample of matching rows (not an aggregate) for queries that explicitly
    ask to see the underlying records/raw data, e.g. 'show subscribers in Gujarat'. Requires an
    explicit table/alias mention (never falls back to a default table) so unrelated 'show'/'list'
    queries aren't hijacked into dumping a table."""
    q = query.lower()
    if not any(k in q for k in ROW_SAMPLE_KEYWORDS):
        return None
    if any(k in q for k in ROW_SAMPLE_EXCLUDE_KEYWORDS):
        return None

    table_name = None
    for actual_name in schema:
        name_l = actual_name.lower()
        if name_l in q or name_l.replace("_", " ") in q:
            table_name = actual_name
            break
    if table_name is None:
        lookup = {name.lower(): name for name in schema}
        for sheet_key, aliases in SHEET_ALIASES.items():
            actual_name = lookup.get(sheet_key)
            if actual_name and any(alias in q for alias in aliases):
                table_name = actual_name
                break
    if table_name is None:
        return None

    where_clause, params, label = "", [], None
    if table_name == "subscribers" and dims is not None:
        match = resolve_category_sql_filter(query, dims)
        if match is not None:
            where_clause, params, label = match

    sql = f"SELECT * FROM {table_name}"
    if where_clause:
        sql += f" WHERE {where_clause}"
    sql += " LIMIT ?"
    table_df = sql_df(conn, sql, [*params, ROW_SAMPLE_LIMIT])

    total_sql = f"SELECT COUNT(*) FROM {table_name}" + (f" WHERE {where_clause}" if where_clause else "")
    total = sql_scalar(conn, total_sql, params)

    scope_label = f" in **{label}**" if label else ""
    shown = len(table_df)
    coverage = f" (showing first {shown:,} of {total:,})" if total > shown else f" ({total:,} total)"
    return f"Sample records from **{table_name}**{scope_label}{coverage}:", table_df


# --------------------------------------------------------------------------- #
# Lightweight in-app scoring model (interactive demo only, used only when the
# persisted champion bundle from train_churn_models.py is unavailable).
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Training churn risk model for interactive lookup (one-time)...")
def build_risk_model(_df: pd.DataFrame) -> dict[str, Any]:
    y = _df[TARGET].astype(int)
    features = engineer_scoring_features(_df)

    numeric_cols = [
        c for c in features.columns if pd.api.types.is_numeric_dtype(features[c])
    ]
    categorical_cols = [c for c in features.columns if c not in numeric_cols]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical_cols),
    ])
    X_proc = preprocessor.fit_transform(features)

    neg, pos = np.bincount(y)
    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=neg / pos, eval_metric="aucpr",
        random_state=42, n_jobs=-1,
    )
    model.fit(X_proc, y)

    return {"model": model, "preprocessor": preprocessor, "features_template": features}


# --------------------------------------------------------------------------- #
# Page 1: Executive Overview & Metrics
# --------------------------------------------------------------------------- #
def render_overview() -> None:
    st.markdown('<h1 class="page-title">Executive Overview &amp; Metrics</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">Project 5: Jio Subscriber Churn Intelligence &amp; Revenue Protection - '
        "a snapshot of the subscriber base, baseline churn risk, and champion model performance.</p>",
        unsafe_allow_html=True,
    )

    eda = load_eda_stats()
    metrics_df = load_comprehensive_metrics()

    if eda is None:
        st.warning("`eda_summary_stats.json` not found. Run `comprehensive_eda.py` to populate dataset context.")
    else:
        ds = eda.get("dataset", {})
        tgt = eda.get("target_analysis", {})
        st.markdown('<div class="section-heading">Dataset Snapshot</div>', unsafe_allow_html=True)
        cols = st.columns(4)
        cards = [
            ("\U0001F465", "Total Subscribers", f"{ds.get('n_rows', 0):,}", ""),
            ("\U0001F4C9", "Baseline 30-Day Churn Rate", f"{tgt.get('churn_rate_pct', 0):.2f}%",
             f"{tgt.get('churn_count', 0):,} churned"),
            ("\U0001F4B0", "Avg ARPU (Last Month)", f"INR {ds.get('avg_arpu_last_month_inr', 0):,.2f}", ""),
            ("\u2696\uFE0F", "Class Imbalance", f"{tgt.get('imbalance_ratio_non_churn_to_churn', 0):.0f} : 1",
             "retained : churned"),
        ]
        for col, (icon, label, value, sub) in zip(cols, cards):
            with col:
                st.markdown(metric_card_html(icon, label, value, sub), unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Predictive Model Performance</div>', unsafe_allow_html=True)
    if metrics_df is None:
        st.info("`comprehensive_model_metrics.csv` not found. Run `train_churn_models.py` to populate model metrics.")
    else:
        champion = metrics_df.sort_values("PR_AUC", ascending=False).iloc[0]
        cols2 = st.columns(4)
        cards2 = [
            ("\U0001F3C6", "Champion Model", str(champion["Model"]), "ranked by PR-AUC"),
            ("\U0001F3AF", "PR-AUC", f"{champion['PR_AUC']:.4f}", "precision-recall AUC"),
            ("\U0001F4C8", "ROC-AUC", f"{champion['ROC_AUC']:.4f}", ""),
            ("\U0001F50E", "Recall @ Top 10% Risk", f"{champion.get('Recall_at_Top_Decile_Pct', float('nan')):.1f}%",
             "of true churners captured"),
        ]
        for col, (icon, label, value, sub) in zip(cols2, cards2):
            with col:
                st.markdown(metric_card_html(icon, label, value, sub), unsafe_allow_html=True)

        st.markdown("##### High-Level Model Summary (All Models)")
        summary_cols = ["Model", "ROC_AUC", "PR_AUC", "Accuracy", "Precision", "Recall"]
        available_cols = [c for c in summary_cols if c in metrics_df.columns]
        summary_df = metrics_df[available_cols].sort_values("PR_AUC", ascending=False).reset_index(drop=True)
        # Format as plain strings rather than df.style.format(...) - the Styler accessor requires
        # jinja2 and silently breaks this whole page (including the charts below) if that optional
        # dependency is missing or outdated in the active environment.
        format_map = {
            "ROC_AUC": "{:.4f}", "PR_AUC": "{:.4f}",
            "Accuracy": "{:.2%}", "Precision": "{:.2%}", "Recall": "{:.2%}",
        }
        display_df = summary_df.copy()
        for col, fmt in format_map.items():
            if col in display_df.columns:
                display_df[col] = display_df[col].map(fmt.format)
        st.dataframe(display_df, width="stretch", hide_index=True)

        with st.expander("View full model comparison table (all columns)"):
            st.dataframe(metrics_df, width="stretch", hide_index=True)

    st.markdown('<div class="section-heading">Key Diagnostic Insights</div>', unsafe_allow_html=True)
    findings = (eda or {}).get("key_findings", [])
    if findings:
        for finding in findings:
            st.markdown(f"- {finding}")
    else:
        st.info("Key findings not available yet.")

    st.markdown('<div class="section-heading">Visual Diagnostics</div>', unsafe_allow_html=True)
    chart_candidates = [
        (FIGURES_DIR / "exec_circle_risk_chart.png", "Geographic Churn Hotspots"),
        (FIGURES_DIR / "exec_recharge_gap_chart.png", "Recharge Silent-Churn Threshold"),
        (FIGURES_DIR / "exec_arpu_band_chart.png", "ARPU Band Analysis"),
        (FIGURES_DIR / "exec_network_kpi_chart.png", "Network KPI Overview"),
        (REPORTS_DIR / "best_model_shap_summary.png", "SHAP Churn Drivers (Champion Model)"),
        (REPORTS_DIR / "confusion_matrix_xgboost.png", "Champion Model Confusion Matrix"),
    ]
    available = [(p, c) for p, c in chart_candidates if p.exists()]
    if not available:
        st.info("No chart assets found yet. Run the EDA / modeling scripts to generate them.")
    else:
        chart_cols = st.columns(2)
        for i, (path, caption) in enumerate(available):
            with chart_cols[i % 2]:
                st.image(str(path), caption=caption, width="stretch")


# --------------------------------------------------------------------------- #
# Page 2: Subscriber Risk Lookup
# --------------------------------------------------------------------------- #
def indicator_card_html(icon: str, label: str, value: str, elevated: bool, sublabel: str = "") -> str:
    """Metric card color-coded red when the indicator itself signals elevated churn risk."""
    accent = "#C0392B" if elevated else "#1B7A43"
    return metric_card_html(icon, label, value, sublabel, accent=accent)


def build_risk_explanation(row: pd.Series, eda: dict[str, Any] | None) -> list[str]:
    """Plain-English, subscriber-specific rationale mirroring the champion model's SHAP drivers
    (MNP enquiry, recharge silence, ARPU decline, competitor usage, unresolved complaints)."""
    numeric_stats = (eda or {}).get("numeric_stats", {})
    avg_recharge_gap = numeric_stats.get("days_since_last_recharge", {}).get("mean")
    avg_arpu = numeric_stats.get("arpu_last_month_inr", {}).get("mean")
    avg_competitor_pct = numeric_stats.get("outgoing_to_competitor_pct", {}).get("mean")

    reasons: list[str] = []

    if bool(row.get("mnp_enquiry_flag")):
        reasons.append(
            "**MNP enquiry on record** - this subscriber has already checked mobile number portability, "
            "the strongest churn signal in our SHAP analysis."
        )

    days_since_recharge = row.get("days_since_last_recharge")
    if pd.notna(days_since_recharge) and avg_recharge_gap:
        if days_since_recharge > avg_recharge_gap * 1.5:
            reasons.append(
                f"**Recharge silence** - {days_since_recharge:.0f} days since last recharge, well above "
                f"the network average of {avg_recharge_gap:.0f} days, a leading 'silent churn' signal."
            )

    arpu_last = row.get("arpu_last_month_inr")
    arpu_3m = row.get("arpu_3m_avg_inr")
    if pd.notna(arpu_last) and pd.notna(arpu_3m) and arpu_last < arpu_3m:
        reasons.append(
            f"**ARPU decline** - last month's spend (INR {arpu_last:.2f}) has dropped below their own "
            f"3-month average (INR {arpu_3m:.2f}), an early revenue-erosion warning."
        )
    elif pd.notna(arpu_last) and avg_arpu and arpu_last < avg_arpu:
        reasons.append(f"**Below-average ARPU** - INR {arpu_last:.2f} vs. the network average of INR {avg_arpu:.2f}.")

    competitor_pct = row.get("outgoing_to_competitor_pct")
    if pd.notna(competitor_pct) and avg_competitor_pct and competitor_pct > avg_competitor_pct * 1.3:
        reasons.append(
            f"**High competitor usage share** - {competitor_pct:.1f}% of outgoing usage already routed to "
            f"competitor networks (network average: {avg_competitor_pct:.1f}%)."
        )

    unresolved = row.get("unresolved_complaints", 0) or 0
    complaints = row.get("complaints_6m", 0) or 0
    if unresolved > 0:
        reasons.append(
            f"**Unresolved complaints** - {int(unresolved)} of {int(complaints)} complaints in the last 6 "
            "months remain unresolved, a direct dissatisfaction signal."
        )

    if not reasons:
        reasons.append(
            "No single dominant risk factor stands out - this subscriber's profile is broadly in line with "
            "the retained population across the key churn drivers we track."
        )
    return reasons


def render_risk_lookup() -> None:
    st.markdown('<h1 class="page-title">Subscriber Risk Lookup</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">Search an individual subscriber to view their profile and a live '
        "churn-risk score.</p>",
        unsafe_allow_html=True,
    )

    df = load_subscribers_data()
    if df is None:
        st.error(f"Subscriber data not available from `{DB_PATH}`. Run `build_database.py` to generate it.")
        return

    eda = load_eda_stats()

    subscriber_id = st.text_input("Enter Subscriber ID", placeholder="e.g. JIO10037241").strip()
    lookup_clicked = st.button("Lookup Subscriber", type="primary")

    if not lookup_clicked:
        st.info("Enter a Subscriber ID above and click **Lookup Subscriber** to view their risk profile.")
        return
    if not subscriber_id:
        st.warning("Please enter a Subscriber ID.")
        return

    matches = df.index[df["subscriber_id"].str.upper() == subscriber_id.upper()]
    if len(matches) == 0:
        st.warning(f"No subscriber found with ID '{subscriber_id}'.")
        return
    row_idx = matches[0]
    row = df.loc[row_idx]

    champion_bundle = load_xgboost_artifacts()
    if champion_bundle is not None:
        model_source = "Evaluated XGBoost Champion Model"
        expected_cols = champion_bundle["numeric_cols"] + champion_bundle["categorical_cols"]
        features_full = engineer_scoring_features(df)
        X_row = features_full.loc[[row_idx], expected_cols]
        X_row_proc = champion_bundle["preprocessor"].transform(X_row)
        risk_proba = float(champion_bundle["model"].predict_proba(X_row_proc)[0, 1])
        high_cutoff = float(champion_bundle.get("decision_threshold", 0.5))
    else:
        model_source = "Lightweight In-App Model (fallback)"
        fallback_bundle = build_risk_model(df)
        X_row = fallback_bundle["features_template"].loc[[row_idx]]
        X_row_proc = fallback_bundle["preprocessor"].transform(X_row)
        risk_proba = float(fallback_bundle["model"].predict_proba(X_row_proc)[0, 1])
        confusion_info = load_confusion_info()
        high_cutoff = float(confusion_info["decision_threshold"]) if confusion_info else 0.5

    med_cutoff = high_cutoff / 2

    if risk_proba >= high_cutoff:
        tier, tier_color = "High Risk", "#C0392B"
    elif risk_proba >= med_cutoff:
        tier, tier_color = "Medium Risk", "#E67E22"
    else:
        tier, tier_color = "Low Risk", "#1B7A43"

    st.markdown('<div class="section-heading">Risk Assessment</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    with cols[0]:
        st.markdown(
            metric_card_html("\U0001F3AF", "Churn Risk Score", f"{risk_proba * 100:.1f}%", "predicted probability"),
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(metric_card_html("\u26A0\uFE0F", "Risk Tier", tier, accent=tier_color), unsafe_allow_html=True)
    with cols[2]:
        actual_label = "Churned" if bool(row[TARGET]) else "Retained"
        st.markdown(
            metric_card_html("\U0001F4CB", "Actual Historical Label", actual_label, "churn_flag_30d (ground truth)"),
            unsafe_allow_html=True,
        )

    st.caption(
        f"Scored using: **{model_source}**. When the persisted champion bundle is unavailable, scores come "
        "from a lightweight in-app model trained for interactive demos instead."
    )

    st.markdown('<div class="section-heading">Key Risk Indicators</div>', unsafe_allow_html=True)
    numeric_stats = (eda or {}).get("numeric_stats", {})
    avg_recharge_gap = numeric_stats.get("days_since_last_recharge", {}).get("mean")
    days_since_recharge = row.get("days_since_last_recharge")
    recharge_elevated = bool(
        avg_recharge_gap and pd.notna(days_since_recharge) and days_since_recharge > avg_recharge_gap * 1.5
    )

    arpu_last = row.get("arpu_last_month_inr")
    arpu_3m = row.get("arpu_3m_avg_inr")
    arpu_elevated = bool(pd.notna(arpu_last) and pd.notna(arpu_3m) and arpu_last < arpu_3m)

    mnp_flag = bool(row.get("mnp_enquiry_flag"))

    indicator_cols = st.columns(3)
    with indicator_cols[0]:
        st.markdown(
            indicator_card_html(
                "\u23F3", "Days Since Last Recharge", f"{days_since_recharge:.0f} days", recharge_elevated,
                f"network avg: {avg_recharge_gap:.0f} days" if avg_recharge_gap else "",
            ),
            unsafe_allow_html=True,
        )
    with indicator_cols[1]:
        st.markdown(
            indicator_card_html(
                "\U0001F4B0", "ARPU Last Month", f"INR {arpu_last:,.2f}", arpu_elevated,
                f"3-month avg: INR {arpu_3m:,.2f}" if pd.notna(arpu_3m) else "",
            ),
            unsafe_allow_html=True,
        )
    with indicator_cols[2]:
        st.markdown(
            indicator_card_html(
                "\U0001F4F1", "MNP Enquiry Status", "Yes - Enquired" if mnp_flag else "No", mnp_flag,
                "actively checking portability" if mnp_flag else "no portability enquiry on record",
            ),
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-heading">Why Is This Subscriber Flagged?</div>', unsafe_allow_html=True)
    for reason in build_risk_explanation(row, eda):
        st.markdown(f"- {reason}")

    st.markdown('<div class="section-heading">Subscriber Profile</div>', unsafe_allow_html=True)
    profile_fields = [
        ("Circle", row.get("circle")), ("Zone", row.get("zone")), ("Plan Type", row.get("plan_type")),
        ("Tenure (months)", row.get("tenure_months")), ("ARPU Last Month (INR)", row.get("arpu_last_month_inr")),
        ("Days Since Last Recharge", row.get("days_since_last_recharge")),
        ("Avg Recharge Gap (days)", row.get("avg_recharge_gap_days")),
        ("Complaints (6m)", row.get("complaints_6m")), ("Unresolved Complaints", row.get("unresolved_complaints")),
        ("Device Brand", row.get("device_brand")), ("Autopay Enabled", row.get("autopay_enabled")),
        ("MNP Enquiry Flag", row.get("mnp_enquiry_flag")),
    ]
    profile_cols = st.columns(3)
    for i, (label, value) in enumerate(profile_fields):
        with profile_cols[i % 3]:
            st.markdown(metric_card_html("", label, str(value)), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Page 3: Conversational Churn Assistant (rule-based, strict data guardrails)
# --------------------------------------------------------------------------- #
SAFETY_MESSAGE = (
    "I am restricted to answering questions related to Jio subscriber churn intelligence and "
    "retention metrics. Data not available for out-of-scope queries."
)

# Topic allowlist: churn, ARPU/revenue, recharge behavior, MNP/portability, retention, and
# model metrics/drivers - anything not touching these topics is rejected before any data lookup.
ALLOWED_TOPIC_KEYWORDS = [
    "churn", "subscriber", "customer", "retention", "retain", "attrition",
    "arpu", "average revenue per user", "revenue", "plan type", "plan price", "price bracket", "price tier",
    "prepaid", "postpaid",
    "recharge", "gap", "silent", "tenure", "autopay", "payment",
    "mnp", "portability", "port out", "port-out", "switch operator", "switch network",
    "circle", "zone", "region", "device brand", "brand", "complaint", "network quality",
    "sinr", "drop call", "congestion", "data usage", "voice minutes", "sms count", "5g",
    "roaming", "family plan", "app login", "competitor", "offer redemption",
    "model", "roc-auc", "roc auc", "pr-auc", "pr auc", "precision", "recall",
    "accuracy", "threshold", "confusion matrix", "shap", "decile",
    "risk", "driver", "factor", "finding", "insight", "champion model",
    "xgboost", "logistic regression", "random forest", "metric", "performance",
]


def is_in_scope(
    query: str,
    dims: dict[str, list[str]] | None = None,
    conn: sqlite3.Connection | None = None,
    schema: dict[str, dict[str, str]] | None = None,
) -> bool:
    """Strict guardrail: reject anything that doesn't touch an allowed churn/retention topic.
    Also accepts queries that resolve to a real circle/zone/plan/device or risk-tier filter, or
    to a real column/table in the SQLite database, so specific questions aren't rejected purely
    for lacking a hardcoded keyword."""
    q = query.lower()
    if any(keyword in q for keyword in ALLOWED_TOPIC_KEYWORDS):
        return True
    if dims is not None and resolve_category_sql_filter(query, dims) is not None:
        return True
    if resolve_risk_tier_query(q) is not None:
        return True
    if conn is not None and schema and resolve_row_sample_query_sql(query, conn, schema, dims) is not None:
        return True
    if conn is not None and schema and resolve_generic_stat_query_sql(query, conn, schema) is not None:
        return True
    return False


def answer_query(
    query: str,
    eda: dict[str, Any] | None,
    metrics_df: pd.DataFrame | None,
    df: pd.DataFrame | None,
    conn: sqlite3.Connection | None = None,
    dims: dict[str, list[str]] | None = None,
    schema: dict[str, dict[str, str]] | None = None,
) -> str | tuple[str, pd.DataFrame]:
    """Answer strictly from project data (fetched live from jio_retention.db wherever possible);
    anything unsupported returns 'Data not available'. List/breakdown-style answers return
    (summary_text, DataFrame) so the UI can render them as a table."""
    q = query.lower()
    fallback = (
        "Data not available for this question. I can currently answer questions about: overall churn rate, "
        "subscriber counts by circle/zone/plan type/device brand, risk tier counts (high/medium/low/top "
        "decile), churn rate by segment, recorded reasons for churn, top churn drivers, model performance, "
        "key findings, sample/raw records (e.g. 'show subscribers in Gujarat'), or any column/aggregate "
        "(average/total/max/min) across any table in the database (subscribers, service requests, network "
        "sites, monthly circle KPIs, circle targets, offer catalogue)."
    )

    if conn is not None and schema:
        row_sample = resolve_row_sample_query_sql(query, conn, schema, dims)
        if row_sample is not None:
            return row_sample

    if df is not None:
        geo_tier = resolve_risk_tier_by_circle_query(query, df)
        if geo_tier is not None:
            return geo_tier

    if conn is not None and dims is not None:
        if any(
            k in q for k in ["how many subscriber", "number of subscriber", "total subscriber", "subscriber count"]
        ):
            match = resolve_category_sql_filter(query, dims)
            if match is not None:
                where_clause, params, label = match
                return format_count_response_sql(conn, where_clause, params, label)
            total = sql_scalar(conn, "SELECT COUNT(*) FROM subscribers")
            return f"There are **{total:,}** subscribers in the dataset."

        if any(k in q for k in ["how many", "count", "number of"]):
            match = resolve_category_sql_filter(query, dims)
            if match is not None:
                where_clause, params, label = match
                return format_count_response_sql(conn, where_clause, params, label)

    if df is not None:
        tier_key = resolve_risk_tier_query(q)
        if tier_key is not None:
            return format_risk_tier_response(tier_key, df)

    if eda is not None:
        tgt = eda.get("target_analysis", {})

        if any(k in q for k in ["overall churn", "baseline churn", "total churn rate", "churn rate overall"]):
            return (
                f"The baseline 30-day churn rate is **{tgt.get('churn_rate_pct', 0):.2f}%** "
                f"({tgt.get('churn_count', 0):,} churned out of {tgt.get('total_subscribers', 0):,} subscribers)."
            )

        segment_rates = tgt.get("segment_churn_rates", {})
        for segment_col, rates in segment_rates.items():
            for category, info in rates.items():
                if category.lower() in q:
                    return (
                        f"The 30-day churn rate for **{category}** ({segment_col}) is "
                        f"**{info['churn_rate_pct']:.2f}%** (n={info['subscriber_count']:,})."
                    )

        if "reason" in q and "churn" in q:
            churn_reason_stats = eda.get("categorical_stats", {}).get("churn_reason", {})
            reason_counts = churn_reason_stats.get("top_10_value_counts", {})
            if reason_counts:
                lines = [f"- **{reason}**: {count:,} churned subscribers" for reason, count in reason_counts.items()]
                return (
                    "Recorded reasons for churn (among subscribers who churned and stated a reason):\n"
                    + "\n".join(lines)
                )
            return "Data not available: churn_reason breakdown has not been generated yet."

        if any(k in q for k in ["driver", "factor", "cause", "why do", "why are", "risk signal"]):
            top_corr = tgt.get("top_numeric_correlations_with_churn", {})
            if top_corr:
                lines = [f"- **{feat}**: r = {val:.3f}" for feat, val in list(top_corr.items())[:5]]
                return "Top churn drivers by correlation with 30-day churn:\n" + "\n".join(lines)
            return "Data not available: churn-driver correlations have not been generated yet."

        if any(k in q for k in ["finding", "insight", "summary", "takeaway"]):
            findings = eda.get("key_findings", [])
            if findings:
                return "Key diagnostic findings:\n" + "\n".join(f"- {f}" for f in findings)
            return "Data not available: key findings have not been generated yet."

    if any(k in q for k in ["model", "accuracy", "auc", "performance", "champion", "precision", "recall"]):
        if metrics_df is None:
            return "Data not available: model metrics have not been generated yet. Run `train_churn_models.py` first."
        best = metrics_df.sort_values("PR_AUC", ascending=False).iloc[0]
        return (
            f"The champion model is **{best['Model']}** with PR-AUC={best['PR_AUC']:.4f}, "
            f"ROC-AUC={best['ROC_AUC']:.4f}, and Recall@Top-Decile="
            f"{best.get('Recall_at_Top_Decile_Pct', float('nan')):.1f}%."
        )

    if conn is not None and schema:
        generic_answer = resolve_generic_stat_query_sql(query, conn, schema)
        if generic_answer:
            return generic_answer

    return fallback


def render_chat_assistant() -> None:
    st.markdown('<h1 class="page-title">Conversational Churn Assistant</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">Ask about churn rates, drivers, or model performance. Answers are fetched '
        "live via SQL from the project database - anything outside that scope receives a clear "
        '"Data not available" response.</p>',
        unsafe_allow_html=True,
    )

    eda = load_eda_stats()
    metrics_df = load_comprehensive_metrics()
    df = load_subscribers_data()
    conn = get_db_connection()
    dims = get_subscriber_dimensions(conn) if conn is not None else None
    schema = get_db_schema(conn) if conn is not None else None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": (
                    "Hi! Ask me about subscriber counts (e.g. 'how many subscribers in Gujarat'), "
                    "risk tiers (e.g. 'how many are high risk'), churn rate by circle/zone/plan type, "
                    "overall churn rate, top churn drivers, model performance, sample/raw records "
                    "(e.g. 'show subscribers in Gujarat'), or any column/aggregate across any table in "
                    "the database (subscribers, service requests, network sites, monthly circle KPIs, "
                    "circle targets, offer catalogue)."
                ),
                "table": None,
            }
        ]

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("table") is not None:
                st.dataframe(msg["table"], width="stretch", hide_index=True)

    user_query = st.chat_input("Ask a question about subscriber churn...")
    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query, "table": None})
        if not is_in_scope(user_query, dims, conn, schema):
            answer_text, answer_table = SAFETY_MESSAGE, None
        else:
            result = answer_query(user_query, eda, metrics_df, df, conn, dims, schema)
            answer_text, answer_table = result if isinstance(result, tuple) else (result, None)
        st.session_state.chat_history.append({"role": "assistant", "content": answer_text, "table": answer_table})
        st.rerun()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    configure_page()
    inject_custom_css()

    with st.sidebar:
        st.markdown('<div class="sidebar-brand">JIO</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-subtitle">Churn Intelligence &amp; Revenue Protection</div>',
                     unsafe_allow_html=True)
        st.markdown("---")
        page = st.radio(
            "Navigation",
            ["Executive Overview & Metrics", "Subscriber Risk Lookup", "Conversational Churn Assistant"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.caption("Project 5 \u00b7 Revenue Protection & Retention Analytics")

    try:
        if page == "Executive Overview & Metrics":
            render_overview()
        elif page == "Subscriber Risk Lookup":
            render_risk_lookup()
        else:
            render_chat_assistant()
    except Exception as e:  # noqa: BLE001 - surface unexpected errors in the UI instead of crashing silently
        st.error(f"An unexpected error occurred while rendering this page: {e}")


if __name__ == "__main__":
    main()
