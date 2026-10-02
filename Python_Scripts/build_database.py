"""
build_database.py
Project 5: Jio Subscriber Churn Intelligence & Revenue Protection

Builds a local SQLite database (Data/jio_retention.db) from subscribers.csv and every
sheet in Jio_Retention_Dataset.xlsx, so the Streamlit app's Conversational Churn
Assistant can answer questions by running real SQL queries against the tables
instead of filtering in-memory DataFrames.

Run with:
    python build_database.py
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"
SUBSCRIBERS_CSV = DATA_DIR / "subscribers.csv"
DATASET_XLSX = DATA_DIR / "Jio_Retention_Dataset.xlsx"
DB_PATH = DATA_DIR / "jio_retention.db"


def load_dataframes() -> dict[str, pd.DataFrame]:
    """subscribers.csv plus every sheet in Jio_Retention_Dataset.xlsx, keyed by table name."""
    dataframes: dict[str, pd.DataFrame] = {"subscribers": pd.read_csv(SUBSCRIBERS_CSV)}
    xlsx_sheets = pd.read_excel(DATASET_XLSX, sheet_name=None)
    for name, sheet_df in xlsx_sheets.items():
        if name.lower() == "subscribers":
            continue  # duplicate of subscribers.csv, already loaded above
        dataframes[name.lower()] = sheet_df
    return dataframes


def build_database() -> None:
    dataframes = load_dataframes()
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Fresh build every run so the DB always mirrors the current source files.
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    try:
        for table_name, df in dataframes.items():
            df.to_sql(table_name, conn, index=False, if_exists="replace")
            print(f"Wrote table '{table_name}': {df.shape[0]:,} rows x {df.shape[1]} columns")

        if "subscribers" in dataframes and "subscriber_id" in dataframes["subscribers"].columns:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_subscribers_id ON subscribers (subscriber_id)")
        conn.commit()
    finally:
        conn.close()

    print(f"\nDatabase written to {DB_PATH}")


if __name__ == "__main__":
    build_database()
