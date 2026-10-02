"""Load subscribers.csv and all sheets of Jio_Retention_Dataset.xlsx into a dict of DataFrames."""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
SUBSCRIBERS_CSV = DATA_DIR / "subscribers.csv"
DATASET_XLSX = DATA_DIR / "Jio_Retention_Dataset.xlsx"


def load_dataframes() -> dict[str, pd.DataFrame]:
    """Return a dict mapping name -> DataFrame for subscribers.csv and every sheet in the xlsx file."""
    dataframes: dict[str, pd.DataFrame] = {"subscribers": pd.read_csv(SUBSCRIBERS_CSV)}
    xlsx_sheets = pd.read_excel(DATASET_XLSX, sheet_name=None)
    dataframes.update(xlsx_sheets)
    return dataframes


if __name__ == "__main__":
    dfs = load_dataframes()
    for name, df in dfs.items():
        print(f"{name}: {df.shape}")
