"""
train_churn_models.py
Project 5: Jio Subscriber Churn Intelligence & Revenue Protection
Workstream 2 - Predictive Modeling: Baseline vs. Champion vs. Alternative

Purpose
-------
Trains and compares three churn-prediction models on subscribers.csv:
  1. Baseline    - Logistic Regression (class_weight="balanced")
  2. Champion    - XGBoost Classifier, Optuna-tuned via 5-fold CV (PR-AUC objective)
                   with early stopping on a true validation set
  3. Alternative - Random Forest Classifier (class_weight="balanced")

Pipeline
--------
  1. Load data, drop leakage/identifier columns, engineer SHAP-informed derived
     features (row-wise only, so no fitting/leakage risk), build numeric +
     categorical feature buckets, and perform a stratified 70/15/15
     train/validation/test split.
  2. Fit a ColumnTransformer (StandardScaler for numerics, OneHotEncoder for
     categoricals) on the training split only; validation/test are transform-only.
  3. Train all three models on the training split, monitoring each on the
     validation split. XGBoost's hyperparameters (learning_rate, max_depth,
     subsample, colsample_bytree, min_child_weight) are tuned with Optuna using
     5-fold stratified CV on the training split (objective = mean PR-AUC),
     with every candidate fit early-stopped against the true validation split.
     The final champion is retrained on the full training split with the tuned
     hyperparameters, early-stopped on the same validation split.
  4. Evaluate all three models on the untouched test split: ROC-AUC, PR-AUC,
     and Recall at Top Decile are threshold-independent ranking metrics.
     Accuracy/Precision/Recall/confusion-matrix figures use each model's own
     validation-optimized decision threshold (F1-maximizing point on the
     validation PR curve) instead of the default 0.5, which is a poor choice
     at a ~1.7% base churn rate.
  5. Save comparative metrics tables, a confusion matrix, and a SHAP summary
     plot for the tuned champion (XGBoost) model.

Outputs (relative to project root)
-----------------------------------
  Reports/model_comparison_metrics.csv
  Reports/comprehensive_model_metrics.csv
  Reports/confusion_matrix_xgboost.png
  Reports/confusion_matrix_xgboost.json
  Reports/best_model_shap_summary.png
  Reports/xgboost_champion_bundle.joblib
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # non-interactive backend

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# --------------------------------------------------------------------------- #
# Path management
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "Data" / "subscribers.csv"
REPORTS_DIR = BASE_DIR / "Reports"
METRICS_CSV_PATH = REPORTS_DIR / "model_comparison_metrics.csv"
COMPREHENSIVE_METRICS_CSV_PATH = REPORTS_DIR / "comprehensive_model_metrics.csv"
SHAP_PNG_PATH = REPORTS_DIR / "best_model_shap_summary.png"
CONFUSION_MATRIX_PNG_PATH = REPORTS_DIR / "confusion_matrix_xgboost.png"
CONFUSION_MATRIX_JSON_PATH = REPORTS_DIR / "confusion_matrix_xgboost.json"
CHAMPION_BUNDLE_PATH = REPORTS_DIR / "xgboost_champion_bundle.joblib"

TARGET = "churn_flag_30d"
RANDOM_STATE = 42
TOP_DECILE = 0.10
OVERFIT_ROC_AUC_GAP_THRESHOLD = 0.05  # flag a model if Train ROC-AUC exceeds Val ROC-AUC by more than this

# Optuna / cross-validation search budget for the XGBoost champion (tune down for faster reruns).
N_OPTUNA_TRIALS = 25
CV_FOLDS = 5
CV_EARLY_STOPPING_ROUNDS = 30
FINAL_EARLY_STOPPING_ROUNDS = 50

# Columns excluded from modeling: identifier, and post-outcome fields that leak
# information about the target (a different churn horizon, churn reason/date).
EXCLUDE_COLS = ["subscriber_id", "churn_flag_90d", "churn_reason", "churn_date", "join_date"]

ENGINEERED_FEATURE_NAMES = [
    "arpu_ratio_1m_to_3m",
    "arpu_decline_flag",
    "recharge_overdue_ratio",
    "long_recharge_gap_flag",
    "network_complaint_intensity",
]


# --------------------------------------------------------------------------- #
# 1. Data preparation, feature engineering & 3-way stratified split
# --------------------------------------------------------------------------- #
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find subscribers.csv at expected location: {DATA_PATH}\n"
            "Ensure the Data/subscribers.csv file exists relative to the project root."
        )
    return pd.read_csv(DATA_PATH)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """SHAP-informed derived features. Each value depends only on its own row (no cross-row
    statistics), so these are safe to compute before the split - there is nothing to "fit"
    and therefore no leakage risk, unlike scaling/imputation which must be fit on training data only."""
    out = df.copy()

    # ARPU trend: falling 1-month ARPU relative to the 3-month average flags early revenue decline.
    out["arpu_ratio_1m_to_3m"] = (
        out["arpu_last_month_inr"] / out["arpu_3m_avg_inr"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    out["arpu_decline_flag"] = (out["arpu_last_month_inr"] < out["arpu_3m_avg_inr"]).astype(int)

    # Recharge overdue: how far past a subscriber's own typical recharge cadence they currently are.
    safe_gap = out["avg_recharge_gap_days"].replace(0, np.nan)
    out["recharge_overdue_ratio"] = (out["days_since_last_recharge"] / safe_gap).replace(
        [np.inf, -np.inf], np.nan
    )
    out["long_recharge_gap_flag"] = (out["recharge_overdue_ratio"] > 1.5).astype(int)

    # Combined complaint severity: unresolved complaints weighted higher than resolved ones.
    out["network_complaint_intensity"] = out["complaints_6m"] + 2 * out["unresolved_complaints"]

    return out


def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Drop leakage/identifier columns, engineer derived features, and split the remaining
    features into numeric/categorical buckets."""
    drop_cols = [c for c in EXCLUDE_COLS + [TARGET] if c in df.columns]
    features = df.drop(columns=drop_cols).copy()
    features = engineer_features(features)

    boolean_cols = [c for c in features.columns if pd.api.types.is_bool_dtype(features[c])]
    features[boolean_cols] = features[boolean_cols].astype(int)  # booleans as 0/1 numeric features

    numeric_cols = [
        c for c in features.columns
        if pd.api.types.is_numeric_dtype(features[c]) and c not in boolean_cols
    ] + boolean_cols
    categorical_cols = [c for c in features.columns if c not in numeric_cols]

    return features, numeric_cols, categorical_cols


def stratified_three_way_split(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """70% train / 15% validation / 15% test, stratified on the target."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])


# --------------------------------------------------------------------------- #
# 2. Model training (train on 70%, monitor on 15% validation)
# --------------------------------------------------------------------------- #
def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def tune_xgboost_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = N_OPTUNA_TRIALS,
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> tuple[dict[str, Any], float]:
    """Optuna search over key XGBoost hyperparameters, scored by 5-fold stratified CV PR-AUC
    on the training split. Every candidate fit is early-stopped against the true validation
    split (never the CV fold itself), matching how the final champion model will be trained."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }
        fold_scores = []
        for fold_train_idx, fold_holdout_idx in skf.split(X_train, y_train):
            X_fold_train, X_fold_holdout = X_train[fold_train_idx], X_train[fold_holdout_idx]
            y_fold_train, y_fold_holdout = y_train[fold_train_idx], y_train[fold_holdout_idx]
            neg, pos = np.bincount(y_fold_train)

            model = XGBClassifier(
                n_estimators=500,
                eval_metric="aucpr",
                early_stopping_rounds=CV_EARLY_STOPPING_ROUNDS,
                scale_pos_weight=neg / pos,
                random_state=random_state,
                n_jobs=-1,
                **params,
            )
            model.fit(X_fold_train, y_fold_train, eval_set=[(X_val, y_val)], verbose=False)
            proba = model.predict_proba(X_fold_holdout)[:, 1]
            fold_scores.append(average_precision_score(y_fold_holdout, proba))
        return float(np.mean(fold_scores))

    def progress_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if (trial.number + 1) % 5 == 0 or trial.number == n_trials - 1:
            print(f"   Optuna trial {trial.number + 1}/{n_trials} | "
                  f"CV PR-AUC={trial.value:.4f} | best so far={study.best_value:.4f}")

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, callbacks=[progress_callback], show_progress_bar=False)

    return study.best_params, study.best_value


def train_xgboost_final(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    best_params: dict[str, Any],
) -> XGBClassifier:
    """Retrain on the full training split with Optuna-tuned hyperparameters, early-stopped
    on the true validation split (kept fully separate from the CV search above)."""
    neg, pos = np.bincount(y_train)
    model = XGBClassifier(
        n_estimators=500,
        scale_pos_weight=neg / pos,  # up-weight the rare churn class
        eval_metric="aucpr",
        early_stopping_rounds=FINAL_EARLY_STOPPING_ROUNDS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **best_params,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


# --------------------------------------------------------------------------- #
# 3. Evaluation: ROC-AUC, PR-AUC, Recall at Top Decile, optimized decision threshold
# --------------------------------------------------------------------------- #
def recall_at_top_decile(y_true: np.ndarray, y_scores: np.ndarray, decile: float = TOP_DECILE) -> float:
    """Percentage of true churners captured within the top `decile` highest-risk predictions."""
    n = len(y_true)
    top_k = max(1, int(np.ceil(n * decile)))
    top_idx = np.argsort(y_scores)[::-1][:top_k]
    total_churners = y_true.sum()
    if total_churners == 0:
        return np.nan
    captured = y_true[top_idx].sum()
    return float(captured / total_churners * 100)


def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, float]:
    """F1-maximizing point on the validation precision-recall curve. The default 0.5 threshold
    is a poor choice at a ~1.7% base churn rate - it tends to starve recall or precision depending
    on the model's calibration, so the optimal operating point is learned from data instead."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    if len(thresholds) == 0:
        return 0.5, 0.0
    denom = precision[:-1] + recall[:-1]
    f1_scores = np.divide(
        2 * precision[:-1] * recall[:-1], denom, out=np.zeros_like(denom), where=denom > 0
    )
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def evaluate_model(name: str, model: Any, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Threshold-independent ranking metrics (ROC-AUC, PR-AUC, Recall at Top Decile)."""
    proba = model.predict_proba(X)[:, 1]
    return {
        "model": name,
        "roc_auc": round(roc_auc_score(y, proba), 4),
        "pr_auc": round(average_precision_score(y, proba), 4),
        "recall_at_top_decile_pct": round(recall_at_top_decile(y, proba), 2),
    }


def diagnose_overfitting(
    name: str,
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
    threshold: float = OVERFIT_ROC_AUC_GAP_THRESHOLD,
) -> dict[str, Any]:
    """Print train-vs-validation metrics side by side and flag a large ROC-AUC gap as overfitting."""
    roc_gap = train_metrics["roc_auc"] - val_metrics["roc_auc"]
    pr_gap = train_metrics["pr_auc"] - val_metrics["pr_auc"]
    is_overfitting = roc_gap > threshold

    print(f"{name}")
    print(f"    ROC-AUC: Train={train_metrics['roc_auc']:.4f} | Val={val_metrics['roc_auc']:.4f} | Gap={roc_gap:+.4f}")
    print(f"    PR-AUC : Train={train_metrics['pr_auc']:.4f} | Val={val_metrics['pr_auc']:.4f} | Gap={pr_gap:+.4f}")
    flag_msg = (
        f"OVERFITTING DETECTED (ROC-AUC gap {roc_gap:.4f} > {threshold})"
        if is_overfitting
        else f"OK (ROC-AUC gap {roc_gap:.4f} <= {threshold})"
    )
    print(f"    Overfitting check: {flag_msg}")

    return {
        "model": name,
        "train_roc_auc": train_metrics["roc_auc"],
        "val_roc_auc": val_metrics["roc_auc"],
        "roc_auc_gap": round(roc_gap, 4),
        "train_pr_auc": train_metrics["pr_auc"],
        "val_pr_auc": val_metrics["pr_auc"],
        "pr_auc_gap": round(pr_gap, 4),
        "overfitting_flag": is_overfitting,
    }


def evaluate_comprehensive(
    name: str,
    model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    overfitting_gap: float,
) -> dict[str, Any]:
    """Full test-set scorecard using each model's own validation-optimized decision threshold
    (rather than the default 0.5) for Accuracy/Precision/Recall, plus the threshold-independent
    ROC-AUC/PR-AUC/Recall-at-Top-Decile and the train-vs-validation overfitting gap."""
    proba_val = model.predict_proba(X_val)[:, 1]
    threshold, _ = find_optimal_threshold(y_val, proba_val)

    proba_test = model.predict_proba(X_test)[:, 1]
    preds_test = (proba_test >= threshold).astype(int)

    return {
        "Model": name,
        "ROC_AUC": round(roc_auc_score(y_test, proba_test), 4),
        "PR_AUC": round(average_precision_score(y_test, proba_test), 4),
        "Accuracy": round(accuracy_score(y_test, preds_test), 4),
        "Precision": round(precision_score(y_test, preds_test, zero_division=0), 4),
        "Recall": round(recall_score(y_test, preds_test, zero_division=0), 4),
        "Overfitting_Gap": round(overfitting_gap, 4),
        "Recall_at_Top_Decile_Pct": round(recall_at_top_decile(y_test, proba_test), 2),
        "Optimal_Threshold": round(threshold, 4),
    }


# --------------------------------------------------------------------------- #
# 4. SHAP explainability for the champion model (XGBoost)
# --------------------------------------------------------------------------- #
def generate_shap_summary(model: XGBClassifier, X_test: np.ndarray, feature_names: list[str]) -> None:
    sample_size = min(2000, X_test.shape[0])
    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(X_test.shape[0], size=sample_size, replace=False)
    X_sample = X_test[sample_idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)

    shap.summary_plot(shap_values.values, X_sample, feature_names=feature_names, show=False)
    fig = plt.gcf()
    fig.suptitle("SHAP Summary - XGBoost Champion Model (Top Churn Risk Drivers)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(SHAP_PNG_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    model: Any, X_test: np.ndarray, y_test: np.ndarray, threshold: float, path: Path
) -> np.ndarray:
    """Save a labeled confusion-matrix heatmap using the validation-optimized decision threshold."""
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, preds)

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    sns.heatmap(
        cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax,
        xticklabels=["Retained", "Churned"], yticklabels=["Retained", "Churned"],
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title(
        f"XGBoost Champion Model - Test Set Confusion Matrix (threshold={threshold:.3f})",
        fontsize=11, weight="bold",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return cm


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    try:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        print("=" * 70)
        print("1. DATA PREPARATION, FEATURE ENGINEERING & 3-WAY STRATIFIED SPLIT")
        print("=" * 70)
        df = load_data()
        y = df[TARGET].astype(int)
        X, numeric_cols, categorical_cols = build_feature_frame(df)
        engineered_present = [c for c in ENGINEERED_FEATURE_NAMES if c in numeric_cols]
        print(f"Loaded {len(df):,} rows | {len(numeric_cols)} numeric (incl. {len(engineered_present)} "
              f"engineered: {', '.join(engineered_present)}) + {len(categorical_cols)} categorical "
              f"features (target: {TARGET})")

        X_train, X_val, X_test, y_train, y_val, y_test = stratified_three_way_split(X, y)
        y_train, y_val, y_test = y_train.to_numpy(), y_val.to_numpy(), y_test.to_numpy()
        print(f"Train: {len(X_train):,} ({y_train.mean() * 100:.2f}% churn) | "
              f"Validation: {len(X_val):,} ({y_val.mean() * 100:.2f}% churn) | "
              f"Test: {len(X_test):,} ({y_test.mean() * 100:.2f}% churn)")

        preprocessor = build_preprocessor(numeric_cols, categorical_cols)
        X_train_proc = preprocessor.fit_transform(X_train)
        X_val_proc = preprocessor.transform(X_val)
        X_test_proc = preprocessor.transform(X_test)
        feature_names = list(preprocessor.get_feature_names_out())
        print(f"Preprocessed feature matrix width: {len(feature_names)} columns "
              "(StandardScaler + OneHotEncoder, fit on training split only)")

        print("\n" + "=" * 70)
        print("2. MODEL TRAINING (70% TRAIN, MONITORED ON 15% VALIDATION)")
        print("=" * 70)

        print("Training Model 1/3 - Logistic Regression (baseline)...")
        log_reg = train_logistic_regression(X_train_proc, y_train)

        print(f"Tuning Model 2/3 - XGBoost hyperparameters via Optuna "
              f"({N_OPTUNA_TRIALS} trials, {CV_FOLDS}-fold CV, optimizing PR-AUC)...")
        best_params, best_cv_pr_auc = tune_xgboost_hyperparameters(X_train_proc, y_train, X_val_proc, y_val)
        print(f"   Best CV PR-AUC: {best_cv_pr_auc:.4f} | Best params: {best_params}")

        print("Training Model 2/3 - XGBoost (champion, tuned params, early stopping on validation)...")
        xgb_model = train_xgboost_final(X_train_proc, y_train, X_val_proc, y_val, best_params)
        print(f"   Best iteration: {xgb_model.best_iteration} | "
              f"Best validation PR-AUC (aucpr): {xgb_model.best_score:.4f}")

        print("Training Model 3/3 - Random Forest (alternative)...")
        rf_model = train_random_forest(X_train_proc, y_train)

        models = {
            "Logistic Regression (Baseline)": log_reg,
            "XGBoost (Champion)": xgb_model,
            "Random Forest (Alternative)": rf_model,
        }

        print("\n" + "-" * 70)
        print("PER-MODEL OVERFITTING DIAGNOSTICS (70% Train vs. 15% Validation)")
        print("-" * 70)
        overfit_summary = []
        for name, model in models.items():
            train_metrics = evaluate_model(name, model, X_train_proc, y_train)
            val_metrics = evaluate_model(name, model, X_val_proc, y_val)
            overfit_summary.append(diagnose_overfitting(name, train_metrics, val_metrics))
        print("-" * 70)
        flagged = [d["model"] for d in overfit_summary if d["overfitting_flag"]]
        if flagged:
            print(f"Models flagged for overfitting: {', '.join(flagged)}")
        else:
            print("No models flagged for overfitting (all within ROC-AUC gap threshold).")

        print("\n" + "=" * 70)
        print("3. EVALUATION ON UNSEEN TEST SET (15%) - THRESHOLD-INDEPENDENT METRICS")
        print("=" * 70)
        test_results = [evaluate_model(name, model, X_test_proc, y_test) for name, model in models.items()]
        results_df = pd.DataFrame(test_results).sort_values("pr_auc", ascending=False).reset_index(drop=True)
        results_df = results_df.rename(columns={
            "model": "Model",
            "roc_auc": "ROC_AUC",
            "pr_auc": "PR_AUC",
            "recall_at_top_decile_pct": "Recall_at_Top_Decile_Pct",
        })
        print(results_df.to_string(index=False))

        results_df.to_csv(METRICS_CSV_PATH, index=False)
        print(f"\nComparison table saved to: {METRICS_CSV_PATH}")

        print("\n" + "-" * 70)
        print("3B. COMPREHENSIVE METRICS (VALIDATION-OPTIMIZED DECISION THRESHOLDS)")
        print("-" * 70)
        overfit_gap_by_model = {d["model"]: d["roc_auc_gap"] for d in overfit_summary}
        comprehensive_results = [
            evaluate_comprehensive(
                name, model, X_val_proc, y_val, X_test_proc, y_test, overfit_gap_by_model[name]
            )
            for name, model in models.items()
        ]
        comprehensive_df = (
            pd.DataFrame(comprehensive_results).sort_values("PR_AUC", ascending=False).reset_index(drop=True)
        )
        print(comprehensive_df.to_string(index=False))
        comprehensive_df.to_csv(COMPREHENSIVE_METRICS_CSV_PATH, index=False)
        print(f"\nComprehensive metrics table saved to: {COMPREHENSIVE_METRICS_CSV_PATH}")

        print("\n" + "-" * 70)
        print("3C. CONFUSION MATRIX (XGBoost Champion Model, Test Set, Optimized Threshold)")
        print("-" * 70)
        xgb_threshold = next(
            r["Optimal_Threshold"] for r in comprehensive_results if r["Model"] == "XGBoost (Champion)"
        )
        cm = plot_confusion_matrix(xgb_model, X_test_proc, y_test, xgb_threshold, CONFUSION_MATRIX_PNG_PATH)
        tn, fp, fn, tp = cm.ravel()
        print(f"Decision threshold: {xgb_threshold:.4f}")
        print(f"True Negatives={tn:,} | False Positives={fp:,} | False Negatives={fn:,} | True Positives={tp:,}")
        print(f"Confusion matrix plot saved to: {CONFUSION_MATRIX_PNG_PATH}")

        with open(CONFUSION_MATRIX_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "true_negatives": int(tn), "false_positives": int(fp),
                "false_negatives": int(fn), "true_positives": int(tp),
                "decision_threshold": round(float(xgb_threshold), 4),
            }, f, indent=2)
        print(f"Confusion matrix counts saved to: {CONFUSION_MATRIX_JSON_PATH}")

        print("\n" + "=" * 70)
        print("4. SHAP EXPLAINABILITY (Tuned XGBoost Champion Model)")
        print("=" * 70)
        generate_shap_summary(xgb_model, X_test_proc, feature_names)
        print(f"SHAP summary plot saved to: {SHAP_PNG_PATH}")

        print("\n" + "-" * 70)
        print("5. PERSISTING CHAMPION MODEL BUNDLE (for app.py reuse)")
        print("-" * 70)
        joblib.dump({
            "model": xgb_model,
            "preprocessor": preprocessor,
            "feature_names": feature_names,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "decision_threshold": xgb_threshold,
        }, CHAMPION_BUNDLE_PATH)
        print(f"Champion model bundle saved to: {CHAMPION_BUNDLE_PATH}")

        print("\nModel training and evaluation complete.")

    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception:
        print("\n[ERROR] An unexpected error occurred during model training/evaluation:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
