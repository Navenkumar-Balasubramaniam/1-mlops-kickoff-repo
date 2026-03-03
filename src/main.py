"""
Educational Goal:
- Why this module exists in an MLOps system: Provide a single command entry point
  (python -m src.main) that runs the pipeline end-to-end reliably.
- Responsibility (separation of concerns): Orchestrates steps; does not contain
  the detailed logic of cleaning/features/training/evaluation.
- Pipeline contract (inputs and outputs): Produces three artifacts:
  - data/processed/clean.csv
  - models/model.joblib
  - reports/predictions.csv

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.clean_data import clean_dataframe
from src.evaluate import evaluate_model
from src.features import get_feature_preprocessor
from src.infer import run_inference
from src.load_data import load_raw_data
from src.train import train_model
from src.utils import save_csv, save_model
from src.validate import validate_dataframe


# -----------------------------------------------------------------------------
# CONFIGURATION (SETTINGS dictionary bridge)
# Loud reminder: map these to your dataset columns (we aligned to your notebook)
# -----------------------------------------------------------------------------
SETTINGS = {
    "is_example_config": False,
    "problem_type": "regression",
    "target_column": "milliseconds",
    "year_column": "year",
    "test_year": 2023,
    "paths": {
        "raw_data_path": "data/raw/f1_all.parquet",
        "processed_data_path": "data/processed/clean.csv",
        "model_path": "models/model.joblib",
        "predictions_path": "reports/predictions.csv",
    },
    "features": {
        # In our StandardScaler-only implementation, we treat both lists as numeric-to-scale.
        "quantile_bin": [],
        "numeric_passthrough": [
            "lap",
            "grid",
            "Stint",
            "TyreLife",
            "TrackTemp",
            "Humidity",
            "Pressure",
            "Rainfall",
            "WindSpeed",
            "WindDirection",
        ],
        "categorical_onehot": [
            "round",
            "name",
            "constructorId",
            "code",
            "Compound",
            "FreshTyre",
        ],
        "n_bins": 3,
    },
}


def main() -> None:
    """
    Inputs:
    - None (configuration is inside SETTINGS for now)
    Outputs:
    - None (creates artifacts on disk)
    Why this contract matters for reliable ML delivery:
    - A single script entry point makes runs reproducible and CI-friendly.
    """
    print("[main.main] Starting end-to-end pipeline...")  # TODO: replace with logging later

    # Ensure required directories exist (idempotent)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)

    # Paths
    raw_path = Path(SETTINGS["paths"]["raw_data_path"])
    processed_path = Path(SETTINGS["paths"]["processed_data_path"])
    model_path = Path(SETTINGS["paths"]["model_path"])
    predictions_path = Path(SETTINGS["paths"]["predictions_path"])

    target_column = SETTINGS["target_column"]
    problem_type = SETTINGS["problem_type"]

    # 1) Load
    df_raw = load_raw_data(raw_path)

    # 2) Clean
    df_clean = clean_dataframe(df_raw, target_column=target_column)

    # 3) Save processed CSV (required artifact)
    save_csv(df_clean, processed_path)

    # 4) Validate (minimal fail-fast)
    required_columns = [target_column] + SETTINGS["features"]["numeric_passthrough"] + SETTINGS["features"]["categorical_onehot"]
    validate_dataframe(df_clean, required_columns=required_columns)

    # 5) Split (Notebook-aligned year split if possible)
    year_col = SETTINGS.get("year_column")
    test_year = SETTINGS.get("test_year")

    if year_col in df_clean.columns and test_year is not None:
        print(f"[main.main] Using year split: train < {test_year}, test == {test_year}")  # TODO: replace with logging later
        df_train = df_clean[df_clean[year_col] < test_year].copy()
        df_test = df_clean[df_clean[year_col] == test_year].copy()
    else:
        print("[main.main] Year split not possible; falling back to random train_test_split.")  # TODO: replace with logging later
        df_train, df_test = train_test_split(df_clean, test_size=0.2, random_state=42)

    X_train = df_train.drop(columns=[target_column])
    y_train = df_train[target_column]
    X_test = df_test.drop(columns=[target_column])
    y_test = df_test[target_column]

    # Stratify only for classification (and guard for edge cases)
    if problem_type == "classification":
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                df_clean.drop(columns=[target_column]),
                df_clean[target_column],
                test_size=0.2,
                random_state=42,
                stratify=df_clean[target_column],
            )
        except Exception:
            X_train, X_test, y_train, y_test = train_test_split(
                df_clean.drop(columns=[target_column]),
                df_clean[target_column],
                test_size=0.2,
                random_state=42,
                stratify=None,
            )

    # 6) Fail-fast feature checks
    configured_cols = SETTINGS["features"]["numeric_passthrough"] + SETTINGS["features"]["categorical_onehot"]
    missing_feats = [c for c in configured_cols if c not in X_train.columns]
    if missing_feats:
        raise ValueError(f"Configured feature columns missing from data: {missing_feats}")

    # 7) Build feature recipe (unfitted)
    preprocessor = get_feature_preprocessor(
        quantile_bin_cols=SETTINGS["features"]["quantile_bin"],
        categorical_onehot_cols=SETTINGS["features"]["categorical_onehot"],
        numeric_passthrough_cols=SETTINGS["features"]["numeric_passthrough"],
        n_bins=SETTINGS["features"]["n_bins"],
    )

    # 8) Train (fit pipeline on training only)
    model = train_model(X_train=X_train, y_train=y_train, preprocessor=preprocessor, problem_type=problem_type)

    # 9) Save model (required artifact)
    save_model(model, model_path)

    # 10) Evaluate
    score = evaluate_model(model, X_test=X_test, y_test=y_test, problem_type=problem_type)
    print(f"[main.main] Final score returned (single float): {score:.4f}")  # TODO: replace with logging later

    # 11) Inference (example: run on X_test)
    df_pred = run_inference(model, X_infer=X_test)

    # 12) Save predictions (required artifact)
    save_csv(df_pred, predictions_path)

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Add richer reporting (feature importances, experiment tracking)
    # Why: Production systems need traceability and monitoring signals.
    #
    # Placeholder (Remove this after implementing your code):
    print("Warning: Student has not implemented this section yet")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    print("[main.main] Pipeline completed successfully.")  # TODO: replace with logging later


if __name__ == "__main__":
    main()
