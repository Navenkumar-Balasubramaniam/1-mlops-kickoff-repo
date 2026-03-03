# tests/test_main.py
"""
Integration test (no model load) for src.main.main.

This test:
- runs the end-to-end pipeline via main.main()
- asserts required artifacts exist:
    - data/processed/clean.csv
    - models/model.joblib
    - reports/predictions.csv
- verifies that reports/predictions.csv is a non-empty CSV with a single column "prediction"

"""

from pathlib import Path
import pandas as pd

import src.main as main_module

ARTIFACTS = {
    "processed": Path("data/processed/clean.csv"),
    "model": Path("models/model.joblib"),
    "predictions": Path("reports/predictions.csv"),
}


def test_main_creates_artifacts_and_predictions_csv():
    # 1) Run pipeline
    main_module.main()

    # 2) Assert artifacts exist
    assert ARTIFACTS["processed"].exists(), f"Missing artifact: {ARTIFACTS['processed']}"
    assert ARTIFACTS["model"].exists(), f"Missing artifact: {ARTIFACTS['model']}"
    assert ARTIFACTS["predictions"].exists(), f"Missing artifact: {ARTIFACTS['predictions']}"

    # 3) Load and sanity-check processed CSV
    df_processed = pd.read_csv(ARTIFACTS["processed"])
    assert not df_processed.empty, "Processed CSV should not be empty"
    target_col = main_module.SETTINGS["target_column"]
    assert target_col in df_processed.columns, f"Processed data must contain target column '{target_col}'"

    # 4) Load and sanity-check predictions CSV
    df_preds = pd.read_csv(ARTIFACTS["predictions"])
    # Predictions must be non-empty and have exactly one column named "prediction"
    assert not df_preds.empty, "Predictions CSV should not be empty"
    assert list(df_preds.columns) == ["prediction"], "Predictions CSV must have exactly one column named 'prediction'"

    # 5) Optional: prediction count should be <= processed rows (since main runs inference on a subset)
    assert len(df_preds) <= len(df_processed), "Predictions rows cannot exceed processed data rows"