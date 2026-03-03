from __future__ import annotations

import pandas as pd

"""
Educational Goal:
- Why this module exists in an MLOps system: Fail fast before training starts.
  Bad inputs should be detected early, not after the model is trained.
- Responsibility (separation of concerns): Validation is different from cleaning.
  Validation checks assumptions; cleaning changes data.
- Pipeline contract (inputs and outputs): Takes a DataFrame and required column list;
  returns True if valid, otherwise raises a clear error.
"""


def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Validate dataframe schema and basic quality checks.

    Parameters
    df : pd.DataFrame
        DataFrame to validate.
    required_columns : list
        Columns that must exist in df.

    Returns
    bool
        True if validation passes.

    Raises
    ValueError
        If validation fails.
    """
    if df is None or df.empty:
        raise ValueError("Validation failed, dataframe is empty.")

    if not isinstance(required_columns, list) or len(required_columns) == 0:
        raise ValueError("Validation failed, required_columns must be a non-empty list.")

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Validation failed, missing required columns: {missing}")

    # Student checks (safe + generic)
    if df.columns.duplicated().any():
        raise ValueError("Validation failed: dataframe has duplicated column names.")

    all_null_required = [c for c in required_columns if df[c].isna().all()]
    if all_null_required:
        raise ValueError(f"Validation failed, required columns are entirely null: {all_null_required}")

    return True