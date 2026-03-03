# tests/test_train.py
"""
Unit tests for src.train.train_model

This test verifies:
- train_model can accept a minimal DataFrame and unfitted preprocessor,
- returns a fitted sklearn Pipeline with a working predict method.

Run with:
    pytest -q tests/test_train.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import get_feature_preprocessor
from src.train import train_model


def make_dummy_data():
    # Create a tiny, deterministic dataset matching baseline SETTINGS
    X = pd.DataFrame(
        {
            "num_feature": [0.0, 1.0, 2.0, 3.0, 4.0],
            "cat_feature": ["a", "b", "a", "b", "c"],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0], name="target")
    return X, y


def test_train_model_fits_and_predicts_classification():
    """
    Train a simple classification pipeline and ensure predict works and shapes match.
    """
    X, y = make_dummy_data()

    # Build unfitted preprocessor matching the dummy columns
    preprocessor = get_feature_preprocessor(
        quantile_bin_cols=[],
        categorical_onehot_cols=["cat_feature"],
        numeric_passthrough_cols=["num_feature"],
        n_bins=3,
    )

    model = train_model(X_train=X, y_train=y, preprocessor=preprocessor, problem_type="classification")

    # Basic sanity checks
    assert hasattr(model, "predict"), "Returned model must implement predict()"
    preds = model.predict(X)
    assert len(preds) == len(X), "Number of predictions must match number of input rows"

    # Predictions for classification should be integers or convertible to int-like classes
    assert preds.shape == (len(X),)
    # At least one predicted class should be present (sanity)
    unique_preds = np.unique(preds)
    assert unique_preds.size >= 1


def test_train_model_fits_and_predicts_regression():
    """
    Train a simple regression pipeline and ensure predict works and shapes match.
    """
    X, _ = make_dummy_data()
    # Create a continuous target
    y_reg = pd.Series([0.1, 1.5, 0.2, 1.8, 0.0], name="target")

    preprocessor = get_feature_preprocessor(
        quantile_bin_cols=[],
        categorical_onehot_cols=["cat_feature"],
        numeric_passthrough_cols=["num_feature"],
        n_bins=3,
    )

    model = train_model(X_train=X, y_train=y_reg, preprocessor=preprocessor, problem_type="regression")

    preds = model.predict(X)
    assert preds.shape == (len(X),)
    # Regression predictions should be float-ish
    assert preds.dtype.kind in ("f", "i")