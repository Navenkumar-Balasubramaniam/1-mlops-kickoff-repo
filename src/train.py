"""
Educational Goal:
- Why this module exists in an MLOps system: Train the model in a controlled,
  reproducible way that prevents leakage (fit preprocessing on train only).
- Responsibility (separation of concerns): Training logic and estimator choice
  belong here, not mixed into main.py.
- Pipeline contract (inputs and outputs): Accepts train split + preprocessor and
  returns a fitted sklearn Pipeline artifact.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

from __future__ import annotations

import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV, LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline


def train_model(X_train: pd.DataFrame, y_train: pd.Series, preprocessor, problem_type: str):
    """
    Inputs:
    - X_train: training features (DataFrame)
    - y_train: training labels/target (Series)
    - preprocessor: ColumnTransformer (unfitted)
    - problem_type: "regression" or "classification"
    Outputs:
    - model: fitted sklearn Pipeline(preprocess + estimator)
    Why this contract matters for reliable ML delivery:
    - The Pipeline ensures preprocessing learned from training data is reused
      identically during evaluation and inference.
    """
    print("[train.train_model] Training model pipeline...")  # TODO: replace with logging later

    if problem_type not in {"regression", "classification"}:
        raise ValueError("problem_type must be either 'regression' or 'classification'")

    # Notebook-aligned defaults:
    # Regression: preprocess -> LassoCV feature selection -> LinearRegression
    if problem_type == "regression":
        selector = SelectFromModel(
            estimator=LassoCV(
                cv=5,
                n_alphas=100,
                max_iter=20000,
                random_state=42,
            ),
            # threshold=0.0 keeps features with non-zero coefficients
            threshold=0.0,
        )

        model = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("select", selector),
                ("model", LinearRegression()),
            ]
        )

    # Classification: keep simple baseline (LassoCV is not appropriate here)
    else:
        model = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", LogisticRegression(max_iter=500)),
            ]
        )

    # Fit entire pipeline on training data ONLY (preprocess + selector + model)
    model.fit(X_train, y_train)

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Make LassoCV parameters configurable via config.yml
    # (cv, n_alphas, max_iter, threshold, random_state).
    # Why: Different datasets need different regularization strength and runtime tradeoffs.
    #
    # Optional debug: print number of selected features
    # if problem_type == "regression":
    #     support = model.named_steps["select"].get_support()
    #     print(f"[train.train_model] Selected {support.sum()} features after LassoCV")
    #
    # Placeholder (Remove this after implementing your code):
    print("Warning: Student has not implemented this section yet")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    return model
