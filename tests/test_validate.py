import pandas as pd
import pytest

from src.validate import validate_dataframe


def test_validate_passes_for_valid_df():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    assert validate_dataframe(df, ["a", "b"]) is True


def test_validate_fails_for_missing_columns():
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataframe(df, ["a", "b"])


def test_validate_fails_for_empty_df():
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="dataframe is empty"):
        validate_dataframe(df, ["a"])


def test_validate_fails_for_all_null_required_column():
    df = pd.DataFrame({"a": [None, None], "b": [1, 2]})
    with pytest.raises(ValueError, match="required columns are entirely null"):
        validate_dataframe(df, ["a", "b"])


def test_validate_fails_for_duplicated_columns():
    # Build DataFrame with duplicated columns by constructing from dict then renaming
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df.columns = ["a", "a"]  # force duplicate column names
    with pytest.raises(ValueError, match="duplicated column names"):
        validate_dataframe(df, ["a"])