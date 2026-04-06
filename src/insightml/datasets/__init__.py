"""Built-in demo datasets for InsightML — available in v0.5+."""

from __future__ import annotations

import pandas as pd
from pathlib import Path

_DATA_DIR = Path(__file__).parent / "data"


def load_titanic() -> pd.DataFrame:
    """Load the Titanic dataset (891 rows, classification target: 'survived').

    Returns:
        pandas DataFrame with columns: survived, pclass, name, sex, age,
        sibsp, parch, ticket, fare, cabin, embarked.
    """
    path = _DATA_DIR / "titanic.csv"
    if not path.exists():
        raise FileNotFoundError(
            "Titanic dataset not bundled yet (available in v0.5). "
            "Download from: https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        )
    return pd.read_csv(path)


def load_housing() -> pd.DataFrame:
    """Load the California Housing dataset (regression target: 'median_house_value').

    Returns:
        pandas DataFrame with 8 features and median_house_value target.
    """
    path = _DATA_DIR / "housing.csv"
    if not path.exists():
        try:
            from sklearn.datasets import fetch_california_housing
            housing = fetch_california_housing(as_frame=True)
            df = housing.frame
            return df
        except ImportError:
            raise FileNotFoundError(
                "Housing dataset not bundled yet. "
                "Install scikit-learn to use the built-in California Housing dataset."
            )
    return pd.read_csv(path)
