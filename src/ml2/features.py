"""One preprocessing recipe for every model, so results are comparable.

    X, y = xy(train)                    # features and target from a panel slice
    pre = make_preprocessor()           # impute + scale numbers, one-hot categories
    model = Pipeline([("pre", pre), ("est", RandomForestRegressor())])
    model.fit(X, y)                     # the preprocessor is fitted on training rows only

Trees do not need scaling: make_preprocessor(scale=False). KNN, the neural network and
linear models do.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from ml2.config import load_config
from ml2.schema import feature_columns

CATEGORICAL = ["maturity_bucket", "grade_bucket", "debt_type", "sector"]


def categorical_columns() -> list[str]:
    return [c for c in feature_columns() if c in CATEGORICAL]


def numeric_columns() -> list[str]:
    return [c for c in feature_columns() if c not in CATEGORICAL]


def xy(df: pd.DataFrame, target: str | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Allowed features (booleans as 0/1) and the target. Rows with a missing target are dropped."""
    target = target or load_config()["problem"]["target"]
    data = df[df[target].notna()]
    X = data[feature_columns()].copy()
    for col in X.columns:
        if X[col].dtype == bool or str(X[col].dtype) == "boolean":
            X[col] = X[col].astype(float)
    return X, data[target]


def make_preprocessor(scale: bool = True, one_hot: bool = True) -> ColumnTransformer:
    """Median-impute (with missing-value flags) and optionally scale numbers; encode categories.

    one_hot=False uses integer codes instead (fine for trees, smaller and faster).
    """
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    if one_hot:
        encoder = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20, sparse_output=False)
    else:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    cat_steps = [("impute", SimpleImputer(strategy="constant", fill_value="Unknown")), ("encode", encoder)]
    return ColumnTransformer(
        [("num", Pipeline(num_steps), numeric_columns()), ("cat", Pipeline(cat_steps), categorical_columns())],
        verbose_feature_names_out=False,
    )
