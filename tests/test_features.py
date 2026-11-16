import numpy as np
import pandas as pd

from ml2.features import categorical_columns, make_preprocessor, numeric_columns, xy
from ml2.schema import COLUMN_DOCS, NOT_FEATURES, feature_columns


def _fake_panel(n=60, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(size=n) for c, _, _ in COLUMN_DOCS})
    df["maturity_bucket"] = rng.choice(["1-3y", "3-5y"], n)
    df["grade_bucket"] = rng.choice(["IG", "HY"], n)
    df["debt_type"] = rng.choice(["S-NT", "SUB-NT"], n)
    df["sector"] = rng.choice(["Energy", None], n)
    df["is_144a"] = rng.choice([True, False], n)
    df.loc[0, "vol_60d"] = np.nan
    df.loc[1, "spread_bps"] = np.nan
    return df


def test_no_leaky_column_is_a_feature():
    feats = set(feature_columns())
    assert feats.isdisjoint(NOT_FEATURES)
    assert set(numeric_columns()) | set(categorical_columns()) == feats


def test_xy_drops_missing_target_and_converts_booleans():
    X, y = xy(_fake_panel(), target="spread_bps")
    assert len(X) == len(y) == 59
    assert X["is_144a"].dtype == float
    assert list(X.columns) == feature_columns()


def test_preprocessor_handles_missing_values_and_categories():
    X, y = xy(_fake_panel(), target="spread_bps")
    out = make_preprocessor().fit_transform(X)
    assert not np.isnan(out).any()
    out_trees = make_preprocessor(scale=False, one_hot=False).fit_transform(X)
    assert out_trees.shape[0] == len(X)
