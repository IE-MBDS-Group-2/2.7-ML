"""Where the final model helps, on valid only: gain by grade, maturity and sector, a review rule, new companies.

The final model's settings (as in 99_final), fitted on train and scored on valid against matrix pricing re-marked
each date. No test rows are loaded. Run from the repo root: python scripts/valid_breakdowns.py (a few minutes).
Prints aggregates only.
"""
import pandas as pd
import xgboost as xgb
from sklearn.pipeline import Pipeline

from ml2.baseline import matrix_same_date
from ml2.config import load_config
from ml2.data import load_panel
from ml2.features import make_preprocessor, xy
from ml2.final import on_log_spread

TARGET = load_config()["problem"]["target"]


def make_final_model():
    """The final model's settings, as in 99_final."""
    return on_log_spread(Pipeline([
        ("pre", make_preprocessor(scale=False, one_hot=False)),
        ("est", xgb.XGBRegressor(objective="reg:squarederror", n_estimators=500, max_depth=10, learning_rate=0.03,
                                 subsample=0.8, colsample_bytree=0.7, random_state=42, n_jobs=-1)),
    ]))


def gain_table(score: pd.DataFrame, by) -> pd.DataFrame:
    g = score.groupby(by, observed=True).agg(rows=(TARGET, "size"), model=("err_model", "mean"),
                                             bar=("err_bar", "mean"))
    g["gain"] = 1 - g["model"] / g["bar"]
    return g.round(3)


def main() -> None:
    panel = load_panel(partitions=["train", "valid"])
    fit = panel[panel["partition"] == "train"]
    score = panel[(panel["partition"] == "valid") & panel[TARGET].notna()].copy()
    model = make_final_model().fit(*xy(fit))
    score["model"] = model.predict(xy(score)[0])
    score["bar"] = matrix_same_date(score)
    score["err_model"] = (score["model"] - score[TARGET]).abs()
    score["err_bar"] = (score["bar"] - score[TARGET]).abs()
    score["gap"] = (score["model"] - score["bar"]).abs()
    score["new_company"] = ~score["issuer_ticker"].isin(set(fit["issuer_ticker"].dropna()))

    pd.set_option("display.width", 200)
    print(f"Fit on train ({len(fit):,} rows), scored on valid ({len(score):,} rows). MAE in bp.")
    print(gain_table(score, "grade_bucket"))
    print("\nBy grade and maturity bucket:")
    print(gain_table(score, ["grade_bucket", "maturity_bucket"]))
    print("\nBy sector:")
    print(gain_table(score, "sector").sort_values("rows", ascending=False))
    print("\nNew companies (no bond in the training months):")
    print(gain_table(score, ["new_company", "grade_bucket"]))
    print("companies new:", score.loc[score["new_company"], "issuer_ticker"].nunique())

    print("\nReview rule: bonds where the model and matrix pricing differ by more than Z bp")
    for grade, d in score.groupby("grade_bucket"):
        for z in [25, 50, 75, 100, 150, 200]:
            s = d[d["gap"] > z]
            if s.empty:
                continue
            closer = (s["err_model"] < s["err_bar"]).mean()
            print(f"{grade} Z={z:>3}: {len(s) / len(d):6.1%} of rows ({len(s):,}); "
                  f"MAE model {s['err_model'].mean():6.1f}, bar {s['err_bar'].mean():6.1f}; "
                  f"model closer on {closer:.0%}")


if __name__ == "__main__":
    main()
