"""Checks on the panel behind four numbers in the report that no notebook prints. Train, gap and valid only.

Run from the repo root: python scripts/report_checks.py. Prints aggregates only.

1. Median IG spread per month-end: how the market moved across train and valid (why the split follows time).
2. Medium-term note programmes (the issuer name marks a medium-term, structured or retail note programme): their
   share of training rows, and of the most extreme IG spreads.
3. Rows without the annex label (`widens_25bp`, no spread next month), by fifth of spread: missing labels rise with
   the spread, which biases the annex.
4. SOFR is taken on the snapshot date itself, while the New York Fed publishes it the next business morning: a
   one-day look-ahead in one market input (stated in the report's limits; nothing to compute here).
"""
from ml2.data import load_panel


def main() -> None:
    panel = load_panel(partitions=["train", "gap", "valid"])

    ig = panel[panel["grade_bucket"] == "IG"]
    print("1. Median IG spread by month-end (bp):")
    print(ig.groupby("snapshot_date")["spread_bps"].median().round(0).to_string())

    train = panel[panel["partition"] == "train"]
    pattern = r"MEDIUM[ -]?TERM|MED TERM|MTN|STRUCT|NOTES BOOK|SR NTS"
    mtn = train["issuer_name"].str.upper().str.contains(pattern, regex=True, na=False)
    print(f"\n2. Medium-term note programmes: {mtn.mean():.1%} of training rows")
    ig_train = train[train["grade_bucket"] == "IG"]
    for q in [0.001, 0.005, 0.01]:
        lo, hi = ig_train["spread_bps"].quantile([q, 1 - q])
        tails = (ig_train["spread_bps"] <= lo) | (ig_train["spread_bps"] >= hi)
        share = mtn[ig_train.index][tails].mean()
        print(f"   outer {q:.1%} of IG spreads (both ends): {share:.0%} are programme notes")

    print("\n3. Rows without the annex label, by fifth of spread (lowest to highest):")
    for part in ["train", "valid"]:
        rows = panel[panel["partition"] == part].copy()
        rows["fifth"] = rows["spread_bps"].rank(pct=True).mul(5).clip(upper=4.999).astype(int) + 1
        by_fifth = rows.groupby("fifth")["widens_25bp"].apply(lambda s: s.isna().mean())
        print(f"   {part}: overall {rows['widens_25bp'].isna().mean():.1%}; "
              + ", ".join(f"{v:.1%}" for v in by_fifth))


if __name__ == "__main__":
    main()
