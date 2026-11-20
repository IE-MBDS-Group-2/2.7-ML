# 2.7-ML: Corporate bond pricing with machine learning

Machine Learning II group project, IE MBDS (Group 2). **Due Sun 22 Nov 2026, 23:59** on Blackboard.

A bank's pricing team has to put a fair price on thousands of corporate bonds every day, but most bonds do not
trade on a given day. Today the fallback is **matrix pricing**: a spread looked up from rating, maturity and
sector buckets, re-marked each day from the bonds that did trade. We predict each bond's **credit spread** (its extra yield over a Treasury bond of the same maturity)
from the bond's terms, how often it trades, the issuing company's stock and balance sheet, and the market backdrop,
and test whether machine learning beats matrix pricing.

| Read this | For |
|---|---|
| [docs/SCOPE.md](docs/SCOPE.md) | What we are building, the plan, the roles, a finance primer |
| [docs/DATA_ACCESS.md](docs/DATA_ACCESS.md) | Getting the data |
| [docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md) | Every column explained |

## Quick start

```bash
conda env create -f environment.yml
conda activate ml2-group2
copy .env.example .env        # macOS/Linux: cp .env.example .env
```

Put `bond_month_panel.parquet` and `manifest.json` from the group folder in `data/processed/`
([docs/DATA_ACCESS.md](docs/DATA_ACCESS.md)), run `notebooks/00_data_access.ipynb`, then in a notebook:

```python
from ml2.data import load_panel
from ml2.schema import feature_columns

df = load_panel()                           # every partition except test (opened once, in 99_final)
train = load_panel(partitions=["train"])    # one partition
X_cols = feature_columns()                  # allowed model inputs
```

## Repository layout

```text
.
├── README.md
├── config.yaml              # target, split dates, quality gate (single source of truth)
├── environment.yml          # team conda environment (installs src/ml2 in editable mode)
├── pyproject.toml           # the ml2 package
├── .env.example             # copy to .env (gitignored); a database login only if we use one
├── docs/
│   ├── SCOPE.md             # project scope, plan, roles, finance primer
│   ├── DATA_ACCESS.md
│   └── DATA_DICTIONARY.md
├── notebooks/               # the lab: one owner per notebook, RUN_LOG.md for every model run
├── src/ml2/                 # shared code every notebook imports
│   ├── data.py              #   check_panel(), load_panel(): parquet or group database
│   ├── schema.py            #   columns, meanings, allowed features
│   ├── features.py          #   xy() and the shared preprocessing recipe
│   ├── splits.py            #   chronological partitions, walk-forward and issuer-grouped folds
│   ├── baseline.py          #   median and matrix-pricing baselines (one definition for everyone)
│   ├── evaluate.py          #   metrics in basis points, by stratum, quality gate with bootstrap interval
│   └── final.py             #   open_test_once(): test is scored once, for one model
├── scripts/                 # write_data_dictionary.py
├── data/                    # gitignored: processed/bond_month_panel.parquet and manifest.json
├── models/                  # gitignored: saved models
├── reports/                 # gitignored: figures for the report and deck
└── tests/                   # pytest
```

## Rules of the road

- **Data never goes to GitHub.** It is licensed. It lives in the private group folder (and the group database, if used) and in the submission zip only.
- **No random splits.** Use `ml2.splits`: chronological partitions with a one-month gap, plus issuer-grouped folds.
- **The test partition is opened once**, at the end, for the final results table.
- **The quality gate is fixed** in `config.yaml` (tightened once, before any logged run; see `docs/SCOPE.md`). Do not change it after seeing results.
- **Log every model run** (including dropped ideas) in `notebooks/RUN_LOG.md`.
- **Shared code goes in `src/ml2`**, not copied between notebooks.
- Branch from `dev` (`feature/<what>`), open a pull request into `dev`, get one review. `main` gets the final version.

## Checks

Pull requests run the `checks` workflow (lint and tests). Locally:

```bash
pytest
flake8 src scripts tests
```
