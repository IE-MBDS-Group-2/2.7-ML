# 2.7-ML

Machine Learning II group project, IE MBDS (Group 2).

The assignment: pick our own dataset and build a classification or regression model for a chosen target, using the
methods from the course. Deliverables: a summary report (max 4 pages), a technical annex, the database with an
explanation of the variables, the final notebook, and the presentation.

## Setup

```bash
conda env create -f environment.yml
conda activate ml2-group2
copy .env.example .env        # macOS/Linux: cp .env.example .env
```

## Layout

```text
.
├── config.yaml          # settings: paths, target, seed (no hardcoding in notebooks)
├── environment.yml      # shared conda environment
├── .env.example         # names of secrets; copy to .env (gitignored)
├── data/
│   ├── raw/             # source data, never edited (gitignored)
│   └── processed/       # cleaned data (gitignored)
├── notebooks/           # exploration and modelling
├── src/                 # shared code imported by notebooks
├── models/              # saved models (gitignored)
├── reports/figures/     # figures for the report and deck (gitignored)
└── tests/               # pytest
```

## Working together

- Branch per piece of work (`feature/<what>`), pull request into `main`, one review.
- Data, `.env` and models never go to GitHub.
- Code shared by more than one notebook goes in `src/`.
