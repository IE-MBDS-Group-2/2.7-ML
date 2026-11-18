# Run log

One line per model run, including failures and dropped variants. Validation scores only until the final table.

| Date | Who | Notebook | Model and key parameters | Features | Validation | MAE bps (IG / HY) | Notes |
|---|---|---|---|---|---|---|---|
| 2026-11-17 | Miguel | scratch (feasibility check) | LightGBM, 600 trees, learning rate 0.05, max depth 10, 30 leaves; untuned; baselines from `ml2.baseline` | all 34 | fit on train, scored on valid | 19.5 / 70.1 (matrix re-marked each date: 21.0 / 96.8; frozen: 23.0 / 98.5) | Gate check before any logged run. IG gain 7.1%, interval -0.8% to 14.3%: fails; HY 27.6%, passes. Led to the gate change logged in docs/SCOPE.md. Not a candidate model |
| 2026-11-19 | Miguel | 99_final (dry run, `OPEN_TEST = False`) | Placeholder regression tree, max depth 8, min 50 per leaf; untuned | all 34 | fit on train, scored on valid | 25.6 / 84.7 (matrix re-marked each date: 21.0 / 96.8) | Pipeline check of the final notebook, test not opened. Not a candidate model |
