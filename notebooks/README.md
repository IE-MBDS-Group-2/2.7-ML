# notebooks

The lab. One owner per notebook; shared code goes in `src/ml2`, not copied between notebooks.

| Notebook | Content | Owner |
|---|---|---|
| `00_data_access.ipynb` | Check your parquet copy, load the panel, first look | Miguel |
| `01_eda.ipynb` | Exploratory analysis: spreads by grade, maturity, sector; missing values; outliers | Role 2 |
| `02_baseline_matrix_pricing.ipynb` | Baselines from `ml2.baseline`: matrix pricing re-marked each date (the bar to beat), frozen matrix and median for reference | Role 2 |
| `03_trees_ensembles.ipynb` | Regression tree, bagging, random forest, gradient boosting | Role 3 |
| `04_knn_comparables.ipynb` | KNN "comparable bonds" pricing | Role 4 |
| `05_neural_network.ipynb` | On standby: MLP regressor, only once its class notebook is released | |
| `06_annex_classification.ipynb` | `widens_25bp`: Naive Bayes and every classifier | Role 5 |
| `99_final.ipynb` | The final notebook for the submission: runs top to bottom from the panel, scores test once and runs the quality gate (`gate_report` against matrix pricing re-marked each date). `OPEN_TEST = False` dry-runs it on valid; `True` opens test once and locks it to one model (`ml2.final`) | Miguel |

01 to 04 and 06 are filled in: every step of the owner's walkthrough is coded, following the class notebooks,
and runs top to bottom on the shared panel (01 and 02 in about 2 minutes, 03 about 10, 04 about 6, 06 about 3).
The owner runs it, writes the "Your reading" cells from the outputs, pastes the printed lines into `RUN_LOG.md`
and must be able to explain every line. Clear outputs before committing.

Rules: import splits and metrics from `ml2`, never split randomly, do not touch `partition == "test"`
until the final table, and add a line to `RUN_LOG.md` for every model run.
