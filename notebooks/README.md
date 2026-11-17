# notebooks

The lab. One owner per notebook; shared code goes in `src/ml2`, not copied between notebooks.

| Notebook | Content | Owner |
|---|---|---|
| `00_data_access.ipynb` | Connect, load the panel, first look | Miguel |
| `01_eda.ipynb` | Exploratory analysis: spreads by grade, maturity, sector; missing values; outliers | |
| `02_baseline_matrix_pricing.ipynb` | Baselines from `ml2.baseline`: matrix pricing re-marked each date (the bar to beat), frozen matrix and median for reference | |
| `03_trees_ensembles.ipynb` | Regression tree, bagging, random forest, gradient boosting | |
| `04_knn_comparables.ipynb` | KNN "comparable bonds" pricing | |
| `05_neural_network.ipynb` | On standby: MLP regressor, only once its class notebook is released | |
| `06_annex_classification.ipynb` | `widens_25bp`: Naive Bayes and every classifier | |
| `99_final.ipynb` | The final notebook for the submission: runs top to bottom from the panel, scores test once and runs the quality gate (`gate_report` against matrix pricing re-marked each date) | |

Rules: import splits and metrics from `ml2`, never split randomly, do not touch `partition == "test"`
until the final table, and add a line to `RUN_LOG.md` for every model run.
