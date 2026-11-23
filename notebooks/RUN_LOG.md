# Run log

One line per model run, including failures and dropped variants. Validation scores only until the final table.

| Date | Who | Notebook | Model and key parameters | Features | Validation | MAE bps (IG / HY) | Notes |
|---|---|---|---|---|---|---|---|
| 2026-11-17 | Miguel | scratch (feasibility check) | LightGBM, 600 trees, learning rate 0.05, max depth 10, 30 leaves; untuned; baselines from `ml2.baseline` | all 34 | fit on train, scored on valid | 19.5 / 70.1 (matrix re-marked each date: 21.0 / 96.8; frozen: 23.0 / 98.5) | Panel built 16 Nov. Gate check before any logged run. IG gain 7.1%, interval -0.8% to 14.3%: fails; HY 27.6%, passes. Led to the gate change logged in docs/SCOPE.md. Not a candidate model |
| 2026-11-19 | Miguel | 99_final (dry run, `OPEN_TEST = False`) | Placeholder regression tree, max depth 8, min 50 per leaf; untuned | all 34 | fit on train, scored on valid | 25.6 / 84.7 (matrix re-marked each date: 21.0 / 96.8) | Pipeline check of the final notebook on the panel built 16 Nov, test not opened. Not a candidate model |
| 2026-11-19 | Miguel | 99_final (dry run, `OPEN_TEST = False`) | Same placeholder tree | all 34 | fit on train, scored on valid | 25.1 / 85.4 (matrix re-marked each date: 21.2 / 97.3) | Rerun on the panel rebuilt 19 Nov: stock data and sector blanked for 46 issuers (5,188 bond-months) whose stock company could not be confirmed as the bond issuer. Matrix moves because those rows now sit in the Unknown sector. Test not opened. Not a candidate model |
| 2026-11-19 | Miguel | 99_final (dry run, `OPEN_TEST = False`) | Same placeholder tree | all 34 | fit on train, scored on valid | 23.9 / 101.1 (matrix re-marked each date: 21.3 / 96.3) | Rerun on the panel rebuilt 19 Nov 14:23 (sha256 92d3aeb8). Grade fix: bonds whose TRACE master record is newer than all their trades take their company's majority grade (11,523 rows relabelled, 10,802 of them HY to IG; HY rows 25,179 to 15,098). Checked on valid before adopting it: XGBoost from 03 keeps its gain over the bar in IG and HY. The rebuild also picked up changed balance-sheet debt for 10 companies (443 rows, 2024 to early 2025) from the source. Test not opened. Not a candidate model |
| 2026-11-19 | Miguel | 99_final (dry run, `OPEN_TEST = False`) | Same placeholder tree | all 33 | fit on train, scored on valid | 22.1 / 82.2 (matrix re-marked each date: 21.2 / 88.4) | Rerun on the panel rebuilt after the adversarial review (sha256 d817a09a): bonds matched to their company by issuer code where the ticker match was wrong or missing (189,982 rows, 881 companies); stock data per company with one listing per day; debt only from USD filings; market cap left empty on share-count jumps; profile sectors mapped to GICS names; annex label from next month's spread before filters; half-day month-ends skipped; bond_age_years no longer a feature (future information). Test not opened. Not a candidate model |
| 2026-11-19 | Miguel | 02_baseline_matrix_pricing | Median spread (class baseline) | grade, maturity, sector | fit on train, scored on valid | 29.8 / 126.9 | Panel d817a09a. All: 37.1. Reference row |
| 2026-11-19 | Miguel | 02_baseline_matrix_pricing | Matrix pricing, frozen at the training months | grade, maturity, sector | fit on train, scored on valid | 23.2 / 86.1 | Panel d817a09a. All: 27.9. Reference row |
| 2026-11-19 | Miguel | 02_baseline_matrix_pricing | Matrix pricing, re-marked each date (gate baseline) | grade, maturity, sector | fit on train, scored on valid | 21.2 / 88.4 | Panel d817a09a. All: 26.2. The bar every model is gated against. In HY most cells fall back to grade x maturity |
| 2026-11-19 | Miguel | scratch (target choice for the EDA, step 3) | XGBoost with the 03 settings (500 trees, depth 10, learning rate 0.03), on spread and on log(spread + 101) | all 33 | fit on train, scored on valid | spread 15.2 / 53.8; log 13.9 / 46.0 (bar 21.2 / 88.4) | Panel d817a09a. The spread is right-skewed (skew 1.6 IG, 5.4 HY). The log target cuts MAE in both grades: gain over the bar 34% IG (interval from 30%) and 48% HY (from 39%), against 28% and 39% on the spread. Chosen on valid, so pending the team's decision. Not a candidate model |
| 2026-11-19 | Miguel | 03_trees_ensembles | Pruned regression tree (depth 14, 50 per leaf) | all 33 | fit on train, scored on valid | 19.1 / 81.7 | gain vs bar IG 9.7%, HY 7.6%. Run on the final panel (sha256 d817a09a) to check the notebook end to end; the owner reruns it |
| 2026-11-19 | Miguel | 03_trees_ensembles | Bagging (random forest, max_features=None) | all 33 | fit on train, scored on valid | 15.7 / 63.6 | gain vs bar IG 26.1%, HY 28.0%. Run on the final panel (sha256 d817a09a) to check the notebook end to end; the owner reruns it |
| 2026-11-19 | Miguel | 03_trees_ensembles | Random forest | all 33 | fit on train, scored on valid | 15.4 / 62.1 | gain vs bar IG 27.2%, HY 29.7%. Run on the final panel (sha256 d817a09a) to check the notebook end to end; the owner reruns it |
| 2026-11-19 | Miguel | 03_trees_ensembles | XGBoost (500 trees, depth 10, learning rate 0.03) | all 33 | fit on train, scored on valid | 15.2 / 53.8 | gain vs bar IG 28.1%, HY 39.1%. Run on the final panel (sha256 d817a09a) to check the notebook end to end; the owner reruns it |
| 2026-11-19 | Miguel | 03_trees_ensembles | LightGBM | all 33 | fit on train, scored on valid | 17.2 / 64.3 | gain vs bar IG 19.0%, HY 27.2%. Run on the final panel (sha256 d817a09a) to check the notebook end to end; the owner reruns it |
| 2026-11-19 | Miguel | 04_knn_comparables | KNN model 1 (k = 5, distance weights) | all 33 | fit on train, scored on valid | 16.8 / 54.5 | gain vs bar IG 20.6%, HY 38.3%. Run on the final panel (sha256 d817a09a) to check the notebook end to end; the owner reruns it |
| 2026-11-19 | Miguel | 04_knn_comparables | KNN model 2 (grid search: k = 8, distance weights) | all 33 | fit on train, scored on valid | 16.8 / 54.8 | gain vs bar IG 20.9%, HY 38.0%. Run on the final panel (sha256 d817a09a) to check the notebook end to end; the owner reruns it |
| 2026-11-19 | Miguel | 04_knn_comparables | KNN model 3 (bagging of KNNs) | all 33 | fit on train, scored on valid | 17.0 / 56.3 | gain vs bar IG 19.8%, HY 36.3%. Run on the final panel (sha256 d817a09a) to check the notebook end to end; the owner reruns it |
| 2026-11-19 | Miguel | 04_knn_comparables | KNN, trader's short list (7 inputs) | all 33 | fit on train, scored on valid | 22.2 / 63.5 | gain vs bar IG -4.6%, HY 28.1%. Run on the final panel (sha256 d817a09a) to check the notebook end to end; the owner reruns it |
| 2026-11-20 | Miguel | 05_neural_network | Perceptron (one neuron) | all 33 | fit on train to Feb 2025, early stopping on Mar to May 2025, scored on valid | 18.8 / 70.2 | gain vs bar IG 11.0%, HY 20.5%. Run on the final panel (sha256 d817a09a) to check the notebook end to end; the owner reruns it |
| 2026-11-20 | Miguel | 05_neural_network | MLP (64 and 32 ReLU, dropout 0.1, L2 1e-5) | all 33 | fit on train to Feb 2025, early stopping on Mar to May 2025, scored on valid | 13.7 / 56.8 | gain vs bar IG 35.2%, HY 35.8%. Run on the final panel (sha256 d817a09a) to check the notebook end to end; the owner reruns it |
| 2026-11-20 | Miguel | 05_neural_network | Random forest on the same rows as the MLP | all 33 | fit on train to Feb 2025, early stopping on Mar to May 2025, scored on valid | 15.4 / 66.2 | gain vs bar IG 27.4%, HY 25.1%. Run on the final panel (sha256 d817a09a) to check the notebook end to end; the owner reruns it |
| 2026-11-22 | Farah Alnuseirat | 06_annex_classification | Logistic regression | all 33 | fit on train, scored on valid | n/a (AUC 0.802, log loss 0.095, threshold 0.009: precision 0.03, recall 0.93) | widens_25bp, panel d817a09a |
| 2026-11-22 | Farah Alnuseirat | 06_annex_classification | Random forest | all 33 | fit on train, scored on valid | n/a (AUC 0.800, log loss 0.092, threshold 0.076: precision 0.10, recall 0.59) | widens_25bp, panel d817a09a |
| 2026-11-22 | Farah Alnuseirat | 06_annex_classification | KNN (30,000-row sample) | all 33 | fit on train, scored on valid | n/a (AUC 0.777, log loss 0.150, threshold 0.030: precision 0.04, recall 0.73) | widens_25bp, panel d817a09a |
| 2026-11-22 | Farah Alnuseirat | 06_annex_classification | Pruned decision tree | all 33 | fit on train, scored on valid | n/a (AUC 0.775, log loss 0.107, threshold 0.085: precision 0.12, recall 0.42) | widens_25bp, panel d817a09a |
| 2026-11-22 | Farah Alnuseirat | 06_annex_classification | Categorical Naive Bayes | all 33 | fit on train, scored on valid | n/a (AUC 0.735, log loss 0.438, threshold 0.453: precision 0.07, recall 0.43) | widens_25bp, panel d817a09a |
| 2026-11-22 | Farah Alnuseirat | 06_annex_classification | Categorical NB, one per correlated group | 28 predictors | fit on train, scored on valid | n/a (AUC 0.716, log loss 0.288) | widens_25bp, panel d817a09a |
| 2026-11-22 | Farah Alnuseirat | 06_annex_classification | Gaussian Naive Bayes | all 33 | fit on train, scored on valid | n/a (AUC 0.692, log loss 28.694) | widens_25bp, panel d817a09a |
| 2026-11-22 | Farah Alnuseirat | 06_annex_classification | Reference (training rate) | training prevalence | fit on train, scored on valid | n/a (AUC 0.500, log loss 0.104) | widens_25bp, panel d817a09a |
| 2026-11-23 | Miguel | 99_final (dry run, `OPEN_TEST = False`) | Final model: XGBoost on log(spread + 101), 500 trees, max depth 10, learning rate 0.03, subsample 0.8, colsample 0.7 (notebook 03 grid) | all 33 | fit on train, scored on valid | 13.9 / 46.0 (matrix re-marked each date: 21.2 / 88.4) | Panel d817a09a. Reproduces the 19 Nov target check. Gate on valid: IG gain 34.4% (interval 29.8% to 39.3%), HY 47.9% (38.9% to 55.4%): passes. Test not opened |

## Final result (P14)

Test was opened once, after the decision below was pushed (commit 23a3887). Run with xgboost 3.4.1 and scikit-learn 1.9.1: the lock
fingerprint includes every model setting, so rerunning `99_final` with other versions can change it.

| Date | Who | Notebook | Model and key parameters | Features | Validation | MAE bps (IG / HY) | Notes |
|---|---|---|---|---|---|---|---|
| 2026-11-23 | Miguel | 99_final (final, `OPEN_TEST = True`) | XGBoost on log(spread + 101), as chosen below | all 33 | fit on train + valid, scored on test (opened once, 01:25 UTC, lock 4b83ed1b) | 12.3 / 53.5 on **test** (matrix re-marked each date: 22.2 / 104.5) | Panel d817a09a. Gate on test: IG gain 44.8% (interval 41.4% to 47.8%), HY 48.8% (40.9% to 55.4%): **passes**. All grades 15.3 against 28.3. The final table |

## Decision before test (P13), 2026-11-23

Final model: **XGBoost on log(spread + 101)**, settings as in the dry run above, fitted on train + valid and scored
once on test in `99_final`.

Chosen on valid, against the bar of 21.2 / 88.4 bp (IG / HY), from the comparison tables in `reports/`:

| Model | IG | HY |
|---|---|---|
| XGBoost on log(spread + 101) | 13.9 | 46.0 |
| MLP (05) | 13.7 | 56.8 |
| XGBoost on the spread (03) | 15.2 | 53.8 |
| KNN, k = 5, distance-weighted (04) | 16.8 | 54.5 |
| Random forest (03) | 15.4 | 62.1 |

Reasons: it ties the MLP in IG (0.2 bp apart) and is 11 bp (19%) better in HY, where the bar is weakest and the
money is; it has the best HY interval of any model; it trains on all rows (the MLP holds out its early-stopping
months); and notebook 03 explains it with permutation importance and SHAP.

Not tried, and said so in the report: the MLP on the log target, and convolutional networks (our rows are 33
unordered columns with no grid or sequence for a convolution to use).
