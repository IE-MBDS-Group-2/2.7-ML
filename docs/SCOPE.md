# Scope: corporate bond pricing with machine learning

**Due:** Blackboard, **Sun 22 Nov 2026, 23:59** (one zip, `Group <name>.zip`). Presentation: 12 minutes plus 5 minutes
of questions. The professor can ask **anyone** about **any** ML II topic, not only our project.

## 1. The project in one paragraph

A bank's pricing team has to put a fair value on thousands of corporate bonds every day, for client statements,
fund valuations and risk reports. Most bonds do not trade on a given day, so there is no market price to copy.
The usual fallback is **matrix pricing**: look up a typical spread for the bond's rating, maturity and sector, and
apply it. We predict each bond's **credit spread** from its terms, its trading activity, the issuing company's
stock and balance sheet, and the market backdrop, and test whether machine learning beats matrix pricing.
Every ML II method with a class notebook gets a role, and our code follows that notebook: decision trees, bagging,
random forest and boosting (XGBoost, LightGBM), KNN, and Naive Bayes on a classification variant in the technical
annex. The neural network is on standby until its class notebook is released.

**Audience for the report:** the head of a bank's bond pricing team, not the professor.

## 2. Finance primer (5 minutes)

| Term | Meaning |
|---|---|
| **Corporate bond** | A loan to a company that investors can buy and sell. It pays a fixed **coupon** every year and repays the **face value** at **maturity** |
| **Yield** | The annual return you get if you buy the bond at today's price and hold it to maturity. Price up = yield down |
| **Treasury yield** | The yield of a US government bond: the "risk-free" benchmark for the same maturity |
| **Credit spread** | Bond yield minus Treasury yield of the same maturity. The extra return investors want for the risk that the company does not pay. Measured in **basis points** (bp): 1 bp = 0.01%, so 150 bp = 1.5% |
| **Investment grade (IG) / high yield (HY)** | Safer companies (spreads around 50 to 200 bp) versus riskier ones (around 250 to 1,000 bp). Our two strata |
| **Matrix pricing** | Today's fallback: the median spread of similar bonds (rating x maturity x sector) that traded that day, re-marked every day. Our baseline and the bar for the quality gate |
| **TRACE** | The US regulator's record of every corporate bond trade. Our source of real yields and volumes |
| **Liquidity** | How often and how much a bond trades. Rarely traded bonds usually carry a higher spread |
| **Equity signals** | When a company's stock is volatile, falling or highly leveraged, its bonds usually have higher spreads. That is why the issuer's stock data are features |

## 3. What the brief asks for

| Deliverable | Where it comes from | Owner |
|---|---|---|
| Summary report, max 4 pages: executive summary, technical approach, best solution, recommendations for managers | Results of sections 6 to 8 | Report and deck lead |
| Technical annex | EDA, every model's runs (`notebooks/RUN_LOG.md`), validation detail, the Naive Bayes classification variant | Each owner writes their own part |
| Database plus explanation of the variables | `bond_month_panel.parquet` and [DATA_DICTIONARY.md](DATA_DICTIONARY.md) | Miguel |
| Final notebook | `notebooks/99_final.ipynb`, runs top to bottom from the panel | Miguel (integration), with each owner's code |
| Presentation | Deck, 12 minutes, every member presents what they built | Report and deck lead |

Graded points: executive summary, professional look, EDA, data processing and feature work, technical approach,
**validation strategy**, model evaluation, conclusions and recommendations, technical annex.

## 4. Data

One table, `bond_month_panel`: one row per bond per month-end, January 2024 to March 2026. Miguel prepares it from
TRACE (via IE's WRDS subscription), the US Treasury yield curve, market indicators and the issuers' stock prices
and balance sheets. You get it through the read-only group database or as a parquet file:
[DATA_ACCESS.md](DATA_ACCESS.md). Every column is explained in [DATA_DICTIONARY.md](DATA_DICTIONARY.md).

- **Bonds kept:** straight fixed-coupon corporate bonds of companies with a listed stock, traded on the month-end
  day, 1 to 30.5 years to maturity.
- **Size:** about 181,000 bond-months: some 13,800 bonds from about 800 companies over 27 month-ends, about
  6,700 bonds per snapshot. Most rows are flagged investment grade (about 85%). Exact counts are in the data
  dictionary ("Last build").
- **Licensed data:** it stays in the group and in the submission zip. Never commit it or share it.

## 5. Target and features

**Target:** `spread_bps`, the bond's yield minus the Treasury yield at the same maturity, in basis points (regression).

**Features** (all already in the table; `ml2.schema.feature_columns()` lists them):

| Group | Examples | Why it should matter |
|---|---|---|
| Bond terms | years to maturity, coupon, IG/HY, seniority, 144A | Longer, junior and lower-grade bonds pay more |
| Liquidity | trades, volume, days traded in the month | Illiquid bonds pay a premium |
| Issuer | number of bonds outstanding, sector | Bigger issuers trade more; sectors differ in risk |
| Issuer's stock | volatility, past return, beta, market cap, leverage, distance to default | The stock market's view of the company's risk |
| Market | Treasury yields, curve slope, high-yield and BBB index spreads, VIX, SOFR | Spreads move together with the market |

**Not features:** `yield_pct`, `ust_yield_at_maturity_pct` and `price` contain the target; the annex columns contain
the future. Using any of them is leakage.

## 6. Models

| Role | Model | Class notebook it follows | Why it is here |
|---|---|---|---|
| **Baselines, the bar to beat** (`ml2.baseline`) | Matrix pricing re-marked on every snapshot date: median same-date spread of other companies' bonds in the grade x maturity bucket x sector cell (the gate baseline). Reference rows: the median of the training spreads, and matrix pricing frozen at the training months | Regression_Improve Bagg & RF with Boosting | What pricing teams do today, and the class baseline |
| Interpretable | Regression tree, pruned by depth and leaf size, then cost-complexity pruning with the 1-SE rule | RTrees_boston | Rules a pricing team can read |
| Ensembles | Bagging, random forest, XGBoost, LightGBM; out-of-bag error, permutation importance, SHAP | Regression_Improve Bagg & RF with Boosting, RF_risk | Likely the most accurate on tables like ours |
| Comparable bonds | KNN regression, k tuned, scaled features, bagged KNN | 2. KNN_IBM attrition | "Price a bond from its most similar bonds": how traders already think |
| Classification annex | Gaussian and Categorical Naive Bayes against the other classifiers, threshold from out-of-fold predictions | 2. Naive_Bayes_SMS | Section 8 |
| *On standby* | Neural network (MLP) | None yet | Added only if its class notebook is released in time |

Adaptations to the class notebooks, the same everywhere: time-ordered folds from `ml2.splits` instead of shuffled
`KFold`, the shared preprocessing from `ml2.features`, and grid searches on a 30% sample of the 114k training rows.

**Metrics** (`ml2.evaluate`): mean absolute error (MAE), RMSE and median absolute error in basis points, and R².
Always reported separately for IG and HY and by maturity bucket, not only overall.

**Quality gate, fixed before any logged run** (`config.yaml`, checked by `ml2.evaluate.gate_report`): against
matrix pricing re-marked on every snapshot date, the best model must cut test MAE by at least 10% **and** the 95%
interval of that improvement must lie above zero, in **both** IG and HY. The interval comes from resampling whole
companies (1,000 draws), because a company's bonds move together. If no model passes, our recommendation is "keep
matrix pricing", and that is still a valid, honest report.

Why these numbers: replacing a transparent lookup with a model costs validation, governance and explaining marks to
auditors, so the gain must be material (10%); and with four test months and a few hundred companies per grade, a
smaller gain cannot be told apart from luck (the interval condition).

*Change log.* 17 Nov 2026, before any logged run and with test unopened: the baseline was tightened from matrix
pricing frozen at the training months to matrix pricing re-marked on each date, the way a desk marks it, and the
interval condition was added. A train/valid-only feasibility check showed the frozen matrix was too easy a bar.
The frozen version stays in the final table as a reference row.

## 7. Validation (`ml2.splits`)

**No random splits.** Bonds of the same company on the same date move together, and a bond's spread this month is
close to last month's. A random split puts near-copies of test rows into training and makes every model look
better than it is.

- **Partitions** (column `partition`): train 2024-01 to 2025-05 (17 month-ends), *gap* 2025-06, valid 2025-07 to
  2025-10, *gap* 2025-11, test 2025-12 to 2026-03. The gap months are left out to limit leakage.
- **Tuning:** `walk_forward_folds()`: train on earlier months, validate on later ones, inside the training window.
- **Two questions, two evaluations:**
  - *Value a bond that did not trade today* (the company is known): the chronological partitions above.
  - *Value a bond from a company the model has never seen:* `issuer_group_folds()`, so none of that company's
    bonds are in training. Harder, and where KNN versus boosting gets interesting.
- **Test partition:** opened once, at the end, for the final table. Nobody looks at it before.
- **Run log:** every run, including dropped variants, in `notebooks/RUN_LOG.md`. It feeds the technical annex.

## 8. Classification variant (technical annex)

Target `widens_25bp`: 1 if the bond's spread is 25 bp or more wider at the next month-end, 0 if not. Bin the
features for Categorical Naive Bayes, and run Gaussian Naive Bayes as a contrast. Compare Naive Bayes, KNN, tree,
random forest and boosting on ROC AUC and log loss against always predicting the base rate, and choose the decision
threshold on out-of-fold predictions as in the class notebook. A weak result is
fine: the point is to show the classifiers from the course, and to explain why Naive Bayes's independence
assumption hurts when features are correlated (for example `vol_20d` and `vol_60d`).

## 9. Roles and proposed issues

Six numbered roles. With five people, roles 4 and 5 merge (KNN and the annex fit together well). Everyone writes
their part of the annex and presents what they built.

| # | Role | Owner | Notebook | Issues |
|---|---|---|---|---|
| 1 | Data and integration | Miguel | `00_data_access`, `99_final` | P1, P2, P3, P13, P14, P18 |
| 2 | EDA and baseline | | `01_eda`, `02_baseline_matrix_pricing` | P4, P5 |
| 3 | Trees and ensembles | | `03_trees_ensembles` | P6, P7, P8 |
| 4 | KNN comparable bonds | | `04_knn_comparables` | P9, P10 |
| 5 | Classification annex (Naive Bayes) | | `06_annex_classification` | P11, P12 |
| 6 | Report and deck | | report, annex assembly, deck | P15, P16, P17 |
| | *Neural network: on standby* | | `05_neural_network`, empty until its class notebook exists | none yet |

If this proposal is chosen, each row below becomes a GitHub issue in a milestone "Bond pricing, submission Sun 22
Nov", labelled by type (`decision`, `infra`, `data`, `model`, `docs`) and by wave (`wave-0` to `wave-3`, the
day block it is due in). P numbers are placeholders until the issues are opened; the owners table then lists the
real numbers. Every issue closes through a pull request into `dev`, reviewed by one teammate.

| # | Issue | Role | Wave | Done when |
|---|---|---|---|---|
| P1 | Decision: agree roles, the IG/HY strata and the quality gate on the kickoff call | 1 | 0 (Wed 18) | Owners filled in above; decisions written here |
| P2 | Setup: scaffold, shared `ml2` package, config and CI on `dev` | 1 | 0 (Wed 18) | PRs #1 and #2 merged; `pytest` and `flake8` pass in CI |
| P3 | Data: group database, read-only logins and the parquet fallback | 1 | 0 (Wed 18) | Everyone runs `00_data_access` and sees the row counts by partition |
| P4 | Notebook: `01_eda`, the target, gaps, correlations and the IG/HY definition | 2 | 1 (Thu 19) | Figures in `reports/figures/`; the strata decision posted to the team |
| P5 | Notebook: `02_baseline_matrix_pricing`, the three baselines on valid | 2 | 1 (Thu 19) | Valid MAE for IG, HY and all in RUN_LOG; slide 7 figure saved |
| P6 | Model: regression tree with cost-complexity pruning | 3 | 1 (Thu 19) | Best alpha from time-ordered folds; valid MAE in RUN_LOG |
| P7 | Model: bagging and random forest, with feature importance | 3 | 2 (Fri 20) | Valid MAE in RUN_LOG; importance plot saved |
| P8 | Model: gradient boosting (XGBoost, LightGBM), tuned on a sample of train | 3 | 2 (Fri 20) | Best settings and valid MAE in RUN_LOG; gain over the gate baseline on valid |
| P9 | Model: KNN comparable bonds, scaling and the choice of k | 4 | 1 (Thu 19) | Error against k plotted; valid MAE in RUN_LOG |
| P10 | Model: bagged KNN and the "company never seen before" test | 4 | 2 (Fri 20) | Issuer-grouped folds result in RUN_LOG, next to the time-ordered one |
| P11 | Model: Naive Bayes on `widens_25bp` | 5 | 2 (Fri 20) | Recall, precision and AUC on valid in RUN_LOG |
| P12 | Model: Naive Bayes against the other classifiers (tree, forest, KNN) | 5 | 2 (Fri 20) | One comparison table and the annex section drafted |
| P13 | Decision: choose the best model on valid, with settings frozen on Fri 20 | 1 | 3 (Sat 21) | Choice and reasons in RUN_LOG before test is opened |
| P14 | Notebook: `99_final`, refit, score test once, `gate_report` and the headline table | 1 | 3 (Sat 21) | Runs top to bottom from the panel; gate verdict with 95% intervals |
| P15 | Docs: the 4-page report for the head of pricing | 6 | 3 (Sun 22) | Numbers only from `99_final`; a teammate has reviewed it |
| P16 | Docs: technical annex assembled from RUN_LOG and each owner's part | 6 | 3 (Sun 22) | Every model and every run in RUN_LOG covered, gate change stated |
| P17 | Docs: final deck and dry run | 6 | 3 (Sun 22) | Every member presents what they built, within 12 minutes |
| P18 | Release: submission zip (report, annex, database with data dictionary, notebook, deck) | 1 | 3 (Sun 22) | Zip checked against the brief's five items and uploaded before 23:59 |

## 10. Plan

| Day | Work | Output | Issues |
|---|---|---|---|
| Wed 18 | Project chosen, roles agreed, everyone set up with data access; EDA starts | Logins or parquet sent, environment working | P1 to P3 |
| Thu 19 | EDA and baselines (the baseline code is ready); first tree and KNN runs. Light day: ML Ops presentation | `01_eda`, `02_baseline`, first runs in the log | P4 to P6, P9 |
| Fri 20 | Ensembles, bagged KNN and the new-company test, Naive Bayes annex; model settings frozen by the evening | Validation results in the run log | P7, P8, P10 to P12 |
| Sat 21 | Morning: choose the best model on validation, then open the test partition **once**. Afternoon: report, annex and deck drafts | Final results table, drafts | P13, P14 |
| Sun 22 | Report, annex and deck finished; dry run, zip, submit before 23:59 | Submission | P15 to P18 |

## 11. Known limits (say them in the report, briefly)

- The reported yield is "yield to worst" for bonds the company can repay early (callable bonds); we have no call
  schedules, so those spreads carry some noise.
- The IG/HY grade flag is today's, not the one on each snapshot date, and it is noisy: some large banks' small
  retail notes are flagged HY. How we define the IG and HY strata is an open decision for the EDA.
- The last snapshot is 17 March 2026 (where the data ends), not a month-end.
- Two years of history, one interest-rate regime: results may not hold in a crisis.
