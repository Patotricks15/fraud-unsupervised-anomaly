# fraud-unsupervised-anomaly (FD03)

**Question:** Can we flag card fraud without labels? Isolation Forest and Local Outlier Factor (LOF) are fitted without using `Class`. Labels only tune hyperparameters on validation PR-AUC and evaluate after scoring.

## Data

- Credit Card Fraud Detection, Kaggle `mlg-ulb/creditcardfraud` (real anonymized transactions, September 2013, 284,807 rows, 492 frauds). The same dataset as [FD01](../fraud-logistic-baseline/README.md).
- The raw file is read in place and is not committed. See [data/README.md](data/README.md).
- `Time` covers about two days, so the time split spans two days only.

## Method

- Features: `V1`-`V28` (PCA components, used as provided) and `log(1 + Amount)`. `Time` and `Class` are excluded from features.
- Split (time-ordered, no shuffle): earliest 70% of `Time` for training, next 15% for validation, last 15% for test. The test window is scored once per model.
- Isolation Forest: `IsolationForest(random_state=42)`, raw features. Baseline: `n_estimators=200`, `max_samples="auto"`, default contamination. Tuned: `n_estimators` in {100, 300} and `max_samples` in {256, 1024, 4096, "auto"}, chosen by validation PR-AUC.
- LOF: `LocalOutlierFactor(novelty=True, contamination="auto")` on features scaled by a `StandardScaler` fitted on training rows only. LOF is fitted on a random subsample of 50,000 training rows (`random_state=42`, or all training rows if fewer), because LOF on the full training set is too heavy. `n_neighbors` in {20, 35, 50} is chosen by validation PR-AUC.
- Anomaly score: `-score_samples`, so higher means more anomalous.
- Metrics: PR-AUC (primary), ROC-AUC, recall at a 1% alert rate.

## Results

Real data, test window scored once after tuning on validation. Validation: 56 frauds. Test: 52 frauds.

| Model | Window | PR-AUC | ROC-AUC | Recall @ 1% alerts |
|---|---|---|---|---|
| FD01 logistic baseline (tuned C) | Test | 0.690 | 0.976 | 0.808 |
| Isolation Forest, baseline | Validation | 0.037 | 0.951 | 0.375 |
| Isolation Forest, tuned (300 trees, max_samples 4096) | Validation | 0.049 | 0.954 | 0.536 |
| Isolation Forest, tuned | Test | 0.078 | 0.936 | 0.596 |
| LOF, tuned (n_neighbors 50, subsample 50,000) | Validation | 0.050 | 0.731 | 0.393 |
| LOF, tuned (n_neighbors 50, subsample 50,000) | Test | 0.041 | 0.774 | 0.308 |

Reading: unsupervised scores rank fraud far below the supervised baseline on PR-AUC (0.078 vs 0.690 on test). Isolation Forest is the better unsupervised detector here. LOF has the weakest ROC-AUC on validation (0.731). Neither reaches the FD01 PR-AUC. Whether they catch frauds that FD01 misses was not measured, so the second clause of the decision rule is still open.

Figures (written by the notebook): `reports/figures/fd03_score_hist.png` (anomaly score by class on test, log y-axis) and `reports/figures/fd03_pr_curve.png` (PR curves for both models on test).

## Decision rule

Flag the top 1% of test transactions by anomaly score. Labels are used only to validate after scoring. An unsupervised detector is worth keeping only if its test PR-AUC or recall at 1% alerts clears the FD01 logistic baseline, or if it catches frauds the supervised model misses.

## Limitation

- Anomaly detectors flag any unusual spending, including legitimate transactions. Fraud is about 0.17% of rows, so most alerts can be false positives.
- LOF is fitted on a 50,000-row subsample of training data, so its scores depend on that sample.
- Features are anonymized PCA components, so alerts cannot be checked against business rules. The two-day window is too short to show drift.

## Reproduce

```bash
pip install -e ".[dev]"
pytest
# On Kaggle: attach mlg-ulb/creditcardfraud, upload the notebook, Run All.
# Locally (development only): export FRAUD_DATA_PATH=/path/to/creditcard.csv
make notebook
```

## Layout

```text
src/fraud_unsupervised/       data loading, time split, metrics, models, plots
notebooks/                    jupytext source (.py) and paired notebook (.ipynb)
tests/                        split, metrics, scaler leakage, LOF subsample, score direction, tuning
data/                         raw data (not committed) and derived test scores (git-ignored)
reports/figures/              exported figures
```
