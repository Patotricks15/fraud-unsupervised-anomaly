# %% [markdown]
# # FD03: Unsupervised anomaly detection
# Source for the FD03 notebook. Convert with `jupytext --to ipynb` and run on Kaggle, or set `FRAUD_DATA_PATH` locally.
# Question: can we flag card fraud without labels? Isolation Forest and Local Outlier Factor are fitted without labels.
# Labels are used only to tune on validation PR-AUC and to evaluate after scoring.
# Dataset: Credit Card Fraud Detection (`mlg-ulb/creditcardfraud`), time-ordered split 70/15/15.

# %%
from pathlib import Path

import pandas as pd

from fraud_unsupervised import plots
from fraud_unsupervised.data import load_creditcard, time_split
from fraud_unsupervised.metrics import metric_summary, pr_auc
from fraud_unsupervised.model import (
    FEATURES,
    add_features,
    anomaly_scores,
    build_isolation_forest,
    build_lof,
    fit_scaler,
    subsample_train,
    tune_isolation_forest,
    tune_lof,
)

IN_NOTEBOOKS = Path.cwd().name == "notebooks"
FIG_DIR = Path("../reports/figures") if IN_NOTEBOOKS else Path("reports/figures")
DATA_DIR = Path("../data") if IN_NOTEBOOKS else Path("data")
ALERT_RATE = 0.01

# %%
df = add_features(load_creditcard())
split = time_split(df)
y = df["Class"]
print(f"rows: {len(df):,}  frauds: {int(y.sum())}")
print(f"train end (Time): {split.train_end:,.0f}s  val end: {split.val_end:,.0f}s")
for name, mask in (("train", split.train), ("val", split.val), ("test", split.test)):
    print(f"{name}: {int(mask.sum()):,} rows, {int(y[mask].sum())} frauds")

X_train = df.loc[split.train, FEATURES]
X_val = df.loc[split.val, FEATURES]
X_test = df.loc[split.test, FEATURES]
y_val = y[split.val]
y_test = y[split.test]

# %% [markdown]
# ## Scaling for LOF (fit on train features only)

# %%
scaler = fit_scaler(X_train)
X_train_s = scaler.transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

# %% [markdown]
# ## Isolation Forest baseline (default contamination and max_samples) on validation

# %%
baseline_if = build_isolation_forest(n_estimators=200, max_samples="auto").fit(X_train)
baseline_val_scores = anomaly_scores(baseline_if, X_val)
print(metric_summary(y_val, baseline_val_scores, ALERT_RATE))

# %% [markdown]
# ## Tune Isolation Forest on validation PR-AUC only

# %%
if_params, if_table = tune_isolation_forest(X_train, X_val, y_val)
print(if_table)
print(f"chosen Isolation Forest params: {if_params}")

# %%
final_if = build_isolation_forest(**if_params).fit(X_train)
if_val_scores = anomaly_scores(final_if, X_val)
print("validation, tuned IF:", metric_summary(y_val, if_val_scores, ALERT_RATE))

# %% [markdown]
# ## Local Outlier Factor: fit on a training subsample, tune n_neighbors on validation
# LOF on all training rows is too heavy, so it is fitted on a random subsample of at most 50,000 train rows.

# %%
train_sub = subsample_train(df.loc[split.train])
X_sub_s = scaler.transform(train_sub[FEATURES])
print(f"LOF training subsample: {len(train_sub):,} of {int(split.train.sum()):,} train rows")

# %%
k, lof_table = tune_lof(X_sub_s, X_val_s, y_val)
print(lof_table)
print(f"chosen LOF n_neighbors: {k}")

# %%
final_lof = build_lof(k).fit(X_sub_s)
lof_val_scores = anomaly_scores(final_lof, X_val_s)
print("validation, tuned LOF:", metric_summary(y_val, lof_val_scores, ALERT_RATE))

# %% [markdown]
# ## Validation comparison (selection only)

# %%
val_rows = {
    "IF baseline": metric_summary(y_val, baseline_val_scores, ALERT_RATE),
    "IF tuned": metric_summary(y_val, if_val_scores, ALERT_RATE),
    "LOF tuned": metric_summary(y_val, lof_val_scores, ALERT_RATE),
}
print(pd.DataFrame(val_rows).T)

# %% [markdown]
# ## Final scores on the test window (scored once, reported)

# %%
test_if_scores = anomaly_scores(final_if, X_test)
test_lof_scores = anomaly_scores(final_lof, X_test_s)
test_y = y_test.to_numpy()
test_rows = {
    "IF tuned": metric_summary(test_y, test_if_scores, ALERT_RATE),
    "LOF tuned": metric_summary(test_y, test_lof_scores, ALERT_RATE),
}
print(pd.DataFrame(test_rows).T)

# %%
title_suffix = (
    f"creditcardfraud, time test split (last 15% of Time), "
    f"IF n_estimators={if_params['n_estimators']} max_samples={if_params['max_samples']}, "
    f"LOF n_neighbors={k} on {len(train_sub):,} train rows"
)
plots.plot_score_histograms(
    test_y,
    {"Isolation Forest (tuned)": test_if_scores, f"LOF (k={k})": test_lof_scores},
    f"Anomaly score by class, {title_suffix}",
    FIG_DIR / "fd03_score_hist.png",
)
plots.plot_pr_curves(
    test_y,
    {"Isolation Forest (tuned)": test_if_scores, f"LOF (k={k})": test_lof_scores},
    f"PR curve, {title_suffix}",
    FIG_DIR / "fd03_pr_curve.png",
)

# %%
DATA_DIR.mkdir(parents=True, exist_ok=True)
scores_out = pd.DataFrame(
    {
        "Time": df.loc[split.test, "Time"].to_numpy(),
        "Class": test_y,
        "if_score": test_if_scores,
        "lof_score": test_lof_scores,
    }
)
scores_out.to_csv(DATA_DIR / "fd03_test_scores.csv", index=False)
print(f"wrote {len(scores_out):,} test scores to {DATA_DIR / 'fd03_test_scores.csv'}")
print(f"test PR-AUC  IF: {pr_auc(test_y, test_if_scores):.4f}  LOF: {pr_auc(test_y, test_lof_scores):.4f}")
