"""Unsupervised anomaly detectors: Isolation Forest and Local Outlier Factor.

Labels are never used to fit a model. They only enter through the tuning helpers,
which choose hyperparameters by validation PR-AUC.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from .data import AMOUNT, PCA_FEATURES
from .metrics import pr_auc

RANDOM_STATE = 42
FEATURES = [*PCA_FEATURES, "log_amount"]
IF_BASELINE_N_ESTIMATORS = 200
IF_N_ESTIMATORS_GRID = (100, 300)
IF_MAX_SAMPLES_GRID = (256, 1024, 4096, "auto")
LOF_SUBSAMPLE_SIZE = 50_000
LOF_N_NEIGHBORS_GRID = (20, 35, 50)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    if AMOUNT not in df.columns:
        raise ValueError(f"Column {AMOUNT!r} is required to build log_amount")
    out = df.copy()
    out["log_amount"] = np.log1p(out[AMOUNT])
    return out


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Scaler fitted on training rows only; apply its transform to every window."""
    if len(X_train) == 0:
        raise ValueError("Cannot fit a scaler on an empty training set")
    return StandardScaler().fit(X_train[FEATURES])


def build_isolation_forest(
    n_estimators: int = IF_BASELINE_N_ESTIMATORS, max_samples: int | str = "auto"
) -> IsolationForest:
    """Isolation Forest with default contamination; the decision threshold is not used."""
    return IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        random_state=RANDOM_STATE,
    )


def build_lof(n_neighbors: int = LOF_N_NEIGHBORS_GRID[0]) -> LocalOutlierFactor:
    """LOF in novelty mode so that it can score rows it was not fitted on."""
    return LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, contamination="auto")


def subsample_train(
    train: pd.DataFrame, size: int = LOF_SUBSAMPLE_SIZE, random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """Random subsample of training rows for LOF; returns all rows when train is smaller."""
    if size <= 0:
        raise ValueError("subsample size must be positive")
    if len(train) == 0:
        raise ValueError("Cannot subsample an empty training set")
    if len(train) <= size:
        return train
    return train.sample(n=size, random_state=random_state).sort_index()


def anomaly_scores(model, X) -> np.ndarray:
    """Higher means more anomalous: the negated sklearn score_samples value."""
    return -np.asarray(model.score_samples(X))


def _check_labels(y_val) -> np.ndarray:
    y = np.asarray(y_val)
    if y.sum() == 0:
        raise ValueError("Validation labels contain no frauds; PR-AUC is undefined")
    return y


def _param(value) -> int | str:
    return value if isinstance(value, str) else int(value)


def tune_isolation_forest(
    X_train,
    X_val,
    y_val,
    n_estimators_grid: Sequence[int] = IF_N_ESTIMATORS_GRID,
    max_samples_grid: Sequence[int | str] = IF_MAX_SAMPLES_GRID,
) -> tuple[dict, pd.DataFrame]:
    """Pick (n_estimators, max_samples) by validation PR-AUC. Ties go to the earlier grid entry."""
    y = _check_labels(y_val)
    rows = []
    for n_estimators in n_estimators_grid:
        for max_samples in max_samples_grid:
            model = build_isolation_forest(n_estimators, max_samples).fit(X_train)
            scores = anomaly_scores(model, X_val)
            rows.append(
                {
                    "n_estimators": int(n_estimators),
                    "max_samples": max_samples,
                    "val_pr_auc": pr_auc(y, scores),
                }
            )
    table = pd.DataFrame(rows)
    best = table.loc[table["val_pr_auc"].idxmax()]
    params = {
        "n_estimators": int(best["n_estimators"]),
        "max_samples": _param(best["max_samples"]),
    }
    return params, table


def tune_lof(
    X_train_sub,
    X_val,
    y_val,
    n_neighbors_grid: Sequence[int] = LOF_N_NEIGHBORS_GRID,
) -> tuple[int, pd.DataFrame]:
    """Pick n_neighbors by validation PR-AUC on a LOF fitted to the training subsample."""
    y = _check_labels(y_val)
    rows = []
    for n_neighbors in n_neighbors_grid:
        model = build_lof(n_neighbors).fit(X_train_sub)
        scores = anomaly_scores(model, X_val)
        rows.append({"n_neighbors": int(n_neighbors), "val_pr_auc": pr_auc(y, scores)})
    table = pd.DataFrame(rows)
    best = table.loc[table["val_pr_auc"].idxmax()]
    return int(best["n_neighbors"]), table
