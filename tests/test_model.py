import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

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


def _data(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(size=(n, 28)), columns=[f"V{i}" for i in range(1, 29)])
    df["Amount"] = rng.uniform(1, 500, n)
    df["Class"] = (rng.uniform(size=n) < 0.05).astype(int)
    return add_features(df)


def _cluster_with_outlier(seed: int = 3):
    """200 inliers around the origin plus one far-away point in the last row."""
    rng = np.random.default_rng(seed)
    inliers = rng.normal(size=(200, 2))
    X = np.vstack([inliers, [[15.0, 15.0]]])
    return X, inliers


def test_scaler_is_fit_on_training_rows_only():
    df = _data()
    train, test = df.iloc[:300], df.iloc[300:]
    scaler = fit_scaler(train)
    np.testing.assert_allclose(scaler.mean_, train[FEATURES].mean().to_numpy())
    assert not np.allclose(scaler.mean_, df[FEATURES].mean().to_numpy())
    assert len(test) == 100


def test_scaler_rejects_empty_training_set():
    with pytest.raises(ValueError):
        fit_scaler(_data().iloc[:0])


def test_log_amount_is_added_without_changing_input():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"Amount": rng.uniform(1, 500, 10)})
    out = add_features(df)
    assert "log_amount" not in df.columns
    np.testing.assert_allclose(out["log_amount"], np.log1p(df["Amount"]))


def test_lof_subsample_is_drawn_from_train_and_is_deterministic():
    df = _data(3000)
    train = df.iloc[:2000]
    first = subsample_train(train, size=500, random_state=42)
    second = subsample_train(train, size=500, random_state=42)
    assert len(first) == 500
    assert set(first.index) <= set(train.index)
    pd.testing.assert_frame_equal(first, second)


def test_lof_subsample_returns_all_train_rows_when_smaller():
    train = _data(100).iloc[:80]
    pd.testing.assert_frame_equal(subsample_train(train, size=50_000), train)


def test_anomaly_score_is_higher_for_outlier_in_isolation_forest_and_lof():
    X, inliers = _cluster_with_outlier()
    outlier_idx = len(X) - 1

    forest = build_isolation_forest(n_estimators=200).fit(inliers)
    if_scores = anomaly_scores(forest, X)
    assert if_scores[outlier_idx] > np.quantile(if_scores[:-1], 0.95)

    lof = build_lof(n_neighbors=20).fit(inliers)
    lof_scores = anomaly_scores(lof, X)
    assert lof_scores[outlier_idx] > np.quantile(lof_scores[:-1], 0.95)


def test_isolation_forest_tuning_picks_best_validation_pr_auc():
    rng = np.random.default_rng(4)
    X_train = rng.normal(size=(600, 3))
    X_val = np.vstack([rng.normal(size=(180, 3)), rng.normal(loc=6, size=(20, 3))])
    y_val = np.array([0] * 180 + [1] * 20)
    params, table = tune_isolation_forest(
        X_train, X_val, y_val, n_estimators_grid=(20, 40), max_samples_grid=(64, "auto")
    )
    best = table.loc[table["val_pr_auc"].idxmax()]
    assert params == {"n_estimators": int(best["n_estimators"]), "max_samples": best["max_samples"]}
    assert len(table) == 4


def test_lof_tuning_picks_best_validation_pr_auc():
    rng = np.random.default_rng(5)
    X_train = rng.normal(size=(300, 2))
    X_val = np.vstack([rng.normal(size=(90, 2)), [[8.0, 8.0], [9.0, -8.0]]])
    y_val = np.array([0] * 90 + [1, 1])
    k, table = tune_lof(X_train, X_val, y_val, n_neighbors_grid=(5, 10, 20))
    expected = {
        n: average_precision_score(y_val, anomaly_scores(build_lof(n).fit(X_train), X_val))
        for n in (5, 10, 20)
    }
    assert k == max(expected, key=expected.get)
    assert dict(zip(table["n_neighbors"], table["val_pr_auc"])) == pytest.approx(expected)


def test_tuning_helpers_take_no_test_window_inputs():
    for func in (tune_isolation_forest, tune_lof):
        names = inspect.signature(func).parameters
        assert not [name for name in names if "test" in name.lower()]


def test_lof_tuning_is_stable_across_repeated_calls():
    rng = np.random.default_rng(6)
    X_train = rng.normal(size=(300, 2))
    X_val = np.vstack([rng.normal(size=(90, 2)), [[8.0, 8.0], [9.0, -8.0]]])
    y_val = np.array([0] * 90 + [1, 1])
    first, _ = tune_lof(X_train, X_val, y_val, n_neighbors_grid=(5, 10, 20))
    second, _ = tune_lof(X_train, X_val, y_val, n_neighbors_grid=(5, 10, 20))
    assert first == second


def test_tuning_rejects_validation_without_frauds():
    X = np.zeros((10, 2))
    with pytest.raises(ValueError):
        tune_lof(X, X, np.zeros(10, dtype=int), n_neighbors_grid=(2,))
