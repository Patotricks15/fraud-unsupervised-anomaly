"""Figures for the FD03 report. Each title names the dataset, split and model."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402


def save(fig, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_score_histograms(y_true, scores_by_model: dict, title: str, path: str | Path, bins: int = 60) -> Path:
    """One panel per model: anomaly score histogram split by Class, log y-axis."""
    y = np.asarray(y_true)
    fig, axes = plt.subplots(1, len(scores_by_model), figsize=(6 * len(scores_by_model), 4.5), squeeze=False)
    for ax, (name, scores) in zip(axes[0], scores_by_model.items()):
        s = np.asarray(scores)
        edges = np.histogram_bin_edges(s, bins=bins)
        ax.hist(s[y == 0], bins=edges, alpha=0.6, label="Class 0 (legitimate)", log=True)
        ax.hist(s[y == 1], bins=edges, alpha=0.8, label="Class 1 (fraud)", log=True)
        ax.set_xlabel("Anomaly score (higher = more anomalous)")
        ax.set_ylabel("Count (log scale)")
        ax.set_title(name)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    return save(fig, path)


def plot_pr_curves(y_true, scores_by_model: dict, title: str, path: str | Path) -> Path:
    """Precision-recall curves on one axis, with PR-AUC (average precision) in the legend."""
    y = np.asarray(y_true)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    baseline = y.mean()
    ax.axhline(baseline, color="grey", linestyle="--", label=f"Random ({baseline:.4f})")
    for name, scores in scores_by_model.items():
        precision, recall, _ = precision_recall_curve(y, scores)
        ap = average_precision_score(y, scores)
        ax.plot(recall, precision, label=f"{name} (PR-AUC {ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    return save(fig, path)
