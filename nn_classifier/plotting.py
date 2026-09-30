"""Figures, returned rather than shown.

Every function hands back the matplotlib figure instead of calling
`plt.show()`. A notebook renders it by being the last expression in a
cell; a script saves it. Calling `show()` inside would make the second
of those impossible.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


def plot_training_curves(history, level_columns: Optional[Sequence[str]] = None):
    """Loss and per-level accuracy against epoch.

    Look at the gap between the two loss curves rather than at either
    one. Training loss below validation loss and still falling while
    validation flattens is the model memorising part numbers, and the
    fix is upstream -- in `min_token_count` -- not in the optimiser.
    """
    import matplotlib.pyplot as plt

    epochs = range(1, len(history.train_loss) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history.train_loss, marker="o", label="train")
    axes[0].plot(epochs, history.val_loss, marker="o", label="validation")
    if history.best_epoch:
        axes[0].axvline(history.best_epoch, ls="--", c="grey", lw=1,
                        label=f"best (epoch {history.best_epoch})")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()

    n_levels = len(history.val_acc[0]) if history.val_acc else 0
    names = list(level_columns) if level_columns else [f"L{i+1}" for i in range(n_levels)]
    for i in range(n_levels):
        axes[1].plot(epochs, [row[i] for row in history.val_acc],
                     marker="o", label=names[i])
    axes[1].set_title("Validation accuracy by level")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    fig.tight_layout()
    return fig


def plot_reliability(report, title: Optional[str] = None):
    """Section D's two panels: reliability diagram and confidence histogram.

    The diagonal is perfect calibration. A curve below it means the
    model claims more confidence than it earns -- the usual case, and
    what the temperature is there to pull back. The histogram beside
    it says whether the curve is worth trusting: a bin with nine rows
    in it is noise, not evidence.
    """
    import matplotlib.pyplot as plt

    table = report.reliability
    valid = table[table["count"] > 0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    axes[0].plot(valid["mean_confidence"], valid["accuracy"], marker="o", label="model")
    axes[0].set_title(title or f"Reliability -- {report.level} (ECE={report.ece:.4f})")
    axes[0].set_xlabel("mean predicted confidence")
    axes[0].set_ylabel("accuracy")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].legend()

    if report.confidence is not None:
        axes[1].hist(report.confidence, bins=20)
    axes[1].set_title(f"Top-1 confidence -- {report.level}")
    axes[1].set_xlabel("confidence")
    axes[1].set_ylabel("count")

    fig.tight_layout()
    return fig


def plot_coverage_curve(report, target_accuracy: Optional[float] = None):
    """Coverage and accuracy against threshold, on twin axes.

    Where the threshold decision actually gets made. The two curves
    move against each other; the question is where the accuracy line
    crosses the bar you have been given, and how much coverage is
    left at that point.
    """
    import matplotlib.pyplot as plt

    table = report.coverage
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(table["threshold"], table["coverage"], marker="o", label="coverage")
    ax.set_xlabel("confidence threshold")
    ax.set_ylabel("coverage (share auto-assigned)")
    ax.set_ylim(0, 1)

    ax2 = ax.twinx()
    ax2.plot(table["threshold"], table["accuracy_at_threshold"],
             marker="s", color="tab:orange", label="accuracy")
    ax2.set_ylabel("accuracy of auto-assigned rows")
    ax2.set_ylim(0, 1)
    if target_accuracy is not None:
        ax2.axhline(target_accuracy, ls="--", c="grey", lw=1)

    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="lower left")
    ax.set_title(f"Coverage vs accuracy -- {report.level}")
    fig.tight_layout()
    return fig


def plot_reliability_grid(reports: Dict[str, object], n_cols: int = 2):
    """One reliability panel per level, on a shared scale.

    Side by side because the levels usually fail differently: the
    coarse heads sit close to the diagonal and the deepest one sags
    well below it. Seeing them on one figure is what justifies fitting
    a separate temperature per level rather than one for the model.
    """
    import matplotlib.pyplot as plt

    items = list(reports.items())
    n = len(items)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows),
                             squeeze=False)
    for ax, (name, rep) in zip(axes.ravel(), items):
        valid = rep.reliability[rep.reliability["count"] > 0]
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.plot(valid["mean_confidence"], valid["accuracy"], marker="o")
        ax.set_title(f"{name} (ECE={rep.ece:.4f}, acc={rep.accuracy:.3f})")
        ax.set_xlabel("confidence")
        ax.set_ylabel("accuracy")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_accuracy_by_bucket(buckets: pd.DataFrame, title: str = "Accuracy by confidence"):
    """Bars for accuracy, a line for how many rows sit in each band."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(buckets["bucket"], buckets["accuracy"])
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)

    ax2 = ax.twinx()
    ax2.plot(buckets["bucket"], buckets["count"], color="tab:orange", marker="o")
    ax2.set_ylabel("rows")
    fig.tight_layout()
    return fig


def plot_per_level_accuracy(per_level: pd.DataFrame, prefix: Optional[pd.DataFrame] = None):
    """Per-level accuracy, optionally with the cumulative path accuracy on top.

    The gap between the two lines is the cost of coherence: how much
    of the per-level accuracy survives the requirement that the levels
    agree with each other.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(per_level["level"], per_level["accuracy"], marker="o", label="per level")
    if prefix is not None and len(prefix):
        ax.plot(prefix["level"], prefix["prefix_accuracy"], marker="s",
                label="cumulative path")
    ax.set_ylim(0, 1)
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy by level")
    ax.tick_params(axis="x", rotation=20)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_top_confusions(confusions: pd.DataFrame, n: int = 15, title: str = "Top confusions"):
    """Horizontal bars, most frequent error pair at the top."""
    import matplotlib.pyplot as plt

    top = confusions.head(n).iloc[::-1]
    labels = [f"{a}  ->  {p}" for a, p in zip(top["actual"], top["predicted"])]
    fig, ax = plt.subplots(figsize=(9, max(3, 0.35 * len(top))))
    ax.barh(labels, top["count"])
    ax.set_xlabel("errors")
    ax.set_title(title)
    fig.tight_layout()
    return fig
