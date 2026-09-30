"""Metrics, computed the same way for every model family.

Kept deliberately plain: functions over frames and arrays, nothing
that needs a fitted model. That is what lets the same numbers be
produced for a torch network and an sklearn estimator, which is the
only way the comparison between them means anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def per_level_accuracy(
    frame: pd.DataFrame,
    truth: pd.DataFrame,
    level_columns: Sequence[str],
) -> pd.DataFrame:
    """Accuracy at each level, independent of the others."""
    rows = []
    for col in level_columns:
        pred_col = f"Predicted {col}"
        if pred_col not in frame.columns:
            continue
        pred = frame[pred_col].astype(str).to_numpy()
        actual = truth[col].astype(str).to_numpy()
        rows.append({
            "level": col,
            "accuracy": float((pred == actual).mean()),
            "n_classes": int(truth[col].nunique()),
        })
    return pd.DataFrame(rows)


def prefix_accuracy(
    frame: pd.DataFrame,
    truth: pd.DataFrame,
    level_columns: Sequence[str],
) -> pd.DataFrame:
    """Accuracy of the whole path down to each depth.

    Falls monotonically with depth, by construction, and the shape of
    that fall is the useful part: a big drop between two levels is
    where the taxonomy stops being learnable from a description, and
    is usually where the cascade should be told to stop.
    """
    level_columns = list(level_columns)
    rows = []
    running = np.ones(len(frame), dtype=bool)
    for depth, col in enumerate(level_columns, start=1):
        pred_col = f"Predicted {col}"
        if pred_col not in frame.columns:
            continue
        match = (
            frame[pred_col].astype(str).to_numpy()
            == truth[col].astype(str).to_numpy()
        )
        running = running & match
        rows.append({
            "depth": depth,
            "level": col,
            "prefix_accuracy": float(running.mean()),
        })
    return pd.DataFrame(rows)


def path_accuracy(
    frame: pd.DataFrame,
    truth: pd.DataFrame,
    level_columns: Sequence[str],
) -> float:
    """Share of rows whose entire path is right."""
    correct = np.ones(len(frame), dtype=bool)
    for col in level_columns:
        pred_col = f"Predicted {col}"
        if pred_col not in frame.columns:
            continue
        correct &= (
            frame[pred_col].astype(str).to_numpy()
            == truth[col].astype(str).to_numpy()
        )
    return float(correct.mean()) if len(correct) else float("nan")


def top_k_accuracy(
    probs: np.ndarray,
    targets: np.ndarray,
    k: int = 5,
    batch_size: Optional[int] = None,
) -> float:
    """Is the right answer anywhere in the top k?

    Worth reporting next to top-1 on a many-class problem. A large gap
    means the model has the right neighbourhood and is losing on the
    final discrimination -- a different problem, with different fixes,
    than a model that is simply lost.

    Batched. `np.argpartition(-p, ...)` negates the whole array first,
    which on 526,670 x 3,473 is a 6.8 GB copy before argpartition has
    allocated its own workspace -- for one scalar.
    """
    y = np.asarray(targets)
    n_rows, n_cols = probs.shape
    k = min(k, n_cols)
    batch = batch_size or max(1, int((64 * 1024 ** 2) // max(n_cols * 4, 1)))

    hits = 0
    for lo in range(0, n_rows, batch):
        hi = min(lo + batch, n_rows)
        block = np.array(probs[lo:hi], dtype=np.float32)
        np.negative(block, out=block)
        top = np.argpartition(block, kth=k - 1, axis=1)[:, :k]
        hits += int((top == y[lo:hi, None]).any(axis=1).sum())
        del block, top
    return float(hits / max(n_rows, 1))


def top_confusions(
    frame: pd.DataFrame,
    truth: pd.DataFrame,
    level: str,
    n: int = 20,
) -> pd.DataFrame:
    """The most frequent wrong (actual -> predicted) pairs at one level.

    Usually the most actionable output in this module. A confusion
    that dominates the list is normally a taxonomy problem -- two
    categories that no description could distinguish -- rather than a
    model problem, and no amount of tuning will fix it.
    """
    pred_col = f"Predicted {level}"
    pred = frame[pred_col].astype(str).to_numpy()
    actual = truth[level].astype(str).to_numpy()
    wrong = pred != actual
    if not wrong.any():
        return pd.DataFrame(columns=["actual", "predicted", "count", "share_of_errors"])
    pairs = pd.DataFrame({"actual": actual[wrong], "predicted": pred[wrong]})
    counts = pairs.value_counts().reset_index(name="count")
    counts["share_of_errors"] = counts["count"] / int(wrong.sum())
    return counts.head(n)


def accuracy_by_bucket(
    confidence: np.ndarray,
    correct: np.ndarray,
    bucket_width: float = 0.1,
) -> pd.DataFrame:
    """Accuracy and volume by confidence band."""
    conf = np.asarray(confidence, dtype=np.float64)
    corr = np.asarray(correct, dtype=np.float64)
    n_bins = max(1, int(round(1.0 / bucket_width)))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.clip(np.digitize(conf, edges) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = ids == b
        rows.append({
            "bucket": f"[{edges[b]:.1f}, {edges[b+1]:.1f})",
            "count": int(mask.sum()),
            "share": float(mask.mean()) if len(conf) else 0.0,
            "accuracy": float(corr[mask].mean()) if mask.any() else float("nan"),
        })
    return pd.DataFrame(rows)


@dataclass
class EvaluationResult:
    """Everything a model notebook should print before it saves anything."""

    per_level: pd.DataFrame
    prefix: pd.DataFrame
    path_accuracy: float
    confusions: Dict[str, pd.DataFrame] = field(default_factory=dict)
    buckets: Optional[pd.DataFrame] = None
    top_k: Dict[str, float] = field(default_factory=dict)
    n_rows: int = 0

    def summary(self) -> str:
        lines = [f"rows evaluated: {self.n_rows:,}",
                 f"full-path accuracy: {self.path_accuracy:.4f}"]
        for _, r in self.per_level.iterrows():
            lines.append(
                f"  {r['level']}: {r['accuracy']:.4f} "
                f"over {int(r['n_classes']):,} classes"
            )
        for name, value in self.top_k.items():
            lines.append(f"  {name}: {value:.4f}")
        return "\n".join(lines)


def evaluate(
    frame: pd.DataFrame,
    truth: pd.DataFrame,
    level_columns: Sequence[str],
    level_probs: Optional[Sequence[np.ndarray]] = None,
    level_targets: Optional[np.ndarray] = None,
    confusion_levels: Optional[Sequence[str]] = None,
    bucket_width: float = 0.1,
    confidence_col: str = "Confidence",
    verbose: bool = True,
) -> EvaluationResult:
    """The standard evaluation, for any model that produced a prediction frame."""
    level_columns = list(level_columns)

    correct_path = np.ones(len(frame), dtype=np.float32)
    for col in level_columns:
        pred_col = f"Predicted {col}"
        if pred_col in frame.columns:
            correct_path *= (
                frame[pred_col].astype(str).to_numpy()
                == truth[col].astype(str).to_numpy()
            ).astype(np.float32)

    buckets = None
    if confidence_col in frame.columns:
        buckets = accuracy_by_bucket(
            frame[confidence_col].to_numpy(), correct_path, bucket_width
        )

    topk: Dict[str, float] = {}
    if level_probs is not None and level_targets is not None:
        for i, col in enumerate(level_columns[: len(level_probs)]):
            topk[f"{col} top-5"] = top_k_accuracy(level_probs[i], level_targets[:, i], 5)

    confusions = {
        col: top_confusions(frame, truth, col)
        for col in (confusion_levels or [level_columns[-1]])
        if f"Predicted {col}" in frame.columns
    }

    result = EvaluationResult(
        per_level=per_level_accuracy(frame, truth, level_columns),
        prefix=prefix_accuracy(frame, truth, level_columns),
        path_accuracy=float(correct_path.mean()) if len(frame) else float("nan"),
        confusions=confusions,
        buckets=buckets,
        top_k=topk,
        n_rows=int(len(frame)),
    )
    if verbose:
        print(result.summary())
    return result


def compare_modes(results: Dict[str, EvaluationResult]) -> pd.DataFrame:
    """Side-by-side table of several runs -- joint modes, or model families.

    The comparison this repo exists to make. Run the same data through
    `independent`, `conditional` and `None` and read the rows against
    each other; the per-level accuracies usually barely move while the
    path accuracy does, which is the whole argument for constraining
    predictions to real paths.
    """
    rows = []
    for name, res in results.items():
        row = {"run": name, "path_accuracy": res.path_accuracy, "n_rows": res.n_rows}
        for _, r in res.per_level.iterrows():
            row[str(r["level"])] = r["accuracy"]
        rows.append(row)
    return pd.DataFrame(rows)
