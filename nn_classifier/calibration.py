"""Turning a confidence number into an accuracy you can act on.

This is section D of the draft notebooks, generalised. The argument
there is the one that matters: on unlabelled data you cannot measure
accuracy, so you characterise the confidence -> correctness
relationship on validation *first*, then reuse the threshold you chose
on rows you will never be able to check.

A model's raw softmax is not that relationship. Label smoothing
compresses it, a large output layer inflates it, and a network trained
to convergence is systematically over-confident regardless. So there
are two steps, and they answer different questions:

  fit a calibrator   makes the number mean something
                     (temperature for a network, isotonic for an
                      sklearn estimator with no logits to scale)
  validate it        reliability diagram, ECE, and a coverage/accuracy
                     table -- does it now mean what it claims, and
                     where should the review threshold sit

The second step is not optional. A calibrator fitted and never checked
is a worse position than no calibrator, because the number now looks
trustworthy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

EPS = 1e-12


# ---------------------------------------------------------------------------
# Calibrators
# ---------------------------------------------------------------------------

@dataclass
class TemperatureScaler:
    """One scalar dividing the logits before the softmax.

    The whole model, and that is the appeal: a single parameter cannot
    change which class wins, so accuracy is untouched and only the
    confidence moves. T > 1 means the model was over-confident, which
    is the normal finding.

    Fitted with LBFGS against plain cross-entropy -- not the smoothed
    loss used in training. Smoothing is a regulariser; fitting the
    temperature against it would calibrate toward the smoothed target
    rather than toward being right.
    """

    temperature: float = 1.0
    fitted: bool = False

    def fit(self, logits, targets, max_iter: int = 100, verbose: bool = True):
        from .deps import auto_install

        auto_install("torch", purpose="temperature scaling")

        import torch
        import torch.nn as nn

        lg = torch.as_tensor(np.asarray(logits), dtype=torch.float32)
        y = torch.as_tensor(np.asarray(targets), dtype=torch.long)

        log_t = torch.zeros(1, requires_grad=True)   # optimise log T, so T > 0
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = criterion(lg / torch.exp(log_t), y)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float(torch.exp(log_t).item())
        self.fitted = True
        if verbose:
            direction = "over" if self.temperature > 1 else "under"
            print(
                f"[calib] temperature = {self.temperature:.4f} "
                f"(model was {direction}-confident)"
            )
        return self

    def transform(self, logits) -> np.ndarray:
        """Calibrated probabilities from raw logits."""
        lg = np.asarray(logits, dtype=np.float32) / max(self.temperature, EPS)
        lg = lg - lg.max(axis=1, keepdims=True)
        e = np.exp(lg)
        return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


@dataclass
class IsotonicConfidenceScaler:
    """A monotone map from top-1 confidence to observed accuracy.

    What to use when there are no logits to divide -- an sklearn
    `predict_proba` is already a probability, and rescaling it the way
    temperature does is not available.

    It calibrates the *top-1 confidence*, not the full distribution:
    it learns 'when this model says 0.8, it is right 0.64 of the
    time' and rewrites 0.8 as 0.64. The runner-up probabilities are
    then rescaled to keep the row summing to one. That is enough for
    thresholding and for the reliability diagram, which is what this
    number is used for, but it is not a full multiclass calibration
    and should not be read as one.
    """

    calibrator: object = None
    fitted: bool = False

    def fit(self, probs, targets, verbose: bool = True):
        from .deps import auto_install

        auto_install("scikit-learn", purpose="isotonic calibration")

        from sklearn.isotonic import IsotonicRegression

        p = np.asarray(probs, dtype=np.float32)
        y = np.asarray(targets)
        conf = p.max(axis=1)
        correct = (p.argmax(axis=1) == y).astype(np.float32)

        self.calibrator = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds="clip"
        ).fit(conf, correct)
        self.fitted = True
        if verbose:
            before = float(np.abs(conf.mean() - correct.mean()))
            after = float(np.abs(
                self.calibrator.predict(conf).mean() - correct.mean()
            ))
            print(
                f"[calib] isotonic fitted; mean confidence gap "
                f"{before:.4f} -> {after:.4f}"
            )
        return self

    def transform(self, probs) -> np.ndarray:
        p = np.asarray(probs, dtype=np.float32).copy()
        if not self.fitted:
            return p
        top_idx = p.argmax(axis=1)
        conf = p.max(axis=1)
        new_conf = np.clip(self.calibrator.predict(conf), EPS, 1.0 - EPS)

        rows = np.arange(len(p))
        rest = 1.0 - conf
        scale = np.where(rest > EPS, (1.0 - new_conf) / np.clip(rest, EPS, None), 0.0)
        p *= scale[:, None]
        p[rows, top_idx] = new_conf
        return p.astype(np.float32)


class IdentityScaler:
    """No calibration. Present so the call sites do not need a branch."""

    fitted = True
    temperature = 1.0

    def fit(self, *args, **kwargs):
        return self

    def transform(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 2 and not np.allclose(x.sum(axis=1), 1.0, atol=1e-3):
            # Logits were handed in; softmax them so the return type is
            # always probabilities, whatever the method.
            x = x - x.max(axis=1, keepdims=True)
            e = np.exp(x)
            return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)
        return x


def make_calibrator(method: Optional[str]):
    if method is None or method == "none":
        return IdentityScaler()
    if method == "temperature":
        return TemperatureScaler()
    if method == "isotonic":
        return IsotonicConfidenceScaler()
    raise ValueError(
        f"Unknown calibration method: {method!r}; expected 'temperature', "
        "'isotonic' or None."
    )


@dataclass
class LevelCalibrators:
    """One calibrator per level, fitted together and saved together.

    Per level rather than one shared, because the levels are not
    equally hard. Level 1 has a dozen classes and is nearly always
    right; level 4 has thousands and is not. A single temperature that
    fixes one will break the other.
    """

    method: Optional[str] = "temperature"
    level_columns: List[str] = field(default_factory=list)
    calibrators: List[object] = field(default_factory=list)

    def fit(self, level_scores: Sequence[np.ndarray], targets: np.ndarray,
            max_iter: int = 100, verbose: bool = True) -> "LevelCalibrators":
        targets = np.asarray(targets)
        self.calibrators = []
        for i, scores in enumerate(level_scores):
            name = self.level_columns[i] if i < len(self.level_columns) else f"L{i+1}"
            cal = make_calibrator(self.method)
            if verbose:
                print(f"[calib] {name}:", end=" ")
            if isinstance(cal, TemperatureScaler):
                cal.fit(scores, targets[:, i], max_iter=max_iter, verbose=verbose)
            elif isinstance(cal, IsotonicConfidenceScaler):
                cal.fit(scores, targets[:, i], verbose=verbose)
            elif verbose:
                print("no calibration")
            self.calibrators.append(cal)
        return self

    def transform(self, level_scores: Sequence[np.ndarray]) -> List[np.ndarray]:
        if not self.calibrators:
            return [IdentityScaler().transform(s) for s in level_scores]
        return [c.transform(s) for c, s in zip(self.calibrators, level_scores)]


# ---------------------------------------------------------------------------
# Section D: validate the signal, then choose a threshold
# ---------------------------------------------------------------------------

def reliability_table(
    confidence: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Accuracy against mean confidence, in equal-width bins.

    The table behind the reliability diagram. Read the `gap` column:
    positive means over-confident in that bin, which is the direction
    a neural network fails in.
    """
    conf = np.asarray(confidence, dtype=np.float64)
    corr = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(conf, edges) - 1, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = bin_ids == b
        count = int(mask.sum())
        if count:
            mean_conf = float(conf[mask].mean())
            acc = float(corr[mask].mean())
        else:
            mean_conf = float((edges[b] + edges[b + 1]) / 2)
            acc = float("nan")
        rows.append({
            "bin": f"[{edges[b]:.1f}, {edges[b+1]:.1f})",
            "count": count,
            "mean_confidence": mean_conf,
            "accuracy": acc,
            "gap": mean_conf - acc if count else float("nan"),
        })
    return pd.DataFrame(rows)


def expected_calibration_error(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10
) -> float:
    """Count-weighted mean |accuracy - confidence| across the bins.

    One number for 'how far off is the confidence'. An ECE of 0.15 on
    a model reporting 0.9 confidence means the true accuracy is closer
    to 0.75, and any threshold chosen from the raw number is wrong by
    about that much.
    """
    table = reliability_table(confidence, correct, n_bins)
    valid = table[(table["count"] > 0) & table["accuracy"].notna()]
    if not len(valid) or not len(confidence):
        return float("nan")
    return float((valid["count"] * valid["gap"].abs()).sum() / len(confidence))


def coverage_table(
    confidence: np.ndarray,
    correct: np.ndarray,
    thresholds: Sequence[float] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
) -> pd.DataFrame:
    """What you keep, and how right it is, at each candidate threshold.

    The decision table. Coverage is the share of rows auto-assigned;
    accuracy is how often those are right; `reviewed` is the workload
    the rest represents. Choosing a threshold is choosing a row here,
    and the trade is explicit rather than implied.
    """
    conf = np.asarray(confidence)
    corr = np.asarray(correct)
    rows = []
    for thr in thresholds:
        keep = conf >= thr
        n_keep = int(keep.sum())
        rows.append({
            "threshold": float(thr),
            "coverage": float(keep.mean()) if len(conf) else 0.0,
            "n_auto": n_keep,
            "accuracy_at_threshold": float(corr[keep].mean()) if n_keep else float("nan"),
            "n_reviewed": int(len(conf) - n_keep),
            "errors_auto": int((1 - corr[keep]).sum()) if n_keep else 0,
        })
    return pd.DataFrame(rows)


def choose_threshold(
    confidence: np.ndarray,
    correct: np.ndarray,
    target_accuracy: float = 0.95,
    min_coverage: float = 0.0,
    grid: Optional[Sequence[float]] = None,
) -> float:
    """The lowest threshold whose auto-assigned rows hit the target.

    Lowest, not highest: among thresholds that all meet the accuracy
    bar, the lowest is the one that sends the fewest rows to a human.
    Returns `inf` when the target is unreachable at any threshold --
    which is a real answer, and means nothing should be auto-assigned
    at that level rather than that the target should be quietly
    lowered.
    """
    conf = np.asarray(confidence)
    corr = np.asarray(correct)
    grid = grid if grid is not None else np.round(np.arange(0.05, 1.0, 0.01), 3)

    best = float("inf")
    for thr in grid:
        keep = conf >= thr
        if not keep.any():
            continue
        coverage = float(keep.mean())
        accuracy = float(corr[keep].mean())
        if accuracy >= target_accuracy and coverage >= min_coverage:
            best = float(thr)
            break
    return best


@dataclass
class ConfidenceReport:
    """Everything section D produces, in one object worth saving."""

    level: str
    accuracy: float
    ece: float
    reliability: pd.DataFrame
    coverage: pd.DataFrame
    chosen_threshold: float
    n_rows: int
    confidence: np.ndarray = field(repr=False, default=None)
    correct: np.ndarray = field(repr=False, default=None)

    def summary(self) -> str:
        thr = ("unreachable" if not np.isfinite(self.chosen_threshold)
               else f"{self.chosen_threshold:.2f}")
        return (
            f"{self.level}: accuracy={self.accuracy:.3f}  ECE={self.ece:.4f}  "
            f"threshold={thr}  (n={self.n_rows:,})"
        )


def validate_confidence(
    probs: np.ndarray,
    targets: np.ndarray,
    level: str = "L4",
    n_bins: int = 10,
    thresholds: Sequence[float] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    target_accuracy: float = 0.95,
    verbose: bool = True,
) -> ConfidenceReport:
    """Section D for one level: reliability, ECE, coverage, threshold."""
    p = np.asarray(probs)
    y = np.asarray(targets)
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(np.float32)

    report = ConfidenceReport(
        level=level,
        accuracy=float(correct.mean()) if len(correct) else float("nan"),
        ece=expected_calibration_error(conf, correct, n_bins),
        reliability=reliability_table(conf, correct, n_bins),
        coverage=coverage_table(conf, correct, thresholds),
        chosen_threshold=choose_threshold(conf, correct, target_accuracy),
        n_rows=int(len(y)),
        confidence=conf,
        correct=correct,
    )
    if verbose:
        print(f"[calib] {report.summary()}")
    return report


def validate_all_levels(
    level_probs: Sequence[np.ndarray],
    targets: np.ndarray,
    level_columns: Sequence[str],
    cfg=None,
    verbose: bool = True,
) -> Dict[str, ConfidenceReport]:
    """Section D for every active level at once."""
    from .config import CalibrationConfig

    cfg = cfg or CalibrationConfig()
    targets = np.asarray(targets)
    out: Dict[str, ConfidenceReport] = {}
    for i, name in enumerate(level_columns):
        out[name] = validate_confidence(
            level_probs[i], targets[:, i], level=name,
            n_bins=cfg.n_bins, thresholds=cfg.thresholds,
            target_accuracy=cfg.target_accuracy, verbose=verbose,
        )
    return out


def validate_joint(
    frame: pd.DataFrame,
    truth: pd.DataFrame,
    level_columns: Sequence[str],
    cfg=None,
    confidence_col: str = "Confidence",
    verbose: bool = True,
) -> ConfidenceReport:
    """Section D applied to a joint-path prediction.

    'Correct' here means the *whole path* matched, which is a stricter
    and more useful bar than the deepest level alone: a prediction
    that gets L4 right by luck while disagreeing with its own L2 is
    not something to auto-assign.
    """
    from .config import CalibrationConfig

    cfg = cfg or CalibrationConfig()
    level_columns = list(level_columns)
    pred_cols = [f"Predicted {c}" for c in level_columns]
    missing = [c for c in pred_cols if c not in frame.columns]
    if missing:
        raise KeyError(f"Prediction frame is missing {missing}.")

    correct = np.ones(len(frame), dtype=np.float32)
    for pred_col, true_col in zip(pred_cols, level_columns):
        correct *= (
            frame[pred_col].astype(str).to_numpy()
            == truth[true_col].astype(str).to_numpy()
        ).astype(np.float32)

    conf = frame[confidence_col].to_numpy(dtype=np.float64)
    report = ConfidenceReport(
        level="joint path",
        accuracy=float(correct.mean()) if len(correct) else float("nan"),
        ece=expected_calibration_error(conf, correct, cfg.n_bins),
        reliability=reliability_table(conf, correct, cfg.n_bins),
        coverage=coverage_table(conf, correct, cfg.thresholds),
        chosen_threshold=choose_threshold(conf, correct, cfg.target_accuracy),
        n_rows=int(len(frame)),
        confidence=conf,
        correct=correct,
    )
    if verbose:
        print(f"[calib] {report.summary()}")
    return report


def prefix_thresholds(
    frame: pd.DataFrame,
    truth: pd.DataFrame,
    level_columns: Sequence[str],
    target_accuracy: float = 0.95,
    verbose: bool = True,
) -> Dict[int, float]:
    """A threshold per depth, for variable-depth assignment.

    The cascade rule from the old repo, kept because it is the thing
    that makes a hierarchy worth having: when the model cannot pick a
    level-4 category confidently enough, it can still assign level 2
    correctly, and a correct level 2 is worth more than a coin-flip
    level 4. Try the deepest level first and walk up until one clears
    its bar.

    A depth whose target is unreachable gets `inf`, so the cascade
    falls straight through it.
    """
    level_columns = list(level_columns)
    out: Dict[int, float] = {}
    for depth in range(1, len(level_columns) + 1):
        col = f"{level_columns[depth-1]} Prefix Probability"
        if col not in frame.columns:
            out[depth] = float("inf")
            continue
        correct = np.ones(len(frame), dtype=np.float32)
        for d in range(depth):
            pred_col = f"Predicted {level_columns[d]}"
            correct *= (
                frame[pred_col].astype(str).to_numpy()
                == truth[level_columns[d]].astype(str).to_numpy()
            ).astype(np.float32)
        thr = choose_threshold(
            frame[col].to_numpy(dtype=np.float64), correct, target_accuracy
        )
        out[depth] = thr
        if verbose:
            shown = "unreachable" if not np.isfinite(thr) else f"{thr:.2f}"
            print(
                f"[calib] depth {depth} ({level_columns[depth-1]}): "
                f"threshold={shown}  base accuracy={correct.mean():.3f}"
            )
    return out


def level_confidence(
    level_probs: Sequence[np.ndarray],
    predicted_codes: np.ndarray,
    batch_size: int = 50000,
) -> np.ndarray:
    """Each level's own probability for the class actually predicted.

    Not the level's argmax. Under a joint mode the emitted path is
    chosen over whole paths, so the class at level d need not be that
    level's own top choice -- and it is the probability of *what was
    predicted* that has to be thresholded, not the probability of
    something else the model preferred in isolation.

    Returns `(n_rows, n_levels)`.
    """
    predicted_codes = np.asarray(predicted_codes)
    n_rows = predicted_codes.shape[0]
    out = np.zeros((n_rows, len(level_probs)), dtype=np.float32)

    for i, probs in enumerate(level_probs):
        for lo in range(0, n_rows, batch_size):
            hi = min(lo + batch_size, n_rows)
            block = np.asarray(probs[lo:hi], dtype=np.float32)
            out[lo:hi, i] = block[np.arange(hi - lo), predicted_codes[lo:hi, i]]
            del block
    return out


def per_level_cascade_thresholds(
    confidences: np.ndarray,
    predicted_codes: np.ndarray,
    true_codes: np.ndarray,
    level_columns: Sequence[str],
    target_accuracy: float = 0.95,
    verbose: bool = True,
) -> Dict[int, float]:
    """Cascade thresholds from each level's own calibrated confidence.

    The alternative to `prefix_thresholds`, and on a well-calibrated
    model a much better one. That function thresholds the *joint*
    prefix mass, which under `independent` mode is the product of four
    probabilities renormalised over every valid path -- nearly
    winner-take-all, saturated near 1.0 at every depth, and empirically
    an order of magnitude worse calibrated than the per-level numbers
    it was built from (ECE ~0.105 against ~0.015). When the joint
    saturates, no threshold can isolate an accurate subset and every
    shallow depth comes back `unreachable`, even where the level's own
    probability reaches the target comfortably.

    This uses the calibrated per-level probability instead, which is
    the quantity the temperature was actually fitted to.

    One fact makes this exactly right rather than merely convenient:
    in a tree, a node at depth d determines its entire ancestry. So
    "the prefix through depth d is correct" is equivalent to "the
    level-d prediction is correct", and a per-level threshold is a
    prefix threshold. That equivalence does not hold for a DAG, and
    this function would be wrong for one.
    """
    confidences = np.asarray(confidences)
    predicted_codes = np.asarray(predicted_codes)
    true_codes = np.asarray(true_codes)

    out: Dict[int, float] = {}
    for depth in range(1, len(level_columns) + 1):
        i = depth - 1
        correct = (predicted_codes[:, i] == true_codes[:, i]).astype(np.float32)
        thr = choose_threshold(confidences[:, i], correct, target_accuracy)
        out[depth] = thr
        if verbose:
            shown = "unreachable" if not np.isfinite(thr) else f"{thr:.2f}"
            keep = confidences[:, i] >= thr if np.isfinite(thr) else np.zeros(
                len(correct), dtype=bool)
            cov = float(keep.mean())
            print(f"[calib] depth {depth} ({level_columns[i]}): threshold={shown}  "
                  f"base accuracy={correct.mean():.3f}  coverage at threshold={cov:.3f}")
    return out


def apply_cascade_per_level(
    frame: pd.DataFrame,
    confidences: np.ndarray,
    level_columns: Sequence[str],
    thresholds: Dict[int, float],
) -> pd.DataFrame:
    """Assign at the deepest level whose own confidence clears its bar.

    Same contract as `apply_cascade` -- adds `Assigned Depth`, blanks
    the predictions below it, sets `Needs Review` -- but driven by the
    per-level confidences rather than the joint prefix mass.
    """
    level_columns = list(level_columns)
    out = frame.copy()
    confidences = np.asarray(confidences)
    depth_assigned = np.zeros(len(out), dtype=np.int32)

    for depth in range(len(level_columns), 0, -1):
        thr = thresholds.get(depth, float("inf"))
        if not np.isfinite(thr):
            continue
        eligible = (depth_assigned == 0) & (confidences[:, depth - 1] >= thr)
        depth_assigned[eligible] = depth

    out["Assigned Depth"] = depth_assigned
    out["Assigned Confidence"] = np.where(
        depth_assigned > 0,
        confidences[np.arange(len(out)), np.clip(depth_assigned - 1, 0, None)],
        np.nan,
    )
    for d, col in enumerate(level_columns, start=1):
        pred_col = f"Predicted {col}"
        if pred_col in out.columns:
            out.loc[depth_assigned < d, pred_col] = None
    out["Needs Review"] = depth_assigned < len(level_columns)
    return out


def cascade_report(
    assigned: pd.DataFrame,
    predicted_codes: np.ndarray,
    true_codes: np.ndarray,
    level_columns: Sequence[str],
) -> pd.DataFrame:
    """Rows, share and realised accuracy at each assigned depth.

    The accuracy column is the one to read: it is what the threshold
    was chosen to guarantee, measured after the fact.
    """
    level_columns = list(level_columns)
    depths = assigned["Assigned Depth"].to_numpy()
    predicted_codes = np.asarray(predicted_codes)
    true_codes = np.asarray(true_codes)

    rows = []
    for depth in range(0, len(level_columns) + 1):
        mask = depths == depth
        n = int(mask.sum())
        if depth == 0:
            rows.append({"assigned_depth": 0, "level": "no assignment",
                         "rows": n, "share": n / max(len(assigned), 1),
                         "accuracy": float("nan")})
            continue
        i = depth - 1
        acc = (float((predicted_codes[mask, i] == true_codes[mask, i]).mean())
               if n else float("nan"))
        rows.append({"assigned_depth": depth, "level": level_columns[i],
                     "rows": n, "share": n / max(len(assigned), 1),
                     "accuracy": acc})

    table = pd.DataFrame(rows)
    assigned_rows = int((depths > 0).sum())
    table.attrs["coverage"] = assigned_rows / max(len(assigned), 1)
    return table


def apply_cascade(
    frame: pd.DataFrame,
    level_columns: Sequence[str],
    thresholds: Dict[int, float],
) -> pd.DataFrame:
    """Assign each row at the deepest depth that clears its threshold.

    Adds `Assigned Depth` (0 = nothing confident enough, send it to a
    human) and blanks the predictions below that depth, so the output
    cannot be read as claiming more precision than was earned.
    """
    level_columns = list(level_columns)
    out = frame.copy()
    depth_assigned = np.zeros(len(out), dtype=np.int32)

    for depth in range(len(level_columns), 0, -1):
        col = f"{level_columns[depth-1]} Prefix Probability"
        thr = thresholds.get(depth, float("inf"))
        if col not in out.columns or not np.isfinite(thr):
            continue
        eligible = (depth_assigned == 0) & (out[col].to_numpy() >= thr)
        depth_assigned[eligible] = depth

    out["Assigned Depth"] = depth_assigned
    for d, col in enumerate(level_columns, start=1):
        pred_col = f"Predicted {col}"
        if pred_col in out.columns:
            out.loc[depth_assigned < d, pred_col] = None
    out["Needs Review"] = depth_assigned < len(level_columns)
    return out
