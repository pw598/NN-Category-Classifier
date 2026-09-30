"""Classical estimators on pooled embedding vectors.

Item 3 of the outline: one notebook per model, added as each is
attempted. This module is the part they share, so a new model is a
registry entry and a notebook rather than a new pipeline.

The design mirrors the networks deliberately -- one estimator per
level, producing the same list of per-level probability matrices that
`hierarchy` consumes. The difference is that these estimators do not
share a trunk, so the levels genuinely are separate models and the
coarse levels teach the deep one nothing. That is a real disadvantage
and it should show up in the comparison; it is not something to
paper over.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import SklearnModelConfig


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def _sgd_logistic(**params):
    from sklearn.linear_model import SGDClassifier

    defaults = dict(
        loss="log_loss", alpha=1e-5, max_iter=30, tol=1e-4,
        random_state=42, n_jobs=-1,
    )
    defaults.update(params)
    return SGDClassifier(**defaults)


def _linear_svc(**params):
    """LinearSVC has no predict_proba, so it is wrapped in a calibrator.

    Platt scaling via CalibratedClassifierCV, which refits internally
    on cross-validation folds. That makes it noticeably slower than
    the others and is the price of getting a probability out of a
    margin-based model at all -- a decision-function value is not a
    confidence and must not be thresholded as if it were.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.svm import LinearSVC

    cv = params.pop("cv", 3)
    defaults = dict(C=1.0, max_iter=2000, random_state=42)
    defaults.update(params)
    return CalibratedClassifierCV(LinearSVC(**defaults), cv=cv, method="sigmoid")


def _gaussian_nb(**params):
    from sklearn.naive_bayes import GaussianNB

    return GaussianNB(**params)


def _logistic(**params):
    from sklearn.linear_model import LogisticRegression

    defaults = dict(max_iter=1000, n_jobs=-1, random_state=42)
    defaults.update(params)
    return LogisticRegression(**defaults)


# ---------------------------------------------------------------------------
# Estimators whose cost does not scale with the number of classes
# ---------------------------------------------------------------------------
#
# The reason this section exists. sklearn turns most linear classifiers
# into one-vs-rest, so 3,473 leaf categories means 3,473 binary fits per
# level -- about 14,000 for the four levels, each over half a million
# rows. That is why sgd_logistic does not finish.
#
# Everything below avoids the expansion: one pooled covariance, one
# multi-output solve, or one pass of class means. Cost is essentially
# independent of the class count.


def _softmax_inplace(scores: np.ndarray) -> np.ndarray:
    """Softmax a float32 block, allocating nothing but the row maxima.

    Written this way because the obvious version is not affordable
    here. `exp(s / t - max) / sum` reads as four expressions and
    allocates four full arrays; on a 105,334 x 3,473 block that is
    5.5 GB of transients for a result of 1.4 GB.
    """
    scores -= scores.max(axis=1, keepdims=True)
    np.exp(scores, out=scores)
    scores /= scores.sum(axis=1, keepdims=True)
    return scores


class SoftmaxDecision:
    """Gives `predict_proba` to a margin-based estimator.

    A `decision_function` value is a distance from a boundary, not a
    probability, and the hierarchy and the calibration step both need
    probabilities. Softmaxing the margins produces something with the
    right shape and ordering but no claim to being calibrated -- which
    is precisely what the isotonic step downstream is for, and why
    these models must not be thresholded on the raw number.

    Deliberately not a `sklearn.base.BaseEstimator` subclass: it is
    used only through `HierarchicalSklearnClassifier`, which needs
    `fit`, `predict_proba` and `classes_` and nothing else.
    """

    def __init__(self, estimator, temperature: float = 1.0):
        self.estimator = estimator
        self.temperature = float(temperature)
        self.classes_ = None

    def fit(self, X, y):
        self.estimator.fit(X, y)
        self.classes_ = self.estimator.classes_
        return self

    def decision_function(self, X):
        scores = self.estimator.decision_function(X)
        if scores.ndim == 1:                      # binary: one margin
            scores = np.column_stack([-scores, scores])
        return scores

    def predict_proba(self, X, chunk_size: int = 20000) -> np.ndarray:
        X = np.asarray(X)
        n = X.shape[0]
        out = None
        for lo in range(0, n, chunk_size):
            hi = min(lo + chunk_size, n)
            block = np.asarray(self.decision_function(X[lo:hi]), dtype=np.float32)
            if self.temperature != 1.0:
                block /= max(self.temperature, 1e-6)
            _softmax_inplace(block)
            if out is None:
                out = np.empty((n, block.shape[1]), dtype=np.float32)
            out[lo:hi] = block
            del block
        return out

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class CentroidClassifier:
    """One mean vector per class, and a softmax over similarity to it.

    The cheapest thing that can be called a model: a single pass to
    accumulate class means, then a matrix product at predict time. Cost
    is O(n x d) to fit regardless of how many classes there are.

    Cosine by default rather than Euclidean. Pooled word vectors vary
    in magnitude with how many known tokens a description had, and
    Euclidean distance reads that variation as meaning; cosine does
    not. `temperature` controls how peaked the resulting distribution
    is and is worth tuning -- similarities live in [-1, 1], so the
    default of 0.1 is what turns them into something with any
    discrimination at all.

    Worth running even if you do not ship it. It is the floor: any
    model that cannot beat class means on these features is not
    learning anything the features do not already hand it.
    """

    def __init__(self, metric: str = "cosine", temperature: float = 0.1):
        self.metric = metric
        self.temperature = float(temperature)
        self.classes_ = None
        self.centroids_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_ = np.unique(y)

        centroids = np.zeros((len(self.classes_), X.shape[1]), dtype=np.float32)
        index = {c: i for i, c in enumerate(self.classes_)}
        counts = np.zeros(len(self.classes_), dtype=np.float32)
        rows = np.fromiter((index[v] for v in y), dtype=np.int64, count=len(y))
        np.add.at(centroids, rows, X)
        np.add.at(counts, rows, 1.0)
        centroids /= np.maximum(counts, 1.0)[:, None]

        if self.metric == "cosine":
            norms = np.linalg.norm(centroids, axis=1, keepdims=True)
            np.divide(centroids, norms, out=centroids, where=norms > 0)
        self.centroids_ = centroids
        return self

    def decision_function(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if self.metric == "cosine":
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            Xn = np.divide(X, norms, out=np.zeros_like(X), where=norms > 0)
            return Xn @ self.centroids_.T
        # Negative squared Euclidean, expanded so the (n x k) product is
        # the only large allocation.
        sq = (X ** 2).sum(axis=1, keepdims=True)
        cs = (self.centroids_ ** 2).sum(axis=1)[None, :]
        return -(sq + cs - 2.0 * (X @ self.centroids_.T))

    def predict_proba(self, X, chunk_size: int = 20000) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        n = X.shape[0]
        out = np.empty((n, len(self.classes_)), dtype=np.float32)
        for lo in range(0, n, chunk_size):
            hi = min(lo + chunk_size, n)
            block = self.decision_function(X[lo:hi])
            if self.temperature != 1.0:
                block /= max(self.temperature, 1e-6)
            out[lo:hi] = _softmax_inplace(block)
            del block
        return out

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def _lda(**params):
    """Linear discriminant analysis: one pooled covariance, closed form.

    The best-founded of the fast options for pooled word vectors. It
    assumes every class shares one covariance matrix -- far weaker than
    GaussianNB's assumption that every dimension is independent, and
    much closer to true for mean-pooled embeddings.

    `solver="lsqr"` with Ledoit-Wolf shrinkage rather than the default
    SVD solver: shrinkage is what keeps the 300x300 covariance
    invertible when a class has only a handful of members, which with
    3,473 leaf categories is most of them.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    defaults = dict(solver="lsqr", shrinkage="auto")
    defaults.update(params)
    return LinearDiscriminantAnalysis(**defaults)


def _ridge(**params):
    """Ridge regression on one-hot targets, wrapped to give probabilities.

    Forms `XtX` (300x300) once and then solves for every class in a
    single multi-output least-squares step, so the class count barely
    enters the cost. No probabilities of its own, hence the wrapper.
    """
    from sklearn.linear_model import RidgeClassifier

    temperature = params.pop("temperature", 1.0)
    defaults = dict(alpha=1.0, solver="auto", random_state=42)
    defaults.update(params)
    return SoftmaxDecision(RidgeClassifier(**defaults), temperature=temperature)


def _nearest_centroid(**params):
    return CentroidClassifier(**params)


REGISTRY: Dict[str, Callable[..., Any]] = {
    "sgd_logistic": _sgd_logistic,
    "linear_svc": _linear_svc,
    "gaussian_nb": _gaussian_nb,
    "logistic": _logistic,
    "lda": _lda,
    "ridge": _ridge,
    "nearest_centroid": _nearest_centroid,
}


def build_estimator(cfg: SklearnModelConfig):
    if cfg.name not in REGISTRY:
        raise ValueError(
            f"Unknown model {cfg.name!r}. Registered: {sorted(REGISTRY)}. "
            "Add a factory to sklearn_models.REGISTRY to register a new one."
        )
    return REGISTRY[cfg.name](**(cfg.params or {}))


# ---------------------------------------------------------------------------
# One estimator per level
# ---------------------------------------------------------------------------

@dataclass
class HierarchicalSklearnClassifier:
    """Fits one estimator per level and emits hierarchy-aligned probabilities.

    'Aligned' is the part that needs care. An estimator only knows the
    classes present in the rows it was fitted on, which on a
    cross-validation fold is not necessarily every class in the
    hierarchy. `_expand` scatters each estimator's `predict_proba`
    back into a full-width matrix, so every returned array has exactly
    `hierarchy.n_classes(level)` columns and column j always means the
    same category. Skipping that step produces arrays that look right,
    line up with nothing, and fail silently.
    """

    cfg: SklearnModelConfig
    level_columns: List[str]
    n_classes_per_level: List[int]
    estimators: List[Any] = field(default_factory=list)
    scaler: Any = None

    def _fit_scaler(self, X: np.ndarray) -> np.ndarray:
        if not self.cfg.scale_features:
            return X
        from sklearn.preprocessing import StandardScaler

        # copy=False for the fit only. `X` here is already a fresh copy
        # -- `X[train_idx]` with a fancy index always is -- so scaling
        # it in place saves a second half-gigabyte per fold. It is
        # switched back before any transform, where the input is a
        # *view* of the caller's array and in-place would corrupt it.
        self.scaler = StandardScaler(copy=False)
        scaled = self.scaler.fit_transform(X)
        self.scaler.copy = True
        return scaled

    def _apply_scaler(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(X) if self.scaler is not None else X

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        """y is `(n_rows, n_levels)` of integer class codes."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        Xs = self._fit_scaler(X)

        self.estimators = []
        for i, level in enumerate(self.level_columns):
            est = build_estimator(self.cfg)
            est.fit(Xs, y[:, i])
            self.estimators.append(est)
            if verbose:
                print(
                    f"[sk] fitted {self.cfg.name} for {level} "
                    f"({len(getattr(est, 'classes_', [])):,} classes seen)"
                )
        return self

    def _expand(self, probs: np.ndarray, classes: np.ndarray, n_classes: int) -> np.ndarray:
        full = np.zeros((probs.shape[0], n_classes), dtype=np.float32)
        full[:, np.asarray(classes, dtype=int)] = probs
        return full

    def predict_level_probs(
        self,
        X: np.ndarray,
        out: Optional[List[np.ndarray]] = None,
        row_index: Optional[np.ndarray] = None,
        chunk_size: int = 20000,
    ) -> List[np.ndarray]:
        """Per-level probabilities, written into `out` if one is given.

        Chunked over rows, and for two separate reasons.

        sklearn's `predict_proba` returns float64, so an unchunked call
        on a 105,334-row fold builds a 2.7 GB array for the level-4
        head alone -- and `_expand` then builds a 1.4 GB float32 copy
        beside it. Neither is needed in full: only one chunk has to
        exist at a time.

        And passing `out` lets the caller supply the destination -- the
        out-of-fold array it was going to copy into anyway -- so the
        four per-level blocks for the fold are never materialised as a
        separate list. With `row_index`, the destination rows need not
        be contiguous, which is exactly the cross-validation case.
        """
        X = np.asarray(X, dtype=np.float32)
        n = X.shape[0]

        if out is None:
            out = [np.zeros((n, k), dtype=np.float32)
                   for k in self.n_classes_per_level]
            row_index = None

        for lo in range(0, n, chunk_size):
            hi = min(lo + chunk_size, n)
            Xs = self._apply_scaler(X[lo:hi])
            rows = slice(lo, hi) if row_index is None else row_index[lo:hi]

            for i, est in enumerate(self.estimators):
                probs = est.predict_proba(Xs)
                classes = np.asarray(est.classes_, dtype=int)
                n_classes = self.n_classes_per_level[i]

                if len(classes) == n_classes and classes[-1] == n_classes - 1:
                    # Every class present, already in order: a plain
                    # assignment, no scatter and no intermediate.
                    out[i][rows] = probs
                else:
                    out[i][rows] = self._expand(probs, classes, n_classes)
                del probs
            del Xs

        return out


def allocate_oof(
    n_rows: int,
    n_classes_per_level: Sequence[int],
    memmap_dir=None,
    prefix: str = "oof",
    verbose: bool = True,
) -> List[np.ndarray]:
    """The out-of-fold arrays, in RAM or backed by files on disk.

    These dominate the memory of a cross-validated run: one
    `n_rows x n_classes` float32 per level, and at 526,670 rows with
    3,473 leaf categories that is 8.6 GB held for the whole notebook,
    before any model has allocated anything.

    `memmap_dir` moves them to disk. They are written and read exactly
    once each, in row order, so paging costs almost nothing and the
    8.6 GB stops competing with the per-fold transients. On a 32 GB
    driver this is usually the difference between finishing and not.
    """
    total = sum(n_rows * k * 4 for k in n_classes_per_level)
    if verbose:
        where = "in memory" if memmap_dir is None else f"on disk ({memmap_dir})"
        print(f"[sk] out-of-fold arrays: {total / 1024**3:.2f} GB {where}")

    if memmap_dir is None:
        return [np.zeros((n_rows, k), dtype=np.float32) for k in n_classes_per_level]

    import pathlib

    directory = pathlib.Path(memmap_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return [
        np.lib.format.open_memmap(
            directory / f"{prefix}_level_{i}.npy",
            mode="w+", dtype=np.float32, shape=(n_rows, k),
        )
        for i, k in enumerate(n_classes_per_level)
    ]


def out_of_fold_probs(
    cfg: SklearnModelConfig,
    X: np.ndarray,
    y: np.ndarray,
    level_columns: Sequence[str],
    n_classes_per_level: Sequence[int],
    verbose: bool = True,
    memmap_dir=None,
    chunk_size: int = 20000,
):
    """Cross-validated probabilities for every row, and the folds used.

    Every row gets a prediction from a model that never saw it, so the
    calibration downstream is fitted on the whole catalogue rather
    than on a tenth of it. On a taxonomy with a long tail that matters
    more than it sounds: a single validation split leaves the rare
    categories with a handful of rows each, and a threshold chosen
    from those is chosen from noise.

    This is affordable here and is not for the networks, which is why
    it exists on this side only.
    """
    import gc

    from sklearn.model_selection import StratifiedKFold

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    level_columns = list(level_columns)
    n_rows = X.shape[0]

    oof = allocate_oof(n_rows, n_classes_per_level, memmap_dir, verbose=verbose)
    folds = np.zeros(n_rows, dtype=np.int32)

    skf = StratifiedKFold(
        n_splits=cfg.n_folds, shuffle=True, random_state=cfg.random_state
    )
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y[:, -1])):
        folds[test_idx] = fold
        clf = HierarchicalSklearnClassifier(
            cfg=cfg,
            level_columns=level_columns,
            n_classes_per_level=list(n_classes_per_level),
        ).fit(X[train_idx], y[train_idx], verbose=False)

        # Straight into the out-of-fold arrays. The per-fold blocks are
        # never built as a separate list, which on the level-4 head is
        # 1.4 GB not allocated.
        clf.predict_level_probs(
            X[test_idx], out=oof, row_index=test_idx, chunk_size=chunk_size
        )

        if verbose:
            acc = float((oof[-1][test_idx].argmax(axis=1) == y[test_idx, -1]).mean())
            print(f"[sk] fold {fold + 1}/{cfg.n_folds}: "
                  f"{level_columns[-1]} accuracy={acc:.4f}")

        # Explicitly, and this matters. The fold's fitted estimators and
        # its copy of the training block are several hundred megabytes;
        # left to the next iteration's rebinding they stay alive while
        # the replacement is being built, and the two peaks overlap.
        # That overlap is what kills fold 2 rather than fold 1.
        del clf
        gc.collect()

    return oof, folds


def describe_registry() -> pd.DataFrame:
    """What is available, for a notebook to print before choosing."""
    notes = {
        "sgd_logistic": "One-vs-rest: one binary fit per class. Impractical "
                        "beyond a few hundred classes.",
        "linear_svc": "Usually the strongest linear option, but one-vs-rest AND "
                      "internally cross-validated for probabilities. Slowest here.",
        "logistic": "Full-batch. multi_class='multinomial' avoids one-vs-rest but "
                    "materialises an n_samples x n_classes matrix -- check the "
                    "memory before trying it.",
        "gaussian_nb": "One pass. Assumes every dimension independent, which is "
                       "crude for pooled embeddings but very fast.",
        "lda": "One pooled covariance, closed form. Best-founded of the fast "
               "options for dense embeddings. Start here.",
        "ridge": "One multi-output least-squares solve. Very fast; probabilities "
                 "are softmaxed margins, so lean on the isotonic calibration.",
        "nearest_centroid": "Class means and a cosine softmax. The floor -- any "
                            "model that cannot beat it is not learning anything.",
    }
    scaling = {
        "sgd_logistic": "O(n_classes) fits",
        "linear_svc": "O(n_classes) fits x CV",
        "logistic": "one fit, n x k memory",
        "gaussian_nb": "one pass",
        "lda": "one pass + d x d solve",
        "ridge": "one d x d solve",
        "nearest_centroid": "one pass",
    }
    return pd.DataFrame([
        {"name": k, "cost": scaling.get(k, ""), "notes": notes.get(k, "")}
        for k in sorted(REGISTRY)
    ])
