"""Four levels, one answer.

Every model in this repo produces the same thing: a list of per-level
probability matrices, one per active level, each `(n_rows, n_classes_at_that_level)`.
Everything about the hierarchy lives here and nowhere else, so a new
model family only has to produce that list to get joint scoring,
prefix probabilities, top-k and coherent multi-level output for free.

Three ways to combine the levels, and the difference between them is
an assumption, not a detail:

independent
    Multiply the per-level probabilities, then keep only the paths
    actually seen in training. Assumes P(L1..L4) = P(L1)P(L2)P(L3)P(L4),
    which is plainly false -- the levels are nested, so knowing L4
    determines L1 -- but the falsehood is a systematic over-confidence
    that calibration can absorb, and the mode is cheap and robust.

conditional
    The chain rule: P(L1) * P(L2|L1) * P(L3|L1,L2) * P(L4|L1..L3).
    Each conditional is the level's own head renormalised over just
    the children of the parent under consideration, so a category is
    scored against its siblings rather than against the whole level.
    Better-behaved probabilities, and slower.

None
    No combining at all. Each level is argmaxed on its own. Levels can
    contradict each other, which is exactly what you want when the
    question is 'how good is this model at level 2', rather than
    'what should this SKU be filed under'.

The path constraint in the first two modes is the load-bearing part:
the score is only ever computed over paths that exist, so the answer
is always a real place in the tree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

EPS = 1e-12
NEG_INF = -1e30


def _log(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p, EPS, None)).astype(np.float32)


def _logsumexp(a: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)


@dataclass
class LabelHierarchy:
    """The label tree, learned from the training rows.

    Holds the per-level encoders and the set of valid paths. Picklable
    and saved with the model: a prediction made against a different
    tree than the one the model was trained on is not comparable to
    anything, and the mismatch would otherwise be invisible.
    """

    level_columns: List[str]
    classes_: List[np.ndarray]
    paths_: np.ndarray                 # (n_paths, n_levels), int32 codes
    path_counts_: np.ndarray           # (n_paths,)
    class_to_idx_: List[Dict[str, int]] = field(default_factory=list, repr=False)
    _prefix_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict, repr=False, compare=False
    )

    # -- construction ------------------------------------------------

    @classmethod
    def from_labels(
        cls, labels_df: pd.DataFrame, level_columns: Sequence[str]
    ) -> "LabelHierarchy":
        level_columns = list(level_columns)
        frame = labels_df[level_columns].astype(str)

        classes = [np.array(sorted(frame[c].unique()), dtype=object) for c in level_columns]
        class_to_idx = [{v: i for i, v in enumerate(cl)} for cl in classes]

        codes = np.column_stack([
            frame[c].map(class_to_idx[i]).to_numpy(dtype=np.int32)
            for i, c in enumerate(level_columns)
        ])
        paths, counts = np.unique(codes, axis=0, return_counts=True)

        return cls(
            level_columns=level_columns,
            classes_=classes,
            paths_=paths.astype(np.int32),
            path_counts_=counts.astype(np.int64),
            class_to_idx_=class_to_idx,
        )

    # -- basic accessors ---------------------------------------------

    @property
    def n_levels(self) -> int:
        return len(self.level_columns)

    @property
    def n_paths(self) -> int:
        return int(self.paths_.shape[0])

    def n_classes(self, level: int) -> int:
        return int(len(self.classes_[level]))

    def sizes(self) -> pd.DataFrame:
        return pd.DataFrame({
            "level": self.level_columns,
            "n_classes": [self.n_classes(i) for i in range(self.n_levels)],
        })

    def encode(self, labels_df: pd.DataFrame) -> np.ndarray:
        """Labels to integer codes. Unseen labels become -1.

        -1 rather than an exception on purpose: an evaluation frame
        containing a category that training never saw is a real
        situation, and it should show up as a row that cannot be
        correct rather than as a crash halfway through scoring.
        """
        cols = []
        for i, c in enumerate(self.level_columns):
            mapped = labels_df[c].astype(str).map(self.class_to_idx_[i])
            cols.append(mapped.fillna(-1).to_numpy(dtype=np.int32))
        return np.column_stack(cols)

    def decode(self, codes: np.ndarray, prefix: str = "Predicted ") -> pd.DataFrame:
        """Integer codes back to label strings, one column per level."""
        codes = np.asarray(codes)
        out = {}
        for i, col in enumerate(self.level_columns):
            vals = codes[:, i]
            labels = np.where(
                vals >= 0,
                self.classes_[i][np.clip(vals, 0, self.n_classes(i) - 1)],
                None,
            )
            out[f"{prefix}{col}"] = labels
        return pd.DataFrame(out)

    # -- prefix grouping ---------------------------------------------

    def prefix_groups(self, depth: int) -> Tuple[np.ndarray, np.ndarray]:
        """Group path indices by their first `depth` levels.

        Returns `(group_of_path, representative_path_index)`:
        `group_of_path[p]` is the group id of path p, and
        `representative_path_index[g]` is any path in group g (used to
        read the group's label codes back out).

        Cached, because both the conditional mode and the prefix
        probabilities ask for the same grouping on every batch.
        """
        if depth in self._prefix_cache:
            return self._prefix_cache[depth]
        if depth <= 0:
            # The empty prefix: every path is in one group. np.unique on a
            # zero-width array does not do the right thing here.
            group_of_path = np.zeros(self.n_paths, dtype=np.int64)
            rep = np.zeros(1, dtype=np.int64)
            self._prefix_cache[depth] = (group_of_path, rep)
            return group_of_path, rep
        prefixes = self.paths_[:, :depth]
        _, group_of_path, _ = np.unique(
            prefixes, axis=0, return_inverse=True, return_counts=True
        )
        group_of_path = group_of_path.astype(np.int64).ravel()
        n_groups = int(group_of_path.max()) + 1 if len(group_of_path) else 0
        rep = np.zeros(n_groups, dtype=np.int64)
        # Later assignments overwrite earlier ones; any member will do.
        rep[group_of_path] = np.arange(len(group_of_path))
        self._prefix_cache[depth] = (group_of_path, rep)
        return group_of_path, rep

    # -- joint scoring -----------------------------------------------

    def choose_batch_size(self, n_rows: int, budget_bytes: int = 256 * 1024 ** 2) -> int:
        """Rows per block, so one block stays inside a memory budget.

        The scoring block is `(batch x n_paths)` float32, and a couple
        of them are live at once. On a real taxonomy `n_paths` runs to
        tens of thousands, so scoring forty thousand rows in one go asks
        for several gigabytes per array -- which is how this crashed a
        cluster before it was batched.
        """
        per_row = max(1, self.n_paths) * 4 * 3        # score, exp, working copy
        return max(1, min(n_rows, int(budget_bytes // per_row)))

    def _score_block(
        self,
        level_probs: Sequence[np.ndarray],
        lo: int,
        hi: int,
        mode: str,
    ) -> np.ndarray:
        """Log score for every valid path, for rows [lo, hi).

        The logs are taken here, per block, rather than once up front.
        That ordering is the whole point: `log(p)` over every level of
        a 526,670-row catalogue is 8.6 GB of copies allocated *before*
        any batching happens, which makes the batching decorative. The
        gather comes first, so the only array that exists is the
        `(batch x n_paths)` block we were going to build anyway, and
        the log goes into it in place.
        """
        scores = np.zeros((hi - lo, self.n_paths), dtype=np.float32)
        for level in range(self.n_levels):
            # Gather first, then log in place: one block-sized array.
            contribution = np.asarray(
                level_probs[level][lo:hi][:, self.paths_[:, level]], dtype=np.float32
            )
            np.clip(contribution, EPS, None, out=contribution)
            np.log(contribution, out=contribution)

            if mode == "conditional":
                contribution = self._conditionalise(contribution, level)
            scores += contribution
            del contribution
        return scores

    def _prepare(self, level_probs: Sequence[np.ndarray], mode: Optional[str]):
        """Validate only. Deliberately does not touch the arrays.

        An earlier version returned `[_log(p) for p in level_probs]`,
        which materialised every level in full before scoring started
        and defeated the batching entirely -- and, when the caller
        passes memmaps, pulled all of them into RAM.
        """
        if mode is None:
            raise ValueError("Joint scoring needs a mode; use per_level_argmax.")
        if mode not in ("independent", "conditional"):
            raise ValueError(
                f"Unknown joint mode: {mode!r}; expected 'independent', "
                "'conditional' or None."
            )
        if len(level_probs) != self.n_levels:
            raise ValueError(
                f"Expected probabilities for {self.n_levels} levels, "
                f"got {len(level_probs)}."
            )
        return list(level_probs)

    def path_log_scores(
        self,
        level_probs: Sequence[np.ndarray],
        mode: Optional[str] = "independent",
    ) -> np.ndarray:
        """Log score for every valid path, `(n_rows, n_paths)`.

        `level_probs[l]` is `(n_rows, n_classes_at_level_l)` and must
        sum to one across each row -- these are softmax outputs, not
        logits. Pass them through a calibrator first if you want
        calibrated scores; nothing here rescales anything.

        Materialises the whole matrix, which is exactly the thing
        `score()` exists to avoid. Kept because it is the clearest
        statement of what the arithmetic is, and it is what the tests
        check against -- but on a full validation set, call `score()`.
        """
        prepared = self._prepare(level_probs, mode)
        return self._score_block(prepared, 0, prepared[0].shape[0], mode)

    def _conditionalise(self, contribution: np.ndarray, level: int) -> np.ndarray:
        """Turn P(class) into P(class | parent) for one level.

        The subtlety that makes this more than a grouped log_softmax:
        the columns are *paths*, and several paths share the same class
        at this level. Normalising over columns would divide by each
        sibling as many times as it has descendants, which is not a
        probability and is not even monotone in the right direction --
        a category with many children would be systematically
        penalised.

        So the denominator is built over distinct (parent, class)
        nodes, using one representative column each, and then
        broadcast back out to every path under that parent.

        Level 0 is normalised too, over the distinct root classes. That
        is what makes the whole chain sum to exactly one across valid
        paths, which is the property the mode is worth having for.
        """
        parent_of_path, _ = self.prefix_groups(level)
        node_of_path, rep_node = self.prefix_groups(level + 1)
        parent_of_node = parent_of_path[rep_node]

        node_values = contribution[:, rep_node]
        denom = _segmented_logsumexp(node_values, parent_of_node)
        return (contribution - denom[:, parent_of_path]).astype(np.float32)

    def score(
        self,
        level_probs: Sequence[np.ndarray],
        mode: Optional[str] = "independent",
        normalise: bool = True,
        top_k: int = 1,
        with_prefixes: bool = True,
        batch_size: Optional[int] = None,
        verbose: bool = False,
    ) -> Dict[str, object]:
        """Top-k paths and per-depth prefix statistics, in one batched pass.

        The function everything else should call. Two things it fixes,
        both of which were real:

        *It batches.* Only `(batch x n_paths)` is ever live, rather
        than `(n_rows x n_paths)`. Nothing in the result grows with
        `n_paths`, so the peak is bounded by the block size no matter
        how large the taxonomy gets.

        *It computes the block once.* The top-k answer and the prefix
        masses are two readings of the same matrix, and deriving them
        separately -- as `predict_paths` and `prefix_probabilities`
        used to, each rebuilding the scores from scratch -- doubled
        both the time and the peak memory for no benefit.
        """
        prepared = self._prepare(level_probs, mode)
        n_rows = prepared[0].shape[0]
        k = max(1, min(top_k, self.n_paths))
        batch = batch_size or self.choose_batch_size(n_rows)

        top_idx = np.zeros((n_rows, k), dtype=np.int64)
        top_log = np.zeros((n_rows, k), dtype=np.float32)
        log_total = np.zeros(n_rows, dtype=np.float32)

        depths = range(1, self.n_levels + 1) if with_prefixes else ()
        prefix_best = {d: np.zeros(n_rows, dtype=np.int64) for d in depths}
        prefix_p = {d: np.zeros(n_rows, dtype=np.float32) for d in depths}
        prefix_margin = {d: np.zeros(n_rows, dtype=np.float32) for d in depths}

        if verbose:
            n_blocks = (n_rows + batch - 1) // batch
            mb = batch * self.n_paths * 4 / 1e6
            print(f"[hierarchy] {n_rows:,} rows x {self.n_paths:,} paths in "
                  f"{n_blocks} block(s) of {batch:,} ({mb:.0f} MB each)")

        for lo in range(0, n_rows, batch):
            hi = min(lo + batch, n_rows)
            block = self._score_block(prepared, lo, hi, mode)
            total = _logsumexp(block, axis=1, keepdims=True)
            log_total[lo:hi] = total.ravel()

            # argpartition then sort: a full sort over every path is
            # wasteful when k is 1 or 3 and there are thousands.
            part = np.argpartition(-block, kth=k - 1, axis=1)[:, :k]
            part_scores = np.take_along_axis(block, part, axis=1)
            order = np.argsort(-part_scores, axis=1)
            top_idx[lo:hi] = np.take_along_axis(part, order, axis=1)
            top_log[lo:hi] = np.take_along_axis(part_scores, order, axis=1)

            if with_prefixes:
                normed = np.exp(block - total)
                for depth in depths:
                    group_of_path, rep = self.prefix_groups(depth)
                    n_groups = len(rep)
                    mass = np.zeros((hi - lo, n_groups), dtype=np.float32)
                    np.add.at(mass.T, group_of_path, normed.T)

                    if n_groups > 1:
                        pair = np.argpartition(-mass, kth=1, axis=1)[:, :2]
                        pair_p = np.take_along_axis(mass, pair, axis=1)
                        swap = pair_p[:, 0] < pair_p[:, 1]
                        pair[swap] = pair[swap][:, ::-1]
                        pair_p[swap] = pair_p[swap][:, ::-1]
                        prefix_best[depth][lo:hi] = pair[:, 0]
                        prefix_p[depth][lo:hi] = pair_p[:, 0]
                        prefix_margin[depth][lo:hi] = pair_p[:, 0] - pair_p[:, 1]
                    else:
                        prefix_best[depth][lo:hi] = 0
                        prefix_p[depth][lo:hi] = mass[:, 0]
                        prefix_margin[depth][lo:hi] = mass[:, 0]
                del normed
            del block

        probs = np.exp(top_log - log_total[:, None]) if normalise else np.exp(top_log)

        prefixes = {}
        for depth in depths:
            _, rep = self.prefix_groups(depth)
            prefixes[depth] = {
                "codes": self.paths_[rep[prefix_best[depth]], :depth],
                "probability": prefix_p[depth],
                "margin": prefix_margin[depth],
            }

        return {
            "path_index": top_idx,                       # (n_rows, k)
            "path_codes": self.paths_[top_idx],          # (n_rows, k, n_levels)
            "probability": probs.astype(np.float32),
            "path_evidence": np.exp(top_log).astype(np.float32),
            "log_total": log_total,
            "prefixes": prefixes,
        }

    def predict_paths(
        self,
        level_probs: Sequence[np.ndarray],
        mode: Optional[str] = "independent",
        normalise: bool = True,
        top_k: int = 1,
        batch_size: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """Best path (or top-k paths) per row, with scores.

        `normalise=True` divides by the total mass over valid paths, so
        the number reads as a probability conditional on the answer
        being a real category -- which it always is, by construction.
        The unnormalised value is returned too, as `path_evidence`: a
        row where every path is implausible still produces a confident
        -looking normalised score, and the evidence figure is what
        gives that away.
        """
        out = self.score(level_probs, mode, normalise, top_k,
                         with_prefixes=False, batch_size=batch_size)
        out.pop("prefixes", None)
        return out

    def prefix_probabilities(
        self,
        level_probs: Sequence[np.ndarray],
        mode: Optional[str] = "independent",
        batch_size: Optional[int] = None,
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """For each depth, the best prefix and the mass behind it.

        This is what makes variable-depth answers possible. A model can
        be unsure whether a SKU is a 'ball valve' or a 'gate valve'
        while being certain it is a valve; summing the path mass by
        prefix recovers that certainty instead of throwing it away with
        the losing leaf.

        Returns `{depth: {"codes", "probability", "margin"}}` where
        `margin` is the gap to the runner-up prefix -- often a better
        separator than the probability itself, because it is unaffected
        by how many plausible siblings there happen to be.
        """
        return self.score(level_probs, mode, top_k=1, with_prefixes=True,
                          batch_size=batch_size)["prefixes"]


def _segmented_logsumexp(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """logsumexp over the columns belonging to each group.

    `values` is `(n_rows, n_cols)`, `groups` labels each column, and
    the result is `(n_rows, n_groups)`.

    Written with the max-shift and `np.add.at` rather than a Python
    loop over groups because there are thousands of parents in a real
    taxonomy, and the loop version dominated the runtime of every
    conditional prediction.
    """
    n_groups = int(groups.max()) + 1 if len(groups) else 0
    n_rows = values.shape[0]

    maxes = np.full((n_rows, n_groups), NEG_INF, dtype=np.float32)
    np.maximum.at(maxes.T, groups, values.T)

    shifted = np.exp(values - maxes[:, groups])
    sums = np.zeros((n_rows, n_groups), dtype=np.float32)
    np.add.at(sums.T, groups, shifted.T)

    return (maxes + np.log(np.clip(sums, EPS, None))).astype(np.float32)


def per_level_argmax(
    level_probs: Sequence[np.ndarray],
    hierarchy: LabelHierarchy,
) -> Dict[str, np.ndarray]:
    """Each level decided on its own -- joint_mode=None.

    No path constraint, so the levels can disagree: a row can come back
    as L1 'ABRASIVES' and L4 'SAFETY GLASSES'. That incoherence is the
    honest output of four independent questions, and is the point of
    this mode. Do not ship it to a downstream system that expects a
    real path.
    """
    codes, confidence, margin = [], [], []
    for probs in level_probs:
        p = np.asarray(probs)
        order = np.argsort(-p, axis=1)
        best = order[:, 0]
        best_p = np.take_along_axis(p, best[:, None], axis=1).ravel()
        second_p = (
            np.take_along_axis(p, order[:, 1][:, None], axis=1).ravel()
            if p.shape[1] > 1 else np.zeros_like(best_p)
        )
        codes.append(best.astype(np.int32))
        confidence.append(best_p.astype(np.float32))
        margin.append((best_p - second_p).astype(np.float32))

    return {
        "codes": np.column_stack(codes),
        "confidence": np.column_stack(confidence),
        "margin": np.column_stack(margin),
    }


def predictions_frame(
    hierarchy: LabelHierarchy,
    level_probs: Sequence[np.ndarray],
    mode: Optional[str] = "independent",
    normalise: bool = True,
    predict_all_levels: bool = True,
    top_k: int = 1,
    include_margin_entropy: bool = True,
    batch_size: Optional[int] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """One tidy frame of predictions, whatever the mode.

    Always carries a `Confidence` column, so downstream code -- the
    calibration step, the review-queue threshold, the report -- does
    not have to know which mode produced it.

    `batch_size=None` lets the hierarchy size the blocks itself against
    a memory budget. Pass `cfg.hierarchy.batch_size` to fix it.
    """
    level_probs = [np.asarray(p) for p in level_probs]

    if mode is None:
        res = per_level_argmax(level_probs, hierarchy)
        frame = hierarchy.decode(res["codes"])
        for i, col in enumerate(hierarchy.level_columns):
            frame[f"{col} Confidence"] = res["confidence"][:, i]
            if include_margin_entropy:
                frame[f"{col} Margin"] = res["margin"][:, i]
        deepest = hierarchy.level_columns[-1]
        frame["Confidence"] = frame[f"{deepest} Confidence"]
        if not predict_all_levels:
            keep = [c for c in frame.columns if c.startswith(f"Predicted {deepest}")]
            frame = frame[keep + ["Confidence"]]
        return frame

    # One pass. The top-k answer and the prefix masses are two readings
    # of the same score block, so they are taken together.
    top = hierarchy.score(
        level_probs, mode, normalise, top_k=top_k,
        with_prefixes=predict_all_levels, batch_size=batch_size, verbose=verbose,
    )
    best_codes = top["path_codes"][:, 0, :]
    frame = hierarchy.decode(best_codes)
    frame["Confidence"] = top["probability"][:, 0]
    frame["Path Evidence"] = top["path_evidence"][:, 0]

    if top_k > 1:
        for rank in range(1, top["path_codes"].shape[1]):
            alt = hierarchy.decode(top["path_codes"][:, rank, :], prefix=f"Alt{rank} ")
            frame = pd.concat([frame, alt], axis=1)
            frame[f"Alt{rank} Probability"] = top["probability"][:, rank]

    if predict_all_levels:
        for depth, info in top["prefixes"].items():
            col = hierarchy.level_columns[depth - 1]
            frame[f"{col} Prefix Probability"] = info["probability"]
            if include_margin_entropy:
                frame[f"{col} Prefix Margin"] = info["margin"]
    else:
        deepest = hierarchy.level_columns[-1]
        keep = [f"Predicted {deepest}", "Confidence", "Path Evidence"]
        frame = frame[[c for c in keep if c in frame.columns]]

    if include_margin_entropy:
        frame["Entropy"] = _entropy(level_probs[-1], batch_size=batch_size)
    return frame


def _entropy(probs: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
    """Shannon entropy of the deepest level's distribution, in nats.

    A second opinion on confidence, and they disagree usefully. A row
    can have a top-1 probability of 0.55 because one rival is at 0.45
    (low entropy, a genuine two-way call) or because four hundred
    categories share the rest (high entropy, the model has no idea).
    Only entropy tells those apart.

    Batched, because the one-line version is not affordable at scale:
    `clip` copies the whole array and `p * log(p)` copies it twice
    more, which on 526,670 x 3,473 is 20 GB for a result of 2 MB.
    """
    n_rows, n_cols = probs.shape
    batch = batch_size or max(1, int((64 * 1024 ** 2) // max(n_cols * 4, 1)))
    out = np.empty(n_rows, dtype=np.float32)

    for lo in range(0, n_rows, batch):
        hi = min(lo + batch, n_rows)
        block = np.array(probs[lo:hi], dtype=np.float32)   # one block-sized copy
        np.clip(block, EPS, None, out=block)
        logs = np.log(block)
        block *= logs
        out[lo:hi] = -block.sum(axis=1)
        del block, logs
    return out
