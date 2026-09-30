"""Memorising what the network already memorised, but explicitly.

Notebook 09 split the validation rows by what training knew about
their description, and the result reframed the whole problem:

    seen, one label        41.6% of rows   accuracy 0.9313
    seen, several labels   33.9% of rows   accuracy 0.6382 (ceiling 0.7319)
    unseen in training     24.6% of rows   accuracy 0.5512

The network scores 0.93 on strings it has met and 0.55 on strings it
has not. It is behaving like a lookup table with a weak fallback,
which is also why eight different feature spaces all landed on
0.735-0.741: they were all doing the same job.

If that is what the model is, an explicit table should do it at least
as well, and the one place it plausibly does it *better* is the
middle row -- descriptions seen with conflicting labels, where the
network manages 0.6382 against a majority-vote ceiling of 0.7319.

Three honest caveats, because this is easy to oversell.

**The top row is not free.** "One label in training" does not mean
unambiguous: the split can put every instance of label A in training
and one stray B in validation, and no text-only function recovers
that row. So the 1.0 ceiling that notebook 09 prints for that bucket
is optimistic, and `true_ceiling` below measures the real one.

**The middle row's gap may not be a memorisation failure at all.**
Predictions come from `predictions_frame(mode="independent")`, which
multiplies four levels and picks the best valid path -- that can
override the level-4 head's own argmax, and we have already seen it
cost 0.0036 there. If the raw argmax is already near 0.73, the
hierarchy is the culprit and there is no table worth building.
`compare_by_bucket` exists to settle that before anything is adopted.

**A table generalises to nothing.** Every row it fires on is a row
the model already handles well. It cannot touch the 24.6% of rows
with an unseen description, which is where most of the real loss is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .ambiguity import _normalise_text


@dataclass
class DescriptionLookup:
    """Cleaned description -> the modal full path that carried it.

    The modal *path*, not the modal label at each level independently.
    Taking per-level modes could assemble a combination that appears
    nowhere in the taxonomy -- level 2 from one branch and level 3
    from another -- and every consumer downstream assumes predictions
    are valid paths. Storing whole paths makes that impossible by
    construction rather than by assertion.

    `support` is how many training rows carried the description and
    `purity` the share of them agreeing on the modal path. Together
    they are the gate: purity alone would trust a description seen
    once, which is a coin landing heads and calling itself certain.
    """

    paths: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    support: Dict[str, int] = field(default_factory=dict)
    purity: Dict[str, float] = field(default_factory=dict)
    level_columns: Tuple[str, ...] = ()
    normalise: bool = True
    n_training_rows: int = 0

    def __len__(self) -> int:
        return len(self.paths)

    # -- construction ------------------------------------------------

    @classmethod
    def fit(
        cls,
        df: pd.DataFrame,
        text_col: str,
        level_columns: Sequence[str],
        normalise: bool = True,
        verbose: bool = True,
    ) -> "DescriptionLookup":
        """Build the table from the TRAINING rows only.

        Fitting on anything wider is the same leak as fitting a
        vectorizer on the validation set, and a far more direct one:
        the table would contain the answers it is about to be graded
        on, and would score near 1.0 while being worthless.
        """
        levels = list(level_columns)
        text = (
            _normalise_text(df[text_col]) if normalise
            else df[text_col].fillna("").astype(str)
        )
        path = df[levels].astype(str).agg("\x1f".join, axis=1)

        work = pd.DataFrame({"text": text.to_numpy(), "path": path.to_numpy()})
        counts = work.groupby(["text", "path"], sort=False).size()
        totals = counts.groupby(level="text", sort=False).sum()
        # idxmax over a MultiIndex returns the whole (text, path) key,
        # not the path alone -- hence the [1]. Getting this wrong
        # raises rather than corrupting, which is the good case.
        modal = counts.groupby(level="text", sort=False).idxmax()
        modal_counts = counts.groupby(level="text", sort=False).max()

        paths = {t: tuple(key[1].split("\x1f")) for t, key in modal.items()}
        support = {t: int(n) for t, n in totals.items()}
        pure = {t: float(modal_counts[t]) / float(totals[t]) for t in totals.index}

        obj = cls(
            paths=paths, support=support, purity=pure,
            level_columns=tuple(levels), normalise=normalise,
            n_training_rows=int(len(df)),
        )
        if verbose:
            n_multi = sum(1 for t in pure if pure[t] < 1.0)
            print(
                f"[lookup] {len(paths):,} distinct descriptions from "
                f"{len(df):,} training rows; {n_multi:,} carry more than one path"
            )
        return obj

    # -- use ----------------------------------------------------------

    def _keys(self, texts: Sequence[str]) -> np.ndarray:
        series = pd.Series(list(texts), dtype="object")
        return (
            _normalise_text(series) if self.normalise
            else series.fillna("").astype(str)
        ).to_numpy()

    def predict(
        self,
        texts: Sequence[str],
        min_support: int = 2,
        min_purity: float = 0.0,
    ) -> pd.DataFrame:
        """One row per input: the modal path, or a miss.

        `min_support=2` by default. A description seen once in
        training has purity 1.0 by arithmetic rather than by
        evidence, and letting those fire is how a lookup table
        overfits -- it would claim the whole tail of the catalogue on
        the strength of one example each.
        """
        keys = self._keys(texts)
        n = len(keys)
        hit = np.zeros(n, dtype=bool)
        sup = np.zeros(n, dtype=np.int64)
        pur = np.zeros(n, dtype=np.float32)
        out = {c: np.full(n, "", dtype=object) for c in self.level_columns}

        for i, key in enumerate(keys):
            path = self.paths.get(key)
            if path is None:
                continue
            s, p = self.support[key], self.purity[key]
            sup[i], pur[i] = s, p
            if s < min_support or p < min_purity:
                continue
            hit[i] = True
            for col, value in zip(self.level_columns, path):
                out[col][i] = value

        frame = pd.DataFrame({f"Lookup {c}": out[c] for c in self.level_columns})
        frame["Lookup Hit"] = hit
        frame["Lookup Support"] = sup
        frame["Lookup Purity"] = pur
        return frame

    def coverage(
        self,
        texts: Sequence[str],
        min_support: int = 2,
        min_purity: float = 0.0,
    ) -> float:
        return float(
            self.predict(texts, min_support, min_purity)["Lookup Hit"].mean()
        )

    def true_ceiling(
        self,
        texts: Sequence[str],
        truth: pd.DataFrame,
        level: str,
    ) -> float:
        """What a perfect table would score on rows it fires on.

        The number notebook 09 approximates as 1.0 for the
        `seen, one label` bucket. It is not 1.0: a description whose
        training rows all carried label A can still appear in
        validation carrying B, and nothing reading only the text can
        recover that. This measures the shortfall instead of assuming
        it away.
        """
        pred = self.predict(texts, min_support=1)
        hit = pred["Lookup Hit"].to_numpy()
        if not hit.any():
            return float("nan")
        got = pred.loc[hit, f"Lookup {level}"].to_numpy()
        want = truth.loc[hit, level].astype(str).to_numpy()
        return float((got == want).mean())


def blend(
    frame: pd.DataFrame,
    lookup: DescriptionLookup,
    texts: Sequence[str],
    level_columns: Sequence[str],
    min_support: int = 2,
    min_purity: float = 0.0,
    confidence_col: str = "Confidence",
    verbose: bool = True,
) -> pd.DataFrame:
    """Override the model where the table fires, and record where it did.

    Returns a copy with the `Predicted <level>` columns replaced on
    hit rows, plus a `Source` column reading `lookup` or `model`. The
    provenance column is not decoration: a blended frame that cannot
    say which half produced a row is one nobody can debug, and every
    accuracy computed from it afterwards is unattributable.

    Confidence on overridden rows becomes the table's purity, which
    is a real frequency -- the share of training rows with this
    description that carried this path -- and so is already roughly
    calibrated. It is not comparable to a softmax probability, and
    the cascade thresholds fitted in notebook 03 do not transfer to
    it. Refit them on the blended frame if you intend to threshold.
    """
    out = frame.copy()
    hits = lookup.predict(texts, min_support=min_support, min_purity=min_purity)
    mask = hits["Lookup Hit"].to_numpy()

    out["Source"] = np.where(mask, "lookup", "model")
    for col in level_columns:
        pred_col = f"Predicted {col}"
        if pred_col not in out.columns:
            raise ValueError(f"frame has no {pred_col!r} column.")
        replacement = hits[f"Lookup {col}"].to_numpy()
        out[pred_col] = np.where(mask, replacement, out[pred_col].astype(str))

    if confidence_col in out.columns:
        out[confidence_col] = np.where(
            mask, hits["Lookup Purity"].to_numpy(),
            out[confidence_col].to_numpy(),
        )

    if verbose:
        print(
            f"[lookup] overrode {int(mask.sum()):,} of {len(out):,} rows "
            f"({mask.mean():.1%}) at min_support={min_support}, "
            f"min_purity={min_purity}"
        )
    return out


def compare_by_bucket(
    sources: Mapping[str, Sequence],
    y_true: Sequence,
    buckets: Sequence[str],
    verbose: bool = True,
) -> pd.DataFrame:
    """Accuracy of several prediction sources, per bucket and overall.

    The point of this is the `seen, several labels` column. The
    network scores 0.6382 there against a 0.7319 majority-vote
    ceiling, and there are two very different explanations: it failed
    to memorise the mode, or the hierarchy's path constraint
    overrode a level-4 argmax that had it right. Putting the raw
    argmax, the path-constrained prediction and the table side by
    side on the same rows tells them apart, and they imply different
    work.
    """
    truth = np.asarray(y_true, dtype=object).astype(str)
    bucket = np.asarray(buckets, dtype=object).astype(str)
    names = list(dict.fromkeys(bucket.tolist()))

    rows: List[Dict[str, object]] = []
    for source, values in sources.items():
        pred = np.asarray(values, dtype=object).astype(str)
        if len(pred) != len(truth):
            raise ValueError(
                f"source {source!r} has {len(pred):,} rows, truth has {len(truth):,}."
            )
        correct = pred == truth
        row: Dict[str, object] = {"source": source,
                                  "overall": round(float(correct.mean()), 4)}
        for name in names:
            m = bucket == name
            row[name] = round(float(correct[m].mean()), 4) if m.any() else float("nan")
        rows.append(row)

    out = pd.DataFrame(rows).set_index("source")
    if verbose and "seen, several labels" in out.columns:
        col = out["seen, several labels"]
        print(
            f"[lookup] on 'seen, several labels': best source is "
            f"{col.idxmax()!r} at {col.max():.4f}, worst {col.idxmin()!r} "
            f"at {col.min():.4f}"
        )
    return out
