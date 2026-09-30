"""How much of the remaining error is unlearnable.

Eight sparse configurations -- unigrams, four bigram vocabularies,
three weighting schemes, and character 3-5 grams -- all landed between
0.735 and 0.741 at level 4, with top-5 pinned at 0.913 in every one.
Representations that share almost no features do not agree that
closely by accident. The obvious remaining explanation is that the
descriptions themselves do not determine the label.

This module measures that directly, and it needs no model to do it.

The argument is simple. If two rows have the *same* cleaned
description but different level-4 IDs, no function of the cleaned
description can get both right. The best any classifier can do on
that group is predict its most common label. Summed over every
group, that is an upper bound on accuracy -- the Bayes rate under
this representation -- and it is computable from the labels alone.

Three warnings about what the number means.

It is an upper bound on *this* representation, not on the problem.
The bound is computed on cleaned text, so every token the cleaner
dropped -- part numbers, sizes, unrecognised words -- counts as
information thrown away rather than information absent. Loosening
the cleaning raises the bound. That is the point of also reporting
the bound on raw text: the gap between the two is the price of
cleaning, stated in the same units as accuracy.

It is a bound, not a target. Reaching it would mean predicting the
majority label for every ambiguous group correctly, which no real
model does.

And a description seen once is never ambiguous by this measure,
because it has nothing to disagree with. On a catalogue with a long
tail of unique descriptions the bound is loose, so `coverage` --
the share of rows in groups of two or more -- is reported next to
it and should be read first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class AmbiguityReport:
    """The bound, and enough context to know whether to believe it."""

    level: str
    n_rows: int
    n_groups: int
    # Rows whose description is shared with at least one other row.
    n_rows_in_shared_groups: int
    # Rows whose description is shared AND carries more than one label.
    n_ambiguous_rows: int
    n_ambiguous_groups: int
    # Rows that even a perfect majority-vote predictor must get wrong.
    n_unwinnable_rows: int
    bound: float
    text_col: str = "text"

    @property
    def coverage(self) -> float:
        """Share of rows that a duplicate could have exposed at all."""
        return self.n_rows_in_shared_groups / max(self.n_rows, 1)

    @property
    def ambiguous_share(self) -> float:
        return self.n_ambiguous_rows / max(self.n_rows, 1)

    def to_row(self) -> Dict[str, object]:
        return {
            "level": self.level,
            "rows": self.n_rows,
            "distinct texts": self.n_groups,
            "rows with a duplicate": self.n_rows_in_shared_groups,
            "duplicate coverage": round(self.coverage, 4),
            "ambiguous rows": self.n_ambiguous_rows,
            "ambiguous share": round(self.ambiguous_share, 4),
            "unwinnable rows": self.n_unwinnable_rows,
            "accuracy bound": round(self.bound, 4),
        }


def _normalise_text(series: pd.Series) -> pd.Series:
    """Collapse whitespace and case so trivial differences do not split groups.

    Deliberately conservative: it does not reorder tokens. Two
    descriptions with the same words in a different order are
    genuinely different strings to every vectorizer in this repo
    except the folded-bigram one, so treating them as one group here
    would overstate the ambiguity.
    """
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )


def ambiguity_report(
    df: pd.DataFrame,
    text_col: str,
    level: str,
    normalise: bool = True,
    min_group_size: int = 2,
) -> AmbiguityReport:
    """The majority-vote accuracy bound for one level.

    Groups rows by cleaned description. Within each group the best
    possible prediction is the modal label, so the rows carrying any
    other label are unwinnable. One minus their share is the bound.

    Singleton groups contribute a win each, which is what makes this
    an upper bound rather than an estimate: a unique description is
    assumed perfectly predictable, and in practice most are not.
    """
    work = pd.DataFrame({
        "text": _normalise_text(df[text_col]) if normalise
        else df[text_col].fillna("").astype(str),
        "label": df[level].astype(str).to_numpy(),
    })

    sizes = work.groupby("text", sort=False)["label"].size()
    # Rows per (text, label) pair; the largest per text is the modal count.
    pair_counts = work.groupby(["text", "label"], sort=False).size()
    modal = pair_counts.groupby(level="text", sort=False).max()
    n_labels = pair_counts.groupby(level="text", sort=False).size()

    n_rows = len(work)
    unwinnable = int((sizes - modal).sum())

    shared = sizes[sizes >= min_group_size]
    ambiguous_texts = n_labels[n_labels > 1].index
    n_ambiguous_rows = int(sizes.reindex(ambiguous_texts).sum())

    return AmbiguityReport(
        level=level,
        n_rows=n_rows,
        n_groups=int(len(sizes)),
        n_rows_in_shared_groups=int(shared.sum()),
        n_ambiguous_rows=n_ambiguous_rows,
        n_ambiguous_groups=int(len(ambiguous_texts)),
        n_unwinnable_rows=unwinnable,
        bound=1.0 - unwinnable / max(n_rows, 1),
        text_col=text_col,
    )


def ambiguity_by_level(
    df: pd.DataFrame,
    text_col: str,
    level_columns: Sequence[str],
    normalise: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """One bound per level, plus a full-path bound.

    The path bound is the one to compare against `path_accuracy`, and
    it is *not* the product of the per-level bounds -- the levels are
    nested, so a description ambiguous at level 4 may be perfectly
    determined at level 1. Expect the bound to fall with depth.
    """
    rows = []
    for level in level_columns:
        rows.append(ambiguity_report(df, text_col, level, normalise).to_row())

    path = df[list(level_columns)].astype(str).agg("|".join, axis=1)
    path_df = pd.DataFrame({text_col: df[text_col], "__path__": path})
    rows.append(
        ambiguity_report(path_df, text_col, "__path__", normalise).to_row()
    )
    rows[-1]["level"] = "full path"

    frame = pd.DataFrame(rows)
    if verbose:
        deepest = frame.iloc[-2]
        print(
            f"[ambiguity] {deepest['level']}: bound {deepest['accuracy bound']:.4f} "
            f"({deepest['unwinnable rows']:,} rows unwinnable of "
            f"{deepest['rows']:,})"
        )
        print(
            f"[ambiguity] {deepest['duplicate coverage']:.1%} of rows share a "
            "description with at least one other row -- the bound is only as "
            "tight as this is large"
        )
    return frame


def cleaning_cost(
    df: pd.DataFrame,
    raw_text_col: str,
    clean_text_col: str,
    level: str,
    verbose: bool = True,
) -> pd.DataFrame:
    """What the cleaning gave up, priced in accuracy.

    Same bound computed twice: once on the raw descriptions, once on
    the cleaned ones. The difference is information the cleaner
    removed that was doing discriminative work -- descriptions that
    were distinguishable before `filter_to_known_tokens` dropped the
    part numbers and are not distinguishable after.

    A large gap here is the argument for loosening the cleaning. A
    small one says the discarded tokens were genuinely noise, and the
    cleaning is not what is holding the model back.
    """
    raw = ambiguity_report(df, raw_text_col, level).to_row()
    clean = ambiguity_report(df, clean_text_col, level).to_row()
    raw["text"] = "raw"
    clean["text"] = "cleaned"

    frame = pd.DataFrame([raw, clean])
    frame = frame[["text"] + [c for c in frame.columns if c != "text"]]
    if verbose:
        gap = raw["accuracy bound"] - clean["accuracy bound"]
        print(
            f"[ambiguity] cleaning costs {gap:+.4f} of the {level} bound "
            f"({raw['accuracy bound']:.4f} raw -> {clean['accuracy bound']:.4f})"
        )
    return frame


def ambiguous_examples(
    df: pd.DataFrame,
    text_col: str,
    level: str,
    n: int = 20,
    normalise: bool = True,
    name_lookup: Optional[pd.DataFrame] = None,
    name_col: Optional[str] = None,
) -> pd.DataFrame:
    """The worst offenders, for reading.

    Ranked by how many rows each ambiguous description accounts for,
    because that is what it costs. A description appearing 400 times
    across three categories matters more than forty appearing twice.

    This is the output to actually look at. The bound says how much
    error is structural; these rows say *why*, and whether the answer
    is a taxonomy fix, a cleaning change, or nothing.
    """
    work = pd.DataFrame({
        "text": _normalise_text(df[text_col]) if normalise
        else df[text_col].fillna("").astype(str),
        "label": df[level].astype(str).to_numpy(),
    })

    grouped = work.groupby("text", sort=False)["label"]
    summary = pd.DataFrame({
        "rows": grouped.size(),
        "n_labels": grouped.nunique(),
    })
    summary = summary[summary["n_labels"] > 1]
    if summary.empty:
        return pd.DataFrame(columns=["text", "rows", "n_labels", "labels"])

    summary = summary.sort_values(["rows", "n_labels"], ascending=False).head(n)

    labels = (
        work[work["text"].isin(summary.index)]
        .groupby(["text", "label"], sort=False)
        .size()
        .reset_index(name="count")
    )
    if name_lookup is not None and name_col is not None:
        names = (
            name_lookup[[level, name_col]]
            .astype(str)
            .drop_duplicates(subset=[level])
            .set_index(level)[name_col]
        )
        labels["label"] = labels["label"] + " (" + labels["label"].map(names).fillna("?") + ")"

    joined = (
        labels.sort_values("count", ascending=False)
        .groupby("text", sort=False)
        .apply(lambda g: ", ".join(f"{r.label} x{r.count}" for r in g.itertuples()),
               include_groups=False)
    )

    out = summary.copy()
    out["labels"] = joined
    return out.reset_index()


def _bucket_rows(
    text: np.ndarray,
    labels: np.ndarray,
    reference: Optional[pd.DataFrame],
    text_col: str,
    level: str,
    normalise: bool,
) -> "pd.Series":
    """Label each row with the group whose ceiling it inherits.

    Two modes, and the choice matters more than it looks.

    Without a `reference`, rows are grouped against each other: a
    description appearing once in the frame being scored is a
    `singleton`. That is the weaker reading, because a description
    unique *within validation* may have appeared fifty times in
    training, where the model memorised it. Calling that a
    generalisation test is wrong.

    With a `reference` -- the training rows -- each scored row is
    classified by what training knew about its description:

      `unseen in training`      the real generalisation test. No
                                ceiling applies; the model has to
                                infer from tokens, never from the
                                whole string.
      `seen, one label`         training showed this description and
                                always with the same label. The
                                ceiling is 1.0 and the model has no
                                excuse.
      `seen, several labels`    training showed it with conflicting
                                labels. Structurally capped.

    That three-way split is the one that decides what to do next, so
    pass the reference whenever you have it.
    """
    frame = pd.DataFrame({"text": text, "label": labels})

    if reference is None:
        grouped = frame.groupby("text", sort=False)["label"]
        n_labels = grouped.nunique()
        sizes = grouped.size()
        mapped_labels = frame["text"].map(n_labels).to_numpy()
        mapped_sizes = frame["text"].map(sizes).to_numpy()
        return pd.Series(
            np.where(
                mapped_sizes < 2, "singleton",
                np.where(mapped_labels > 1, "ambiguous", "consistent duplicate"),
            ),
            index=frame.index,
        )

    ref_text = (
        _normalise_text(reference[text_col]) if normalise
        else reference[text_col].fillna("").astype(str)
    )
    ref = pd.DataFrame({"text": ref_text.to_numpy(),
                        "label": reference[level].astype(str).to_numpy()})
    ref_labels = ref.groupby("text", sort=False)["label"].nunique()

    seen = frame["text"].map(ref_labels)
    return pd.Series(
        np.where(
            seen.isna(), "unseen in training",
            np.where(seen.fillna(0).to_numpy() > 1,
                     "seen, several labels", "seen, one label"),
        ),
        index=frame.index,
    )


def bucket_rows(
    truth: pd.DataFrame,
    text_col: str,
    level: str,
    reference: Optional[pd.DataFrame] = None,
    normalise: bool = True,
) -> pd.Series:
    """The bucket each row belongs to, as a Series aligned to `truth`.

    The public form of what `error_overlap` groups by. Exposed so the
    same split can be applied to anything else measured on the same
    rows -- a second model, a lookup table, the raw per-level argmax
    before the hierarchy constrains it -- without recomputing the
    grouping slightly differently each time and quietly comparing
    numbers that were bucketed by different rules.
    """
    text = (
        _normalise_text(truth[text_col]) if normalise
        else truth[text_col].fillna("").astype(str)
    ).to_numpy()
    labels = truth[level].astype(str).to_numpy()
    out = _bucket_rows(text, labels, reference, text_col, level, normalise)
    out.index = truth.index
    return out


_BUCKET_ORDER = [
    "seen, one label", "seen, several labels", "unseen in training",
    "consistent duplicate", "ambiguous", "singleton",
]


def error_overlap(
    predictions: pd.DataFrame,
    truth: pd.DataFrame,
    text_col: str,
    level: str,
    reference: Optional[pd.DataFrame] = None,
    normalise: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Split a model's errors by what could have been known about them.

    The bound says what fraction of rows cannot be got right. This
    says where the errors the model actually made fall -- which is
    the question that decides whether to keep modelling.

    An earlier version of this split two ways, ambiguous against
    everything else, and that was too coarse to act on: it put
    descriptions the model had memorised in training in the same
    bucket as descriptions it had never seen, and reported a single
    accuracy over both against a ceiling of 1.0. On this catalogue
    138,035 of 585,193 rows have a description appearing exactly
    once, so the two halves of that bucket are nothing alike.

    Pass `reference` -- the training frame -- to get the split that
    matters:

      high error share on `seen, one label`
          the model is failing rows it was shown the answer to. Real,
          addressable headroom; keep modelling.
      high error share on `unseen in training`
          the model generalises poorly to descriptions it has not
          met. The fix is more informative tokens, not a bigger net.
      high error share on `seen, several labels`
          structural. No model fixes it; the work is upstream.

    `predictions` needs a `Predicted <level>` column, and it must be
    row-aligned to `truth`.
    """
    pred_col = f"Predicted {level}"
    if pred_col not in predictions.columns:
        raise ValueError(f"predictions has no {pred_col!r} column.")
    if len(predictions) != len(truth):
        raise ValueError(
            f"predictions has {len(predictions):,} rows and truth has "
            f"{len(truth):,}; they must be aligned."
        )
    if reference is not None:
        for col in (text_col, level):
            if col not in reference.columns:
                raise ValueError(f"reference has no {col!r} column.")

    text = (
        _normalise_text(truth[text_col]) if normalise
        else truth[text_col].fillna("").astype(str)
    ).to_numpy()
    y_true = truth[level].astype(str).to_numpy()
    y_pred = predictions[pred_col].astype(str).to_numpy()
    correct = y_pred == y_true
    n_errors = int((~correct).sum())

    bucket = _bucket_rows(text, y_true, reference, text_col, level, normalise)

    # The ceiling each bucket inherits.
    #
    #   one label      1.0 -- the description determines the answer.
    #   several labels the majority-vote bound over just these rows,
    #                  computed the same way as ambiguity_report. Not
    #                  1.0, and not zero either: even a capped bucket
    #                  has headroom, and this says how much.
    #   unseen /
    #   singleton      nothing constrains them, so NaN rather than a
    #                  1.0 that would invent headroom out of nothing.
    fixed_ceilings = {
        "seen, one label": 1.0,
        "consistent duplicate": 1.0,
        "singleton": float("nan"),
        "unseen in training": float("nan"),
    }

    def _bucket_ceiling(name: str, mask: np.ndarray) -> float:
        if name in fixed_ceilings:
            return fixed_ceilings[name]
        sub = pd.DataFrame({"text": text[mask], "label": y_true[mask]})
        sizes = sub.groupby("text", sort=False)["label"].size()
        modal = (sub.groupby(["text", "label"], sort=False).size()
                 .groupby(level="text", sort=False).max())
        return 1.0 - float((sizes - modal).sum()) / max(int(mask.sum()), 1)

    rows = []
    for name in _BUCKET_ORDER:
        mask = (bucket == name).to_numpy()
        n = int(mask.sum())
        if not n:
            continue
        n_wrong = int((~correct & mask).sum())
        rows.append({
            "description group": name,
            "rows": n,
            "share of rows": round(n / max(len(bucket), 1), 4),
            "accuracy": round(float(correct[mask].mean()), 4),
            "ceiling": round(_bucket_ceiling(name, mask), 4),
            "errors": n_wrong,
            "share of all errors": round(n_wrong / max(n_errors, 1), 4),
        })

    out = pd.DataFrame(rows)
    if verbose and len(out):
        for _, r in out.iterrows():
            ceil = r["ceiling"]
            tail = (f", ceiling {ceil:.2f}, {ceil - r['accuracy']:+.4f} headroom"
                    if ceil == ceil else ", no ceiling applies")
            print(
                f"[ambiguity] {r['description group']:<21} "
                f"{r['share of rows']:>6.1%} of rows, accuracy {r['accuracy']:.4f}"
                f"{tail}"
            )
        worst = out.loc[out["share of all errors"].idxmax()]
        print(
            f"[ambiguity] most errors ({worst['share of all errors']:.1%}) fall "
            f"on '{worst['description group']}'"
        )
    return out


def summarise(
    bounds: pd.DataFrame,
    achieved: Dict[str, float],
    verbose: bool = True,
) -> pd.DataFrame:
    """Bound against achieved, per level, with the gap that remains.

    `achieved` maps level name to the accuracy a model actually got.
    The `headroom` column is what is left on the table under this
    representation -- and it is the only number in this repo that
    says whether to keep modelling.
    """
    rows = []
    for _, r in bounds.iterrows():
        level = str(r["level"])
        got = achieved.get(level)
        if got is None:
            continue
        rows.append({
            "level": level,
            "achieved": round(float(got), 4),
            "bound": float(r["accuracy bound"]),
            "headroom": round(float(r["accuracy bound"]) - float(got), 4),
            "share of gap realised":
                round(float(got) / float(r["accuracy bound"]), 4)
                if r["accuracy bound"] else float("nan"),
        })
    out = pd.DataFrame(rows)
    if verbose and len(out):
        worst = out.iloc[-1]
        print(
            f"[ambiguity] {worst['level']}: {worst['achieved']:.4f} achieved "
            f"against a {worst['bound']:.4f} ceiling -- "
            f"{worst['headroom']:+.4f} of headroom"
        )
    return out
