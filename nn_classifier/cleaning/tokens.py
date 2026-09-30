"""Counting and filtering that happens after the text is clean.

Everything here is about deciding what is worth learning from. A token
seen once is memorisation, not evidence. A class with three examples
cannot be split into train and validation and still mean anything. The
outline asks for both counts and the filters, and the counts are the
more useful half: they are how you choose the thresholds rather than
guess them.

One rule runs through the whole module: **counts for filtering are
computed on the training split only**, then applied to both splits. A
vocabulary built on all the data has already seen the validation set,
and the leak shows up as a validation score that does not survive
contact with new data.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

DEFAULT_TOKEN_PATTERN = r"\b\w\w+\b"


def tokenize(
    text: str,
    lowercase: bool = True,
    pattern: str | re.Pattern = DEFAULT_TOKEN_PATTERN,
) -> List[str]:
    """Split on the token pattern, dropping punctuation and 1-char fragments."""
    if not isinstance(text, str):
        return []
    rx = pattern if isinstance(pattern, re.Pattern) else re.compile(pattern)
    tokens = rx.findall(text.lower() if lowercase else text)
    return tokens


def _ngrams(tokens: Sequence[str], n: int) -> Iterable[str]:
    if n == 1:
        yield from tokens
        return
    for i in range(len(tokens) - n + 1):
        yield " ".join(tokens[i:i + n])


def count_ngrams(
    texts: Iterable[str],
    ngram_range: Tuple[int, int] = (1, 1),
    lowercase: bool = True,
    pattern: str = DEFAULT_TOKEN_PATTERN,
    labels: Optional[Sequence] = None,
    min_count: int = 1,
    min_classes: int = 0,
) -> pd.DataFrame:
    """A frame of n-grams with document counts, and class spread if labelled.

    Columns: `ngram`, `n`, `doc_count`, and -- when `labels` is given --
    `n_classes`, the number of distinct target classes the n-gram turns
    up under.

    `n_classes` is the column worth looking at. A token appearing in
    5,000 descriptions spread over 400 categories is a stopword in
    everything but name; one appearing in 30 descriptions all in the
    same category is the kind of feature a classifier is built on.
    Filtering on document count alone cannot tell those apart.

    Document counts, not term counts: a word repeated three times in one
    description is one piece of evidence, not three.
    """
    lo, hi = ngram_range
    counts: Counter = Counter()
    class_sets: Dict[str, set] = {}
    labels = list(labels) if labels is not None else None

    for i, text in enumerate(texts):
        tokens = tokenize(text, lowercase=lowercase, pattern=pattern)
        seen = set()
        for n in range(lo, hi + 1):
            seen.update(_ngrams(tokens, n))
        for gram in seen:
            counts[gram] += 1
            if labels is not None:
                class_sets.setdefault(gram, set()).add(labels[i])

    rows = {
        "ngram": list(counts.keys()),
        "n": [g.count(" ") + 1 for g in counts],
        "doc_count": list(counts.values()),
    }
    out = pd.DataFrame(rows)
    if labels is not None:
        out["n_classes"] = out["ngram"].map(lambda g: len(class_sets.get(g, ())))

    if min_count > 1:
        out = out[out["doc_count"] >= min_count]
    if min_classes > 0 and "n_classes" in out.columns:
        out = out[out["n_classes"] >= min_classes]

    return out.sort_values("doc_count", ascending=False).reset_index(drop=True)


def token_report(
    texts: Iterable[str],
    ngram_range: Tuple[int, int] = (1, 1),
    labels: Optional[Sequence] = None,
    thresholds: Sequence[int] = (1, 2, 3, 5, 10),
    lowercase: bool = True,
) -> pd.DataFrame:
    """How many n-grams survive each minimum-occurrence threshold.

    The answer to 'what should min_df be?'. Vocabulary size usually
    falls off a cliff between 1 and 2 -- most of a catalogue's
    vocabulary is part numbers seen exactly once -- and then flattens.
    Printing the curve turns that choice into an observation.
    """
    counts = count_ngrams(texts, ngram_range, lowercase=lowercase, labels=labels)
    rows = []
    for thr in thresholds:
        kept = counts[counts["doc_count"] >= thr]
        row = {
            "min_doc_count": thr,
            "n_ngrams": len(kept),
            "share_of_total": len(kept) / len(counts) if len(counts) else 0.0,
        }
        if "n_classes" in counts.columns and len(kept):
            row["median_classes_per_ngram"] = float(kept["n_classes"].median())
        rows.append(row)
    return pd.DataFrame(rows)


def drop_rare_classes(
    df: pd.DataFrame,
    label_col: str,
    min_count: int = 6,
    verbose: bool = True,
) -> pd.DataFrame:
    """Remove rows whose class has fewer than min_count members.

    Two jobs at once. A class with three examples is not learnable, and
    more immediately, a stratified split needs at least two members per
    class -- and cross-validation needs at least `n_folds`. Setting this
    to 6 is what makes a 5-fold split possible with one example spare.
    """
    if min_count <= 1:
        return df.reset_index(drop=True)

    counts = df[label_col].value_counts()
    keep_classes = counts[counts >= min_count].index
    keep = df[label_col].isin(keep_classes)
    out = df[keep].reset_index(drop=True)

    if verbose:
        n_dropped_rows = int((~keep).sum())
        n_dropped_classes = int(len(counts) - len(keep_classes))
        share = n_dropped_rows / len(df) if len(df) else 0.0
        print(
            f"[tokens] dropped {n_dropped_classes:,} of {len(counts):,} classes at "
            f"'{label_col}' seen < {min_count} times "
            f"({n_dropped_rows:,} rows, {share:.2%})"
        )
    return out


def drop_rare_paths(
    df: pd.DataFrame,
    level_columns: Sequence[str],
    min_count: int = 2,
    verbose: bool = True,
) -> pd.DataFrame:
    """Remove rows whose full hierarchy path occurs fewer than min_count times.

    Distinct from drop_rare_classes: two rows can share a deepest-level
    class and still sit on different paths if the data has an
    inconsistency upstream. Joint-path scoring can only ever emit a path
    it saw in training, so a path with one example is a class the model
    will hallucinate from a single description.
    """
    level_columns = list(level_columns)
    if min_count <= 1:
        return df.reset_index(drop=True)

    sizes = df.groupby(level_columns, dropna=False)[level_columns[0]].transform("size")
    keep = sizes >= min_count
    if verbose and (~keep).any():
        before = df[level_columns].drop_duplicates().shape[0]
        after = df[keep][level_columns].drop_duplicates().shape[0]
        print(
            f"[tokens] dropped {int((~keep).sum()):,} rows on paths seen "
            f"< {min_count} times ({before:,} -> {after:,} distinct paths)"
        )
    return df[keep].reset_index(drop=True)


def build_keep_vocabulary(
    train_texts: Iterable[str],
    min_count: int = 2,
    min_classes: int = 0,
    train_labels: Optional[Sequence] = None,
    lowercase: bool = True,
) -> set:
    """The set of tokens worth keeping, measured on the training split only."""
    counts = count_ngrams(
        train_texts,
        ngram_range=(1, 1),
        lowercase=lowercase,
        labels=train_labels if min_classes > 0 else None,
        min_count=min_count,
        min_classes=min_classes,
    )
    return set(counts["ngram"])


def prune_to_vocabulary(
    series: pd.Series,
    vocabulary: set,
    lowercase: bool = True,
) -> pd.Series:
    """Keep only the tokens in `vocabulary`, preserving order and rows."""
    def _prune(text: str) -> str:
        tokens = tokenize(text, lowercase=lowercase)
        return " ".join(t for t in tokens if t in vocabulary)

    return series.fillna("").astype(str).map(_prune)


def drop_one_off_tokens(
    train_df: pd.DataFrame,
    other_dfs: Sequence[pd.DataFrame] = (),
    text_col: str = "FullDesc",
    min_count: int = 2,
    min_classes: int = 0,
    label_col: Optional[str] = None,
    lowercase: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[pd.DataFrame], set]:
    """Prune rare tokens from every split, using training counts only.

    Returns `(train, [others...], vocabulary)`. Call this *after* the
    split, never before. Counting across the whole dataset lets the
    validation set vote on which of its own tokens survive, and the
    resulting score is optimistic in a way no later check will catch.
    """
    if min_count <= 1 and min_classes <= 0:
        return train_df.reset_index(drop=True), [d.reset_index(drop=True) for d in other_dfs], set()

    labels = train_df[label_col] if (label_col and min_classes > 0) else None
    vocab = build_keep_vocabulary(
        train_df[text_col], min_count=min_count, min_classes=min_classes,
        train_labels=labels, lowercase=lowercase,
    )

    train_out = train_df.copy()
    train_out[text_col] = prune_to_vocabulary(train_out[text_col], vocab, lowercase)
    others_out = []
    for d in other_dfs:
        dd = d.copy()
        dd[text_col] = prune_to_vocabulary(dd[text_col], vocab, lowercase)
        others_out.append(dd.reset_index(drop=True))

    if verbose:
        n_blank = int((train_out[text_col].str.strip() == "").sum())
        print(
            f"[tokens] kept {len(vocab):,} tokens seen >= {min_count} time(s)"
            + (f" under >= {min_classes} classes" if min_classes > 0 else "")
            + f"; {n_blank:,} training rows left empty"
        )
    return train_out.reset_index(drop=True), others_out, vocab
