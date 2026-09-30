"""BOW and TF-IDF, shared by the sparse network and any sklearn model.

Ported from the old repo's vectorizers.py, with the order-insensitive
bigram folding intact. The reasoning there was that a catalogue writes
the same product both ways -- 'BALL VALVE' and 'VALVE BALL' -- and
treating them as two features halves the evidence for each. Trigrams
and longer keep their order, because by then the order usually does
carry meaning.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from ..config import SparseVectorizerConfig


def fold_bigrams(tokens: List[str]) -> List[str]:
    """Sort the two halves of every bigram so word order stops mattering."""
    out = []
    for tok in tokens:
        parts = tok.split(" ")
        if len(parts) == 2:
            out.append(" ".join(sorted(parts)))
        else:
            out.append(tok)
    return out


class UnorderedBigramCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        base = super().build_analyzer()
        return lambda doc: fold_bigrams(base(doc))


class UnorderedBigramTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        base = super().build_analyzer()
        return lambda doc: fold_bigrams(base(doc))


def build_vectorizer(cfg: SparseVectorizerConfig):
    """Construct the vectorizer described by the config.

    TF-IDF-only parameters are not passed to CountVectorizer, so the
    same config object can describe either without raising.
    """
    common: Dict[str, Any] = dict(
        max_features=cfg.max_features,
        ngram_range=tuple(cfg.ngram_range),
        min_df=cfg.min_df,
        max_df=cfg.max_df,
        stop_words=cfg.stop_words,
        lowercase=cfg.lowercase,
        binary=cfg.binary,
    )
    common.update(cfg.extra or {})

    if cfg.kind == "count":
        klass = UnorderedBigramCountVectorizer if cfg.unordered_bigrams else CountVectorizer
        return klass(**common)
    if cfg.kind == "tfidf":
        klass = UnorderedBigramTfidfVectorizer if cfg.unordered_bigrams else TfidfVectorizer
        return klass(
            sublinear_tf=cfg.sublinear_tf,
            norm=cfg.norm,
            use_idf=cfg.use_idf,
            **common,
        )
    raise ValueError(f"Unknown vectorizer kind: {cfg.kind!r}; expected 'tfidf' or 'count'.")


def fit_transform(
    cfg: SparseVectorizerConfig,
    train_texts: Sequence[str],
    *other_texts: Sequence[str],
    verbose: bool = True,
) -> Tuple[Any, sparse.csr_matrix, Tuple[sparse.csr_matrix, ...]]:
    """Fit on the training texts only, then transform everything.

    Returns `(vectorizer, X_train, (X_other, ...))`. The signature is
    shaped this way to make the leak hard to write: there is nowhere to
    pass the validation texts that would cause them to be fitted on.
    """
    vec = build_vectorizer(cfg)
    X_train = vec.fit_transform(train_texts)
    others = tuple(vec.transform(t) for t in other_texts)
    if verbose:
        print(
            f"[vec] {cfg.kind} vocabulary: {len(vec.vocabulary_):,} features "
            f"(ngram_range={tuple(cfg.ngram_range)}, min_df={cfg.min_df})"
        )
        print(f"[vec] X_train: {X_train.shape}, density={X_train.nnz / np.prod(X_train.shape):.5f}")
    return vec, X_train, others


def _feature_weights(
    vectorizer,
    X: Optional[sparse.spmatrix],
    by: str,
) -> Tuple[np.ndarray, str]:
    """Per-feature commonness, and a label saying how it was measured.

    Three sources, in descending order of directness.

    `X` is the honest one: document frequency is the column-wise
    non-zero count, total weight is the column sum.

    Without `X`, a fitted TF-IDF vectorizer still knows. `idf_` is
    `log((1 + n) / (1 + df)) + 1`, strictly decreasing in `df`, so
    ranking by ascending `idf_` recovers the document-frequency order
    exactly even though the counts themselves are gone.

    A CountVectorizer with no `X` knows nothing about frequency at
    all, and this returns zeros rather than inventing an order. The
    caller is responsible for saying so.
    """
    names = np.asarray(vectorizer.get_feature_names_out())

    if X is not None:
        if X.shape[1] != len(names):
            raise ValueError(
                f"X has {X.shape[1]:,} columns but the vectorizer has "
                f"{len(names):,} features. Pass the matrix this vectorizer "
                "produced, not one from a different fit."
            )
        csr = X.tocsr()
        if by == "documents":
            return csr.getnnz(axis=0).astype(np.float64), "documents"
        if by == "total":
            return np.asarray(csr.sum(axis=0)).ravel().astype(np.float64), "total weight"
        raise ValueError(f"by={by!r}; expected 'documents' or 'total'.")

    idf = getattr(vectorizer, "idf_", None)
    if idf is not None:
        # Monotone in df, so negating gives the right order. The values
        # are not counts and are not reported as such.
        return -np.asarray(idf, dtype=np.float64), "idf rank"

    return np.zeros(len(names), dtype=np.float64), "unranked"


def feature_report(
    vectorizer,
    X: Optional[sparse.spmatrix] = None,
    n: int = 40,
    by: str = "documents",
):
    """The n most common features with their counts, as a frame.

    The diagnostic form of `top_features`. Read `share` before `count`:
    a feature in 40% of the catalogue is a stopword in everything but
    name, however domain-specific it looks.
    """
    import pandas as pd

    names = np.asarray(vectorizer.get_feature_names_out())
    weights, how = _feature_weights(vectorizer, X, by)
    order = np.argsort(-weights, kind="stable")[: min(n, len(names))]

    frame = pd.DataFrame({"feature": names[order], how: weights[order]})
    if X is not None and by == "documents":
        frame["share"] = frame["documents"] / X.shape[0]
    frame.attrs["ranked_by"] = how
    return frame


def top_features(
    vectorizer,
    n: int = 25,
    X: Optional[sparse.spmatrix] = None,
    by: str = "documents",
) -> List[str]:
    """The n most common features, as a sanity check on the vocabulary.

    Worth an eyeball every time the cleaning changes. If this list is
    full of units, sizes or stray digits, the regex pass is not doing
    its job and no amount of model tuning will make up for it.

    Pass `X` -- the matrix this vectorizer produced -- for true
    document-frequency order. Without it, a TF-IDF vectorizer is still
    ranked exactly right via `idf_`; a CountVectorizer cannot be, and
    warns rather than returning an alphabetical slice that looks like
    a frequency ranking.

    That warning exists because this function did exactly that for its
    first several revisions: it sliced `get_feature_names_out()`, which
    sklearn returns sorted alphabetically, while the docstring promised
    the most common features. Anything read out of it before now was an
    alphabetical head, not a frequency ranking.
    """
    names = np.asarray(vectorizer.get_feature_names_out())
    weights, how = _feature_weights(vectorizer, X, by)

    if how == "unranked":
        import warnings

        warnings.warn(
            "top_features cannot rank a CountVectorizer without the matrix "
            "it produced; returning features in alphabetical order. Pass "
            "X=<the matrix> for document-frequency order.",
            stacklevel=2,
        )
        return list(names[:n])

    order = np.argsort(-weights, kind="stable")[: min(n, len(names))]
    return list(names[order])
