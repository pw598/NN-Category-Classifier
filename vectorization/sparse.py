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


def top_features(vectorizer, n: int = 25) -> List[str]:
    """The n most common features, as a sanity check on the vocabulary.

    Worth an eyeball every time the cleaning changes. If this list is
    full of units, sizes or stray digits, the regex pass is not doing
    its job and no amount of model tuning will make up for it.
    """
    names = np.asarray(vectorizer.get_feature_names_out())
    return list(names[:n])
