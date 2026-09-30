"""Turning cleaned text into numbers, three ways.

All three model families share this layer, which is the only reason
their results can be compared at all:

  sparse       BOW / TF-IDF matrices           -> sparse NN, sklearn
  vocab        integer token indices           -> dense NN (EmbeddingBag)
  embeddings   Word2Vec vectors, pooled or not -> dense NN init, sklearn
"""

from .sparse import (  # noqa: F401
    UnorderedBigramCountVectorizer,
    UnorderedBigramTfidfVectorizer,
    build_vectorizer,
    feature_report,
    fit_transform,
    fold_bigrams,
    top_features,
)
from .vocab import TokenVocabulary, build_vocabulary  # noqa: F401
from .embeddings import (  # noqa: F401
    WordVectors,
    build_document_matrices,
    load_or_train_word2vec,
    train_word2vec,
)

__all__ = [
    "TokenVocabulary",
    "UnorderedBigramCountVectorizer",
    "UnorderedBigramTfidfVectorizer",
    "WordVectors",
    "build_document_matrices",
    "build_vectorizer",
    "build_vocabulary",
    "feature_report",
    "fit_transform",
    "fold_bigrams",
    "load_or_train_word2vec",
    "top_features",
    "train_word2vec",
]
