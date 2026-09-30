"""Word2Vec: trained here, used two different ways downstream.

Two consumers, and it is worth being clear about the difference:

  * the dense network uses the vectors to *initialise* its embedding
    matrix, then keeps training them (or freezes them). Word2Vec is a
    starting point there, not the representation.
  * the sklearn models use the vectors to build one fixed document
    vector per description, by pooling. There the representation is
    all Word2Vec is -- the estimator never sees a token.

Two backends. gensim is the default and runs on the driver; the Spark
backend exists because on a large catalogue gensim on one core is the
slowest step in the whole pipeline, and because the older draft
notebook used it to work around a Unity Catalog restriction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from ..cleaning.tokens import tokenize
from ..config import Word2VecConfig


@dataclass
class WordVectors:
    """A plain {token: vector} map, detached from whichever backend made it.

    Deliberately not a gensim object: the artifact that ships with a
    model should not require gensim (or Spark) to be installed to load
    it, and should not break when either changes its pickle format.
    """

    vectors: Dict[str, np.ndarray]
    vector_size: int
    lowercase: bool = True
    token_pattern: str = r"\b\w\w+\b"

    def __len__(self) -> int:
        return len(self.vectors)

    def __contains__(self, token: str) -> bool:
        return token in self.vectors

    def document_vector(self, text: str, pooling: str = "mean") -> np.ndarray:
        """Pool the known tokens of one description into a single vector.

        Unknown tokens are skipped rather than mapped to zero: averaging
        a zero in would drag the document toward the origin in
        proportion to how much of it we failed to recognise, which is
        the opposite of what you want. A description with no known
        tokens returns zeros, and `document_matrix` counts those for
        you -- that count is worth watching.
        """
        tokens = [t for t in tokenize(text, self.lowercase, self.token_pattern)
                  if t in self.vectors]
        if not tokens:
            return np.zeros(self.vector_size, dtype=np.float32)
        stack = np.stack([self.vectors[t] for t in tokens])
        if pooling == "mean":
            return stack.mean(axis=0)
        if pooling == "sum":
            return stack.sum(axis=0)
        if pooling == "max":
            return stack.max(axis=0)
        raise ValueError(f"Unknown pooling: {pooling!r}")

    def document_matrix(
        self,
        texts: Sequence[str],
        pooling: str = "mean",
        normalize: bool = False,
        verbose: bool = True,
    ) -> np.ndarray:
        """One row per description. Reports how many came out empty."""
        out = np.zeros((len(texts), self.vector_size), dtype=np.float32)
        empty = 0
        for i, text in enumerate(texts):
            vec = self.document_vector(text, pooling)
            if not vec.any():
                empty += 1
            out[i] = vec
        if normalize:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            np.divide(out, norms, out=out, where=norms > 0)
        if verbose:
            print(
                f"[w2v] pooled {len(texts):,} documents to {out.shape[1]} dims"
                + (f"; {empty:,} had no known tokens ({empty / max(len(texts), 1):.2%})"
                   if empty else "")
            )
        return out

    def embedding_matrix(
        self,
        itos: Sequence[str],
        pad_index: int = 0,
        seed: int = 42,
        verbose: bool = True,
    ) -> np.ndarray:
        """An initial embedding matrix aligned to a TokenVocabulary.

        Tokens Word2Vec never saw get a small random vector rather than
        zeros. Zero rows are a local minimum the gradient struggles to
        escape, so they stay near-zero and the model quietly learns to
        ignore whichever words happened to be rare in the Word2Vec run.
        """
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(self.vector_size)
        matrix = rng.normal(0.0, scale, size=(len(itos), self.vector_size)).astype(np.float32)
        hits = 0
        for i, token in enumerate(itos):
            if token in self.vectors:
                matrix[i] = self.vectors[token]
                hits += 1
        matrix[pad_index] = 0.0
        if verbose:
            print(
                f"[w2v] initialised {hits:,}/{len(itos):,} embedding rows "
                f"({hits / max(len(itos), 1):.1%} hit rate)"
            )
        return matrix


def train_word2vec(
    texts: Sequence[str],
    cfg: Optional[Word2VecConfig] = None,
    spark=None,
    verbose: bool = True,
) -> WordVectors:
    """Train on the *training* texts and return a backend-free WordVectors."""
    cfg = cfg or Word2VecConfig()
    if cfg.backend == "gensim":
        return _train_gensim(texts, cfg, verbose)
    if cfg.backend == "spark":
        return _train_spark(texts, cfg, spark, verbose)
    raise ValueError(f"Unknown word2vec backend: {cfg.backend!r}")


def _train_gensim(texts: Sequence[str], cfg: Word2VecConfig, verbose: bool) -> WordVectors:
    from ..deps import auto_install

    auto_install("gensim", purpose="the gensim Word2Vec backend")

    from gensim.models import Word2Vec

    sentences = [tokenize(t) for t in texts]
    model = Word2Vec(
        sentences=sentences,
        vector_size=cfg.vector_size,
        window=cfg.window,
        min_count=cfg.min_count,
        epochs=cfg.epochs,
        sg=cfg.sg,
        workers=cfg.workers,
        seed=cfg.seed,
    )
    kv = model.wv
    vectors = {tok: np.asarray(kv[tok], dtype=np.float32) for tok in kv.index_to_key}
    if verbose:
        print(f"[w2v] gensim trained {len(vectors):,} vectors of size {cfg.vector_size}")
    return WordVectors(vectors=vectors, vector_size=cfg.vector_size)


def _train_spark(
    texts: Sequence[str], cfg: Word2VecConfig, spark, verbose: bool
) -> WordVectors:
    if spark is None:
        raise ValueError("The spark word2vec backend needs a session; pass spark=spark.")
    from pyspark.ml.feature import Word2Vec as SparkWord2Vec
    from pyspark.sql import Row

    rows = [Row(tokens=tokenize(t)) for t in texts]
    sdf = spark.createDataFrame(rows).repartition(cfg.num_partitions)
    w2v = SparkWord2Vec(
        vectorSize=cfg.vector_size,
        windowSize=cfg.window,
        minCount=cfg.min_count,
        maxIter=cfg.epochs,
        numPartitions=cfg.num_partitions,
        stepSize=cfg.step_size,
        seed=cfg.seed,
        inputCol="tokens",
        outputCol="vec",
    )
    model = w2v.fit(sdf)
    pdf = model.getVectors().toPandas()
    vectors = {
        str(r.word): np.asarray(r.vector.toArray(), dtype=np.float32)
        for r in pdf.itertuples()
    }
    if verbose:
        print(f"[w2v] spark trained {len(vectors):,} vectors of size {cfg.vector_size}")
    return WordVectors(vectors=vectors, vector_size=cfg.vector_size)


def load_or_train_word2vec(
    texts: Sequence[str],
    cfg: Optional[Word2VecConfig] = None,
    path=None,
    spark=None,
    retrain: bool = False,
    verbose: bool = True,
) -> WordVectors:
    """Reuse the vectors notebook 02 saved, or train and save them.

    Worth the indirection for two reasons. Training Word2Vec is the
    slowest step in the whole pipeline on a large catalogue, and it is
    wasteful to repeat it in notebooks 04 and 05. More importantly,
    repeating it means the dense network and the sklearn models end up
    sitting on *different* vectors -- Word2Vec is stochastic, and two
    runs on the same corpus do not agree. Any difference in their
    scores would then be partly an artefact of that, and there would
    be nothing in the output to say so.
    """
    import joblib

    from .. import paths as _paths

    cfg = cfg or Word2VecConfig()
    path = _paths.output_dir() / "word_vectors.joblib" if path is None else path

    if not retrain:
        try:
            wv = joblib.load(path)
            if verbose:
                print(f"[w2v] loaded {len(wv):,} vectors from {path}")
            if wv.vector_size != cfg.vector_size:
                print(
                    f"[w2v] WARNING: saved vectors are {wv.vector_size}-dim but "
                    f"the config asks for {cfg.vector_size}. Using the saved "
                    "vectors; pass retrain=True to rebuild."
                )
            return wv
        except (FileNotFoundError, OSError):
            if verbose:
                print(f"[w2v] no saved vectors at {path}; training")

    wv = train_word2vec(texts, cfg, spark=spark, verbose=verbose)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(wv, path, compress=3)
        if verbose:
            print(f"[w2v] saved to {path}")
    except OSError as exc:
        print(f"[w2v] could not save vectors: {exc}")
    return wv


def build_document_matrices(
    wv: WordVectors,
    train_texts: Sequence[str],
    *other_texts: Sequence[str],
    cfg: Optional[Word2VecConfig] = None,
    verbose: bool = True,
):
    """Pool every split with the same vectors. Same shape as sparse.fit_transform."""
    cfg = cfg or Word2VecConfig()
    X_train = wv.document_matrix(train_texts, cfg.pooling, cfg.normalize_docvecs, verbose)
    others = tuple(
        wv.document_matrix(t, cfg.pooling, cfg.normalize_docvecs, verbose)
        for t in other_texts
    )
    return X_train, others
