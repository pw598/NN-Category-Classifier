"""Scoring unseen descriptions from a saved bundle.

Section E of the draft notebooks, with the hierarchy attached. The
contract is narrow on purpose: give it a bundle directory and a frame
of raw descriptions, get back predictions with confidence, names, and
a review flag.

Two things this module refuses to do quietly. It will not clean text
with anything other than the cleaner saved in the bundle -- a scoring
path that cleans differently from training sees mostly out-of-
vocabulary words and degrades silently rather than failing. And it
will not hide an unreadable input: a description that cleans away to
nothing, or one where almost no token is recognised, comes back
flagged rather than guessed at.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .artifacts import attach_names, load_bundle
from .calibration import apply_cascade
from .cleaning.descriptions import clean_descriptions_with
from .hierarchy import predictions_frame


def _level_probs_from_bundle(
    bundle: Dict[str, Any],
    texts: Sequence[str],
    batch_size: int = 4096,
    device: Optional[str] = None,
) -> List[np.ndarray]:
    """Run whichever model the bundle holds and return per-level probabilities."""
    model = bundle["model"]
    if model is None:
        raise ValueError("Bundle contains no model.")

    calibrators = bundle.get("calibrators")

    # --- sklearn: pooled word vectors in, probabilities out ---------
    if hasattr(model, "predict_level_probs"):
        wv = bundle.get("word_vectors")
        if wv is None:
            raise ValueError("An sklearn bundle needs saved word_vectors to score.")
        X = wv.document_matrix(list(texts), verbose=False)
        probs = model.predict_level_probs(X)
        return calibrators.transform(probs) if calibrators else probs

    # --- torch: sparse matrix or token indices ----------------------
    from .deps import auto_install

    auto_install("torch", purpose="scoring from a neural-network bundle")

    import torch

    from .paths import torch_device
    from .torch_models import MultiHeadEmbedMLP

    device = device or torch_device()
    model.to(device).eval()

    from .torch_models import MultiHeadAttentionMLP, MultiHeadSparseInputMLP

    is_dense = isinstance(model, MultiHeadEmbedMLP)
    # The attention model takes the same (indices, offsets, values)
    # triple as the sparse-input one -- that was the point of giving it
    # the same signature, so csr_loaders needed no changes. It has to
    # be named here too, or it falls through to the dense branch below,
    # gets handed a `(batch, n_features)` matrix, and fails on unpacking
    # a tensor into three names.
    is_sparse_input = isinstance(
        model, (MultiHeadSparseInputMLP, MultiHeadAttentionMLP)
    )
    if is_dense:
        vocab = bundle.get("vocabulary")
        if vocab is None:
            raise ValueError("A dense-network bundle needs a saved vocabulary.")
        encoded = vocab.encode_all(texts)
    else:
        vec = bundle.get("vectorizer")
        if vec is None:
            raise ValueError("A sparse-network bundle needs a saved vectorizer.")
        X = vec.transform(list(texts)).tocsr()

    chunks: Optional[List[List[np.ndarray]]] = None
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            stop = min(start + batch_size, len(texts))
            if is_dense:
                seqs = encoded[start:stop]
                offsets = torch.tensor(
                    [0] + [len(s) for s in seqs[:-1]], dtype=torch.long
                ).cumsum(dim=0).to(device)
                flat = torch.tensor(
                    [t for s in seqs for t in s], dtype=torch.long
                ).to(device)
                logits = model((flat, offsets))
            elif is_sparse_input:
                # Never densified, the same as during training.
                block = X[start:stop]
                logits = model((
                    torch.from_numpy(block.indices.astype(np.int64)).to(device),
                    torch.from_numpy(block.indptr[:-1].astype(np.int64)).to(device),
                    torch.from_numpy(block.data.astype(np.float32)).to(device),
                ))
            else:
                dense = torch.from_numpy(
                    np.asarray(X[start:stop].toarray(), dtype=np.float32)
                ).to(device)
                logits = model(dense)

            if chunks is None:
                chunks = [[] for _ in logits]
            for i, lg in enumerate(logits):
                chunks[i].append(lg.detach().cpu().numpy())

    level_logits = [np.concatenate(c).astype(np.float32) for c in (chunks or [])]
    if calibrators:
        # Temperature scaling wants logits, so calibration happens here
        # rather than after a softmax that would have discarded them.
        return calibrators.transform(level_logits)

    shifted = [lg - lg.max(axis=1, keepdims=True) for lg in level_logits]
    return [
        (np.exp(s) / np.exp(s).sum(axis=1, keepdims=True)).astype(np.float32)
        for s in shifted
    ]


def level_probabilities(
    bundle: Dict[str, Any],
    texts: Sequence[str],
    batch_size: int = 4096,
    device: Optional[str] = None,
) -> List[np.ndarray]:
    """Per-level probabilities for already-cleaned text.

    The public form of the bundle-dispatch that `score` uses
    internally. Takes cleaned text and returns the calibrated
    per-level probability matrices, doing none of the cleaning,
    thresholding or naming that `score` layers on top.

    For ensembling two models, which is the reason it exists: both
    have to be asked the same question about the same rows, and the
    answer has to come back as plain arrays that can be averaged.

    The text must already be cleaned, and cleaned the *same way* for
    both bundles -- read it from `cleaned_data.txt` rather than
    cleaning it here, or the two models see different inputs and the
    average is meaningless.
    """
    return _level_probs_from_bundle(
        bundle, list(texts), batch_size=batch_size, device=device
    )


def check_compatible(bundles: Sequence[Dict[str, Any]]) -> None:
    """Refuse to ensemble models whose label encodings differ.

    Averaging `probs_a[level][:, j]` with `probs_b[level][:, j]`
    assumes column j means the same category in both. That holds when
    both hierarchies were built from the same training split, and
    silently does not when they were not -- producing an average of
    unrelated numbers, a plausible-looking accuracy, and no error.
    """
    reference = bundles[0]["hierarchy"]
    for i, other in enumerate(bundles[1:], start=1):
        h = other["hierarchy"]
        if list(h.level_columns) != list(reference.level_columns):
            raise ValueError(
                f"bundle {i} has level columns {h.level_columns}, "
                f"expected {reference.level_columns}"
            )
        for level in range(reference.n_levels):
            if not np.array_equal(reference.classes_[level], h.classes_[level]):
                raise ValueError(
                    f"bundle {i} has a different class ordering at level "
                    f"{reference.level_columns[level]!r} "
                    f"({h.n_classes(level)} vs {reference.n_classes(level)} classes). "
                    "Both models must be built from the same training split."
                )


def readability(texts: Sequence[str], bundle: Dict[str, Any]) -> pd.DataFrame:
    """How much of each description the model can actually read.

    Two numbers. `n_tokens` after cleaning, and the share of those the
    model has seen before. Both matter: a prediction from a one-token
    description is a guess dressed up as an answer, and so is one from
    a description made entirely of part numbers the vocabulary has
    never met.
    """
    from .cleaning.tokens import tokenize

    known: Optional[set] = None
    vocab = bundle.get("vocabulary")
    vec = bundle.get("vectorizer")
    wv = bundle.get("word_vectors")
    if vocab is not None:
        known = set(vocab.stoi)
    elif vec is not None and hasattr(vec, "vocabulary_"):
        known = set(vec.vocabulary_)
    elif wv is not None:
        known = set(wv.vectors)

    rows = []
    for text in texts:
        tokens = tokenize(text)
        n_known = sum(1 for t in tokens if known and t in known)
        rows.append({
            "n_tokens": len(tokens),
            "n_known_tokens": n_known,
            "known_share": n_known / len(tokens) if tokens else 0.0,
        })
    return pd.DataFrame(rows)


def _score_block(
    bundle_dir: str,
    df: pd.DataFrame,
    text_col: str = "FullDesc",
    id_col: Optional[str] = "ID",
    carry_cols: Optional[Sequence[str]] = None,
    vendor_col: Optional[str] = None,
    joint_mode: Optional[str] = "independent",
    predict_all_levels: bool = True,
    top_k: int = 1,
    min_known_tokens: int = 1,
    use_cascade: bool = True,
    thresholds: Optional[Dict[int, float]] = None,
    confidence_threshold: Optional[float] = None,
    batch_size: int = 4096,
    device: Optional[str] = None,
    bundle: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """One block of rows, start to finish. Use `score` instead.

    Split out so `score` can chunk. Nothing here is computed across
    rows -- the cleaner, the thresholds and the confidence cut are all
    fixed by the bundle -- so a block gives exactly the same answer it
    would have given inside a larger frame. `test_scoring_in_chunks_
    matches_scoring_whole` pins that.
    """
    bundle = bundle if bundle is not None else load_bundle(bundle_dir, device=device)
    hierarchy = bundle["hierarchy"]
    cleaner = bundle["cleaner"]
    if hierarchy is None:
        raise ValueError("Bundle contains no hierarchy; it cannot be scored against.")

    # A bundle trained with the vendor marker must be scored with it.
    # clean_descriptions_with only attaches the marker when vendor_col
    # is given, so omitting it here produces descriptions that are
    # missing a token the model was trained to expect -- on every row,
    # with no error, and a uniformly worse answer. Exactly the silent
    # degradation this module refuses to allow elsewhere.
    saved_cfg = bundle.get("config")
    wants_vendor = bool(getattr(getattr(saved_cfg, "cleaning", None),
                                "attach_vendor", False))
    # getattr with a default, not attribute access: configs pickled
    # before this field existed have no marker_columns on the instance.
    saved_markers = getattr(getattr(saved_cfg, "cleaning", None),
                            "marker_columns", None) or {}
    if wants_vendor and vendor_col is None:
        expected = getattr(getattr(saved_cfg, "data", None), "vendor_col", None)
        raise ValueError(
            "This bundle was trained with attach_vendor=True, so every "
            "description it saw carried a "
            f"{getattr(getattr(saved_cfg, 'cleaning', None), 'vendor_token_prefix', 'VND_')!r} "
            "marker. Scoring without one silently degrades every row.\n"
            f"  Pass vendor_col=... naming the column that holds the "
            f"supplier. The training config used {expected!r}.\n"
            "  If that was a derived column, recreate it on the frame "
            "first, e.g. df['_vendor'] = df['Global Vendno'].\n"
            "  To score without the marker anyway, pass "
            "vendor_col=False."
        )
    if vendor_col is False:
        vendor_col = None

    # Same contract as the vendor marker: a bundle trained with extra
    # code tokens must be scored with them, or every row arrives
    # missing features the model was fitted on.
    if saved_markers:
        absent = [c for c in saved_markers if c not in df.columns]
        if absent:
            raise KeyError(
                f"This bundle was trained with marker columns "
                f"{list(saved_markers)}, and {absent} are not in the frame.\n"
                f"  Columns: {list(df.columns)}\n"
                "  Join them in, or score with a bundle that does not use "
                "them."
            )

    # Checked before scoring, not after. A missing id_col used to be
    # filtered out silently, so the output came back with no way to
    # join it to anything -- discoverable only by noticing the column
    # was absent. Finding that out after a full forward pass over the
    # catalogue is worse again.
    if id_col is not None and id_col not in df.columns:
        raise KeyError(
            f"id_col={id_col!r} is not in the frame. Columns: "
            f"{list(df.columns)}. Pass the right name, or id_col=None "
            "to score without an identifier."
        )
    missing = [c for c in (carry_cols or []) if c not in df.columns]
    if missing:
        raise KeyError(
            f"carry_cols {missing} are not in the frame. Columns: "
            f"{list(df.columns)}"
        )

    work = df.copy().reset_index(drop=True)
    if cleaner is not None:
        work = clean_descriptions_with(
            work, cleaner, column=text_col, vendor_col=vendor_col,
            vendor_prefix=getattr(getattr(saved_cfg, "cleaning", None),
                                  "vendor_token_prefix", "VND_"),
            markers=saved_markers,
            drop_blank=False, verbose=False,
        )
    texts = work[text_col].fillna("").astype(str).tolist()

    reads = readability(texts, bundle)
    scorable = (reads["n_known_tokens"] >= min_known_tokens).to_numpy()
    if verbose:
        print(
            f"[predict] {int(scorable.sum()):,}/{len(work):,} rows have at least "
            f"{min_known_tokens} recognisable token(s)"
        )

    level_probs = _level_probs_from_bundle(
        bundle, texts, batch_size=batch_size, device=device
    )
    frame = predictions_frame(
        hierarchy,
        level_probs,
        mode=joint_mode,
        normalise=True,
        predict_all_levels=predict_all_levels,
        top_k=top_k,
    )

    # Per-level confidence for the class actually predicted, alongside
    # the joint `Confidence`. Worth having because the joint figure is
    # the worst-calibrated number in the frame: temperature scaling was
    # fitted per level, and multiplying four calibrated numbers
    # compounds their error. Measured on validation, joint ECE 0.052
    # against 0.025 for level 4 on its own.
    #
    # Computed here, before any cascade blanks the Predicted columns.
    per_level = None
    if predict_all_levels and joint_mode is not None:
        pred_cols = [f"Predicted {c}" for c in hierarchy.level_columns]
        if all(c in frame.columns for c in pred_cols):
            from .calibration import level_confidence

            codes = hierarchy.encode(frame[pred_cols].rename(
                columns={f"Predicted {c}": c for c in hierarchy.level_columns}))
            per_level = level_confidence(level_probs, codes)
            for i, col in enumerate(hierarchy.level_columns):
                frame[f"{col} Confidence"] = per_level[:, i]

    if use_cascade and joint_mode is not None and predict_all_levels:
        # Per-level thresholds when the bundle has them. The joint
        # prefix set saturates on this taxonomy -- depth 3 came back
        # unreachable on the vendor model, so that depth would never
        # assign -- and the per-level cascade reached 98.8% coverage
        # where the joint one reached 95.3%.
        per_level_thr = bundle.get("per_level_thresholds") or {}
        if thresholds is not None:
            per_level_thr = thresholds

        if per_level_thr and per_level is not None:
            from .calibration import apply_cascade_per_level

            frame = apply_cascade_per_level(
                frame, per_level, hierarchy.level_columns, per_level_thr)
        else:
            thr = bundle.get("thresholds") or {}
            if thr:
                if verbose:
                    print("[predict] no per-level thresholds in this bundle; "
                          "falling back to the joint prefix set, which "
                          "saturates. Re-save the bundle with "
                          "per_level_thresholds= to fix.")
                frame = apply_cascade(frame, hierarchy.level_columns, thr)

    if "Needs Review" not in frame.columns:
        cut = (
            confidence_threshold
            if confidence_threshold is not None
            else (bundle["manifest"].get("metrics", {}) or {}).get("default_threshold", 0.6)
        )
        frame["Needs Review"] = frame["Confidence"].to_numpy() < float(cut)

    # Unreadable rows are not predictions, whatever the model said.
    frame.loc[~scorable, "Needs Review"] = True
    frame["Review Reason"] = np.where(
        ~scorable, "too few recognisable tokens",
        np.where(frame["Needs Review"], "below confidence threshold", ""),
    )

    frame = pd.concat([reads, frame], axis=1)
    if bundle.get("hierarchy_lookup") is not None:
        name_columns = [
            c for c in bundle["hierarchy_lookup"].columns if c.endswith("Name")
        ]
        frame = attach_names(
            frame, bundle["hierarchy_lookup"], hierarchy.level_columns, name_columns
        )

    carry = ([id_col] if id_col else []) + [text_col] + list(carry_cols or [])
    carry = list(dict.fromkeys(c for c in carry if c in work.columns))
    out = pd.concat([work[carry].reset_index(drop=True), frame], axis=1)

    if verbose:
        n_review = int(out["Needs Review"].sum())
        print(
            f"[predict] {len(out) - n_review:,} auto-assigned, "
            f"{n_review:,} flagged for review ({n_review / max(len(out), 1):.1%})"
        )
    return out


def choose_chunk_size(
    bundle: Dict[str, Any],
    budget_bytes: int = 512 * 1024 ** 2,
    minimum: int = 1000,
) -> int:
    """Rows per block that keep the probability arrays inside a budget.

    The thing that actually runs out of memory is not the model and
    not the frame -- it is the per-level probability matrices.
    `_level_probs_from_bundle` concatenates one column per class at
    every level, so a four-level tree of 23/154/715/3,492 classes is
    4,384 floats, about 17.5 KB, for every single row. Calibration
    then makes a second copy.

    At 600,000 rows that is roughly 20 GB before `predictions_frame`
    is called, which is why scoring a full catalogue in one call dies
    while training on the same machine does not.
    """
    hierarchy = bundle.get("hierarchy")
    if hierarchy is None:
        return 50_000
    per_row = sum(hierarchy.n_classes(i) for i in range(hierarchy.n_levels))
    # x4 bytes, x2 for the calibrated copy, x1.5 headroom for the
    # frame that predictions_frame builds alongside them.
    bytes_per_row = per_row * 4 * 2 * 1.5
    return max(minimum, int(budget_bytes / max(bytes_per_row, 1.0)))


def score(
    bundle_dir: str,
    df: pd.DataFrame,
    chunk_size: Optional[int] = None,
    budget_bytes: int = 512 * 1024 ** 2,
    **kwargs: Any,
) -> pd.DataFrame:
    """Clean, score, calibrate, threshold and name -- in one call.

    Chunked by default. `chunk_size=None` picks a block size from the
    class counts and `budget_bytes`; pass an integer to override, or
    `chunk_size=0` to score everything in one pass (the old
    behaviour, and what runs out of memory on a large file).

    `bundle` can be passed in already-loaded, and for anything large
    it should be: reloading the weights per chunk dominates
    everything else.

    Emits a `<level> Confidence` column per level alongside the joint
    `Confidence`. Prefer the per-level one for thresholding -- see
    `_score_block`.

    Row-preserving. Every input row comes back in its original order,
    including the ones that could not be read; those carry
    `Needs Review = True` and a reason rather than being dropped,
    because a row that silently vanishes between input and output is
    the hardest kind of bug to notice downstream.
    """
    bundle = kwargs.get("bundle")
    if bundle is None:
        bundle = load_bundle(bundle_dir, device=kwargs.get("device"))
        kwargs["bundle"] = bundle

    if chunk_size is None:
        chunk_size = choose_chunk_size(bundle, budget_bytes)

    verbose = kwargs.get("verbose", True)
    if not chunk_size or len(df) <= chunk_size:
        return _score_block(bundle_dir, df, **kwargs)

    n_blocks = (len(df) + chunk_size - 1) // chunk_size
    if verbose:
        print(f"[predict] {len(df):,} rows in {n_blocks} blocks of "
              f"{chunk_size:,}")

    quiet = dict(kwargs)
    quiet["verbose"] = False
    out = []
    for i, start in enumerate(range(0, len(df), chunk_size), start=1):
        block = df.iloc[start:start + chunk_size]
        out.append(_score_block(bundle_dir, block, **quiet))
        if verbose:
            print(f"[predict]   block {i}/{n_blocks} done "
                  f"({start + len(block):,} rows)")

    scored = pd.concat(out, ignore_index=True)
    del out
    if verbose:
        n_review = int(scored["Needs Review"].sum())
        print(f"[predict] {len(scored) - n_review:,} auto-assigned, "
              f"{n_review:,} flagged for review "
              f"({n_review / max(len(scored), 1):.1%})")
    return scored


def score_to_csv(
    bundle_dir: str,
    df: pd.DataFrame,
    path: str,
    chunk_size: Optional[int] = None,
    budget_bytes: int = 512 * 1024 ** 2,
    **kwargs: Any,
) -> str:
    """Score and stream straight to disk, holding one block at a time.

    For when even the output frame will not fit. `score` still
    concatenates its blocks at the end; this never does, so peak
    memory is one block regardless of how large the input is.

    Returns the path. Read it back with `pd.read_csv(path, dtype="str")`
    -- as everywhere else in this repo, string dtypes on purpose,
    because category IDs like '00123' lose their leading zeros the
    moment anything guesses numeric.
    """
    bundle = kwargs.get("bundle")
    if bundle is None:
        bundle = load_bundle(bundle_dir, device=kwargs.get("device"))
        kwargs["bundle"] = bundle
    if chunk_size is None:
        chunk_size = choose_chunk_size(bundle, budget_bytes)
    chunk_size = chunk_size or len(df)

    verbose = kwargs.pop("verbose", True)
    n_blocks = (len(df) + chunk_size - 1) // chunk_size
    if verbose:
        print(f"[predict] {len(df):,} rows -> {path} in {n_blocks} block(s)")

    written = reviewed = 0
    for i, start in enumerate(range(0, len(df), chunk_size), start=1):
        block = _score_block(bundle_dir, df.iloc[start:start + chunk_size],
                             verbose=False, **kwargs)
        block.to_csv(path, index=False, mode="w" if i == 1 else "a",
                     header=(i == 1))
        written += len(block)
        reviewed += int(block["Needs Review"].sum())
        if verbose:
            print(f"[predict]   block {i}/{n_blocks}, {written:,} rows written")
        del block

    if verbose:
        print(f"[predict] {written - reviewed:,} auto-assigned, "
              f"{reviewed:,} flagged for review")
    return path


def score_summary(scored: pd.DataFrame, level_columns: Sequence[str]) -> pd.DataFrame:
    """Volume and mean confidence by assigned depth."""
    if "Assigned Depth" in scored.columns:
        grouped = scored.groupby("Assigned Depth")
        rows = []
        for depth, chunk in grouped:
            label = (
                "no assignment" if depth == 0
                else level_columns[int(depth) - 1]
            )
            rows.append({
                "assigned_depth": int(depth),
                "level": label,
                "rows": len(chunk),
                "share": len(chunk) / len(scored),
                "mean_confidence": float(chunk["Confidence"].mean()),
            })
        return pd.DataFrame(rows).sort_values("assigned_depth")

    return pd.DataFrame([{
        "rows": len(scored),
        "auto_assigned": int((~scored["Needs Review"]).sum()),
        "needs_review": int(scored["Needs Review"].sum()),
        "mean_confidence": float(scored["Confidence"].mean()),
    }])
