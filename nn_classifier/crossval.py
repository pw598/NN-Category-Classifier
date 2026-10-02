"""K-fold out-of-fold predictions from the network, resumable.

`sklearn_models.out_of_fold_probs` says cross-validation "is affordable
here and is not for the networks". That is still true of the cost. What
has changed is the reason to pay it: a second-stage model that combines
this network with the Naive Bayes classifier has to be fitted on
predictions each model made for products it had not seen, and a single
held-out set gives it only a slice of the catalogue -- thinnest exactly
where the two models disagree. Folding gives it every labelled row.

So this module trains the network `n_folds` times. Each fold repeats
the recipe of the training notebook (13b) on the rows outside the fold
-- validation split, token pruning, vectorizer, label tree, network,
early stopping, temperature calibration -- and then scores the fold's
own rows the way `predict.score` would score new products.

Three things it is careful about.

**The folds are reproducible by anyone.** `id_hash_folds` decides a
row's fold from its identifier alone, so the Naive Bayes repo assigns
the same products to the same folds without any file passing between
the two, and so does this module when it is restarted tomorrow.

**It can be stopped.** On CPU a fold takes hours. Each fold's training
checkpoints after every epoch (`training.fit(checkpoint_dir=...)`), a
finished fold writes its predictions and a marker, and calling
`out_of_fold_predictions` again skips the finished folds and resumes
the interrupted one from its last completed epoch. `time_budget_hours`
makes it stop by itself.

**It will not mix two runs.** The settings and the data are
fingerprinted into `run.json` when a run starts. Coming back with a
different config, different rows or a different fold assignment is
refused rather than silently producing a file whose folds were trained
under different settings.

    out_of_fold_predictions()   run, or continue, the folds
    status()                    where a run has got to
    collect()                   the finished predictions, one row per product
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

RUN_FILE = "run.json"
FOLD_FILE = "fold.json"
OOF_FILE = "oof.csv"

# Config fields that do not change what a fold produces: where the file
# lives, how batches are fetched, free text. A run may be resumed with
# these changed -- a different cluster, a different worker count.
_IGNORED_CFG_FIELDS = {
    ("data", "path"),
    ("nn", "num_workers"),
    ("nn", "prefer_gpu"),
    ("hierarchy", "batch_size"),
    ("notes",),
}


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------

def id_hash_folds(ids, n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Fold id per row, decided by the row's identifier and nothing else.

    A fold assignment that depends on which rows are present, or on
    their order, cannot be reproduced by a project that filters the
    catalogue differently -- and this one and the Naive Bayes repo do
    filter differently. Hashing the identifier removes the dependence:
    a product lands in the same fold in every project that uses the
    same `n_folds` and `seed`, and lands there again when a stopped
    run is restarted.

    The Naive Bayes repo carries an identical copy of this function
    (`product_classifier.crossval.id_hash_folds`). The two must stay
    identical: change the hash here and the folds stop matching,
    silently.

    Not stratified -- it cannot be without looking at the other rows.
    A held-out row can therefore occasionally find its category absent
    from training. That is rare (the category needs all its other rows
    in the same fold) and it is honest: such a row is scored wrong, as
    a new product in an unseen category would be.

    sha1 rather than `hash()`: Python salts `hash()` per process, so
    it would give different folds on every run.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")
    keys = pd.Series(np.asarray(ids, dtype=object)).astype(str).str.strip()
    blank = keys.isin(("", "nan", "None", "<NA>"))
    if blank.any():
        raise ValueError(
            f"{int(blank.sum()):,} row(s) have a blank identifier, so they "
            "cannot be given a reproducible fold. Fill or drop them first."
        )

    def _fold(key: str) -> int:
        digest = hashlib.sha1(f"{seed}|{key}".encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % n_folds

    mapping = {key: _fold(key) for key in keys.unique()}
    return keys.map(mapping).to_numpy(dtype=int)


def inner_split(rest: pd.DataFrame, level_columns: Sequence[str], split_cfg):
    """The train/validation split inside a fold.

    `data.split_data`, with one accommodation. Stratifying needs two
    rows per class, and `FilterConfig.min_class_count` guarantees that
    for the whole catalogue -- but not for the four fifths of it left
    once a fold is held out. A class down to its last row here goes to
    training whole, rather than failing the split or being dropped:
    dropping it would take a category away from this fold's model that
    the deployed model has.
    """
    from . import data

    levels = list(level_columns)
    if not split_cfg.stratify:
        return data.split_data(rest, levels, split_cfg, verbose=False)

    sizes = rest[levels[-1]].map(rest[levels[-1]].value_counts())
    lone = (sizes < 2).to_numpy()
    if not lone.any():
        return data.split_data(rest, levels, split_cfg, verbose=False)

    train_df, val_df = data.split_data(rest[~lone], levels, split_cfg, verbose=False)
    train_df = pd.concat([train_df, rest[lone]], ignore_index=True)
    return train_df, val_df


# ---------------------------------------------------------------------------
# The recipe, as 13b has it
# ---------------------------------------------------------------------------

def filtered_frame(cfg, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Null, rare-class and rare-path filters -- `data.prepare` without the split.

    Applied to the whole catalogue once, before folding, exactly as the
    training notebook applies them before its own split. A no-op on a
    file notebook 13a has already filtered.
    """
    from . import data
    from .cleaning.tokens import drop_rare_classes, drop_rare_paths

    levels = list(cfg.data.level_columns)
    out = data.drop_null_rows(df, cfg.data.text_col, levels,
                              cfg.data.drop_null_text, cfg.data.drop_null_labels,
                              verbose=verbose)
    out = drop_rare_classes(out, levels[-1], cfg.filters.min_class_count,
                            verbose=verbose)
    out = drop_rare_paths(out, levels, cfg.filters.min_path_count, verbose=verbose)
    return out.reset_index(drop=True)


def sparse_input_model(input_dim: int, n_classes_per_level: Sequence[int], cfg):
    """The network 13b builds: sparse first layer, shared trunk, one head per level."""
    from . import torch_models

    return torch_models.MultiHeadSparseInputMLP(
        input_dim=input_dim,
        n_classes_per_level=list(n_classes_per_level),
        hidden_dims=cfg.nn.hidden_dims,
        dropout=cfg.nn.dropout,
        sparse_grad=cfg.nn.sparse_grad,
    )


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------

def _write_json(path: Path, record: Dict[str, Any]) -> None:
    partial = path.with_name(path.name + ".tmp")
    partial.write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    os.replace(partial, path)


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fold_dir(run_dir: Path, k: int) -> Path:
    return run_dir / f"fold_{k}"


def _cfg_record(cfg) -> Dict[str, Any]:
    """The config as JSON, minus the fields that do not affect a result."""
    record = json.loads(json.dumps(cfg.to_dict(), default=str))
    for path in _IGNORED_CFG_FIELDS:
        node = record
        for key in path[:-1]:
            node = node.get(key, {})
        node.pop(path[-1], None)
    return record


def _diff(a: Any, b: Any, prefix: str = "") -> List[str]:
    """Dotted paths at which two nested JSON records differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        out: List[str] = []
        for key in sorted(set(a) | set(b)):
            out += _diff(a.get(key), b.get(key), f"{prefix}{key}.")
        return out
    return [] if a == b else [f"{prefix[:-1]}: was {a!r}, now {b!r}"]


def _data_fingerprint(df: pd.DataFrame, key_col: str, text_col: str,
                      label_col: str) -> str:
    """A hash of every row's identifier, description and deepest label.

    Order-independent, so re-sorting the file does not look like a
    change -- the folds do not depend on order either. A re-pull or a
    re-clean does look like one, which is the point: the folds already
    trained saw the old rows.
    """
    row_hash = pd.util.hash_pandas_object(
        df[[key_col, text_col, label_col]].astype(str), index=False
    ).to_numpy(dtype=np.uint64)
    return f"{int(row_hash.sum(dtype=np.uint64)):016x}-{len(df)}"


def _blacklist_fingerprint(ids: Optional[set]) -> Optional[str]:
    if not ids:
        return None
    joined = "\n".join(sorted(str(i) for i in ids))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16] + f"-{len(ids)}"


# ---------------------------------------------------------------------------
# One fold
# ---------------------------------------------------------------------------

def _run_fold(
    k: int,
    cfg,
    df: pd.DataFrame,
    folds: np.ndarray,
    fold_dir: Path,
    train_blacklist: Optional[set],
    model_factory: Callable,
    deadline: Optional[float],
    checkpoint_every_seconds: float,
    checkpoint_max_file_mb: int,
    min_known_tokens: int,
    prune_held_out: bool,
    keep_checkpoints: bool,
    device: Optional[str],
    verbose: bool,
) -> Dict[str, Any]:
    """Train on everything outside fold k, score fold k. Returns the fold record.

    Follows 13b step for step. Everything before `training.fit` is
    deterministic given the config and the rows, so a resumed fold
    rebuilds it and arrives at the same matrices the checkpoint was
    trained on -- `fit` checks that through the model's parameter
    shapes before it loads anything.
    """
    from . import data, hierarchy, predict, torch_models, training, vectorization
    from .calibration import LevelCalibrators
    from .cleaning.tokens import drop_one_off_tokens, prune_to_vocabulary

    started = time.time()
    levels = list(cfg.data.level_columns)
    text_col, id_col = cfg.data.text_col, cfg.data.id_col

    held_out = df[folds == k].reset_index(drop=True)
    rest = df[folds != k].reset_index(drop=True)

    # --- split, blacklist, prune: 13b's settings cell ----------------
    train_df, val_df = inner_split(rest, levels, cfg.split)
    if train_blacklist:
        train_df = data.drop_ids(train_df, train_blacklist, id_col,
                                 label="training rows", verbose=verbose)

    train_df, (val_df,), keep_vocab = drop_one_off_tokens(
        train_df, [val_df],
        text_col=text_col,
        min_count=cfg.filters.min_token_count,
        min_classes=cfg.filters.min_token_classes,
        label_col=levels[-1],
        verbose=verbose,
    )

    # Built from the training rows only, and after the blacklist -- a
    # class whose only rows were excluded must not keep a head.
    H = hierarchy.LabelHierarchy.from_labels(train_df, levels)
    y_train, y_val = H.encode(train_df), H.encode(val_df)

    # A validation row whose class is not in the label tree cannot be
    # used: there is no head for the loss to be taken against, and no
    # way for it to be right. 13b cannot produce one without a
    # blacklist; here the lone-class rule above and a blacklist can.
    winnable = (y_val >= 0).all(axis=1)
    if not winnable.all():
        if verbose:
            print(f"[cv] fold {k}: {int((~winnable).sum()):,} validation rows "
                  "are in classes training does not have; set aside")
        val_df = val_df[winnable].reset_index(drop=True)
        y_val = y_val[winnable]

    # --- vectorize ----------------------------------------------------
    vec, X_train, (X_val,) = vectorization.fit_transform(
        cfg.sparse, train_df[text_col], val_df[text_col], verbose=verbose)
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)

    # --- train --------------------------------------------------------
    #
    # Run the sparse-gradient probe now, before seeding. It is cached
    # per process and builds a small layer the first time, which draws
    # from the random stream -- so the first fit in a process starts
    # from a different point in that stream than every later one.
    # Left alone, a fold's result would depend on whether a restart
    # happened to make it the first fit of its session.
    torch_models.sparse_grad_is_supported()
    training.set_seed(cfg.nn.seed)
    train_loader, val_loader = torch_models.csr_loaders(
        X_train, y_train, X_val, y_val,
        batch_size=cfg.nn.batch_size, num_workers=cfg.nn.num_workers)
    model = model_factory(
        X_train.shape[1], [H.n_classes(i) for i in range(H.n_levels)], cfg)

    history = training.fit(
        model, train_loader, val_loader, cfg.nn, level_columns=levels,
        device=device, verbose=verbose,
        checkpoint_dir=fold_dir / "checkpoint",
        deadline=deadline,
        checkpoint_every_seconds=checkpoint_every_seconds,
        checkpoint_max_file_mb=checkpoint_max_file_mb,
    )
    if not history.completed:
        return {"fold": k, "status": "paused",
                "epochs_done": len(history.train_loss)}

    # --- calibrate, on this fold's own validation rows ----------------
    val_logits, val_targets = training.collect_logits(model, val_loader, device=device)
    val_logits_np = [lg.numpy() for lg in val_logits]
    assert np.array_equal(val_targets.numpy(), y_val), "loader order drifted"

    calibrators = LevelCalibrators(method=cfg.calibration.method, level_columns=levels)
    calibrators.fit(val_logits_np, y_val, max_iter=cfg.calibration.max_iter,
                    verbose=verbose)
    del val_logits, val_logits_np, val_targets, train_loader, val_loader
    del X_train, X_val
    gc.collect()

    # --- score the held-out rows, the way production scores -----------
    #
    # Through predict.score, with the pieces a saved bundle would hold,
    # so the numbers come off the same code path a new product takes.
    # `cleaner` is None because the text is already cleaned and already
    # carries its markers; `config` is None so score() does not ask for
    # a vendor column to attach them a second time.
    bundle = {
        "model": model, "hierarchy": H, "vectorizer": vec,
        "calibrators": calibrators, "cleaner": None, "vocabulary": None,
        "word_vectors": None, "hierarchy_lookup": None, "config": None,
        "thresholds": {}, "per_level_thresholds": {},
        "manifest": {"metrics": {
            "default_threshold": cfg.calibration.default_threshold}},
    }
    to_score = held_out
    if prune_held_out and keep_vocab:
        to_score = held_out.copy()
        to_score[text_col] = prune_to_vocabulary(to_score[text_col], keep_vocab)

    scored = predict.score(
        "", to_score,
        text_col=text_col, id_col=id_col, carry_cols=levels,
        vendor_col=False,
        joint_mode=cfg.hierarchy.joint_mode,
        predict_all_levels=True,
        use_cascade=False,
        min_known_tokens=min_known_tokens,
        device=device,
        bundle=bundle,
        verbose=False,
    )
    assert len(scored) == len(held_out), "rows lost in scoring"

    scored = scored.drop(columns=[text_col])
    scored = scored.rename(columns={c: f"Actual {c}" for c in levels})
    scored.insert(1, "fold", k)

    accuracy = {
        c: float((scored[f"Predicted {c}"].astype(str)
                  == scored[f"Actual {c}"].astype(str)).mean())
        for c in levels
    }
    unseen = float((H.encode(held_out)[:, -1] < 0).mean())

    # --- write: predictions first, the marker last --------------------
    fold_dir.mkdir(parents=True, exist_ok=True)
    partial = fold_dir / (OOF_FILE + ".tmp")
    scored.to_csv(partial, index=False)
    os.replace(partial, fold_dir / OOF_FILE)
    history.to_frame(levels).to_csv(fold_dir / "history.csv", index=False)

    record = {
        "fold": k,
        "status": "done",
        "rows_scored": int(len(scored)),
        "rows_train": int(len(train_df)),
        "rows_validation": int(len(val_df)),
        "features": int(len(vec.vocabulary_)),
        "n_paths": int(H.n_paths),
        "epochs_run": int(len(history.train_loss)),
        "best_epoch": history.best_epoch,
        "stopped_early": bool(history.stopped_early),
        "training_seconds": float(sum(history.epoch_seconds)),
        "temperatures": [float(getattr(c, "temperature", float("nan")))
                         for c in calibrators.calibrators],
        "accuracy": accuracy,
        "held_out_in_unseen_class": unseen,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _write_json(fold_dir / FOLD_FILE, record)

    # Only now. Until fold.json exists the checkpoint is the only copy
    # of hours of work.
    if not keep_checkpoints:
        shutil.rmtree(fold_dir / "checkpoint", ignore_errors=True)

    if verbose:
        print(f"[cv] fold {k} done in {(time.time() - started) / 60:.1f} min this "
              f"session: {levels[-1]} accuracy {accuracy[levels[-1]]:.4f} on "
              f"{len(scored):,} held-out rows (best epoch {history.best_epoch})")

    del model, bundle, scored, vec, H
    gc.collect()
    return record


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def out_of_fold_predictions(
    cfg,
    df: pd.DataFrame,
    run_dir,
    n_folds: int = 5,
    fold_seed: int = 0,
    fold_key_col: Optional[str] = None,
    train_blacklist: Optional[set] = None,
    model_factory: Optional[Callable] = None,
    only_folds: Optional[Sequence[int]] = None,
    time_budget_hours: Optional[float] = None,
    checkpoint_every_minutes: float = 0.0,
    checkpoint_max_file_mb: int = 400,
    keep_checkpoints: bool = False,
    min_known_tokens: int = 2,
    prune_held_out: bool = False,
    restart: bool = False,
    device: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the folds that are not done yet, and return `status(run_dir)`.

    Call it again to continue. A finished fold is skipped; an
    interrupted one resumes from its last saved epoch; the rest follow.
    When every fold is done, `collect(run_dir)` returns the predictions.

      cfg, df          the config and the cleaned frame the training
                       notebook would use. `filtered_frame` is applied
                       to df first.
      run_dir          where the run lives. Must survive the cluster:
                       use somewhere under `paths.output_dir()`, not
                       `paths.scratch_dir()`.
      n_folds, fold_seed, fold_key_col
                       the fold assignment. Must match the Naive Bayes
                       project's. `fold_key_col` defaults to
                       `cfg.data.id_col`.
      train_blacklist  identifiers never trained on, though still
                       scored when their fold is held out -- 13b's
                       BLACKLIST_SCOPE="train". To remove rows
                       entirely, drop them from df before calling.
      model_factory    `f(input_dim, n_classes_per_level, cfg) -> model`.
                       Defaults to 13b's `MultiHeadSparseInputMLP`. Any
                       model taking `csr_loaders` batches will do.
      only_folds       run just these fold numbers, e.g. to spread the
                       folds over several clusters at once. They share
                       `run_dir` and each writes only to its own fold
                       directory.
      time_budget_hours
                       stop after this long. The epoch in progress is
                       finished and saved first, so expect it to run
                       over by up to one epoch.
      checkpoint_every_minutes
                       the least time between saves within a fold. 0
                       saves after every epoch.
      prune_held_out   False scores the held-out text as it is, which
                       is what `predict.score` does to a new product.
                       True first removes the tokens training pruned,
                       which is what 13b's validation set sees.
      restart          discard `run_dir` and begin again.
    """
    from . import paths

    run_dir = Path(run_dir)
    key_col = fold_key_col or cfg.data.id_col
    levels = list(cfg.data.level_columns)
    model_factory = model_factory or sparse_input_model
    session_start = time.time()
    deadline = (session_start + time_budget_hours * 3600.0
                if time_budget_hours else None)

    for col in {key_col, cfg.data.id_col}:
        if not col or col not in df.columns:
            raise KeyError(
                f"{col!r} is not in the frame. Columns: {list(df.columns)}. "
                "Out-of-fold predictions need an identifier to assign folds "
                "by and to be joined on."
            )

    df = filtered_frame(cfg, df, verbose=verbose)
    folds = id_hash_folds(df[key_col], n_folds, fold_seed)
    sizes = np.bincount(folds, minlength=n_folds)

    n_dupes = int(df[cfg.data.id_col].astype(str).str.strip().duplicated().sum())
    if n_dupes and verbose:
        print(f"[cv] {n_dupes:,} rows repeat an earlier {cfg.data.id_col!r}. "
              "They are scored like any other, but the output cannot be "
              "joined one-to-one on that column.")

    # --- is this the run that is already in run_dir? ------------------
    record = {
        "n_folds": int(n_folds),
        "fold_seed": int(fold_seed),
        "fold_key_col": key_col,
        "fold_method": "id_hash",
        "id_col": cfg.data.id_col,
        "level_columns": levels,
        "n_rows": int(len(df)),
        "fold_sizes": [int(s) for s in sizes],
        "data_fingerprint": _data_fingerprint(df, key_col, cfg.data.text_col,
                                              levels[-1]),
        "train_blacklist": _blacklist_fingerprint(train_blacklist),
        "model_factory": getattr(model_factory, "__name__", str(model_factory)),
        "prune_held_out": bool(prune_held_out),
        "min_known_tokens": int(min_known_tokens),
        "config": _cfg_record(cfg),
    }

    if restart and run_dir.exists():
        shutil.rmtree(run_dir)
    existing = _read_json(run_dir / RUN_FILE)
    if existing is not None:
        compare = {k: v for k, v in existing.items() if k != "started_at"}
        changes = _diff(compare, record)
        if changes:
            raise RuntimeError(
                f"{run_dir} holds a run started with different settings or "
                "data:\n    " + "\n    ".join(changes[:20])
                + ("\n    ..." if len(changes) > 20 else "")
                + "\n  Folds trained under one set and folds trained under "
                "another do not make one set of out-of-fold predictions.\n"
                "  Restore what changed, point run_dir somewhere new, or "
                "pass restart=True to discard the folds already there."
            )
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(run_dir / RUN_FILE,
                    {**record, "started_at": time.strftime("%Y-%m-%d %H:%M:%S")})

    if verbose:
        print(paths.describe())
        print(f"[cv] run: {run_dir}")
        print(f"[cv] {len(df):,} rows in {n_folds} folds by hash of {key_col!r} "
              f"(seed {fold_seed}): {sizes.min():,}-{sizes.max():,} rows each")
        if deadline:
            print(f"[cv] time budget {time_budget_hours:g} h; will stop at "
                  f"{time.strftime('%H:%M', time.localtime(deadline))}")

    wanted = list(range(n_folds)) if only_folds is None else [int(k) for k in only_folds]
    bad = [k for k in wanted if not 0 <= k < n_folds]
    if bad:
        raise ValueError(f"only_folds={bad} outside 0..{n_folds - 1}.")

    for k in wanted:
        fold_dir = _fold_dir(run_dir, k)
        done = _read_json(fold_dir / FOLD_FILE)
        if done and done.get("status") == "done":
            if verbose:
                print(f"[cv] fold {k}: already done, skipping")
            continue
        if deadline is not None and time.time() >= deadline:
            if verbose:
                print(f"[cv] out of time before fold {k}")
            break

        if verbose:
            print(f"\n[cv] ===== fold {k} of 0..{n_folds - 1}: training on "
                  f"{int((folds != k).sum()):,}, holding out "
                  f"{int((folds == k).sum()):,} =====")
        result = _run_fold(
            k, cfg, df, folds, fold_dir, train_blacklist, model_factory,
            deadline, checkpoint_every_minutes * 60.0, checkpoint_max_file_mb,
            min_known_tokens, prune_held_out, keep_checkpoints, device, verbose,
        )
        gc.collect()
        if result["status"] == "paused":
            if verbose:
                print(f"[cv] fold {k} paused after {result['epochs_done']} "
                      "epochs. Run this again to continue from there.")
            break

    table = status(run_dir)
    if verbose:
        left = int((table["status"] != "done").sum())
        print(f"\n[cv] {len(table) - left} of {len(table)} folds done"
              + ("" if left else " -- collect(run_dir) returns the predictions"))
    return table


def status(run_dir) -> pd.DataFrame:
    """One row per fold: done, part-way, or not started.

    Reads only the small JSON files, so it is safe to call from another
    notebook while a run is in progress.
    """
    run_dir = Path(run_dir)
    run = _read_json(run_dir / RUN_FILE)
    if run is None:
        raise FileNotFoundError(f"No run in {run_dir} (no {RUN_FILE}).")
    deepest = run["level_columns"][-1]

    rows = []
    for k in range(run["n_folds"]):
        fold_dir = _fold_dir(run_dir, k)
        row: Dict[str, Any] = {"fold": k, "rows": run["fold_sizes"][k]}
        done = _read_json(fold_dir / FOLD_FILE)
        ckpt = _read_json(fold_dir / "checkpoint" / "state.json")
        if done and done.get("status") == "done":
            row.update({
                "status": "done",
                "epochs": done["epochs_run"],
                "best_epoch": done["best_epoch"],
                "training_hours": done["training_seconds"] / 3600.0,
                f"{deepest} accuracy": done["accuracy"][deepest],
                "finished_at": done["finished_at"],
            })
        elif ckpt is not None:
            history = ckpt.get("history", {})
            row.update({
                "status": ("trained, not yet scored" if ckpt.get("finished")
                           else f"in progress: epoch {ckpt['epoch']} saved"),
                "epochs": ckpt["epoch"],
                "best_epoch": history.get("best_epoch"),
                "training_hours": sum(history.get("epoch_seconds", [])) / 3600.0,
            })
        else:
            row["status"] = "not started"
        rows.append(row)
    return pd.DataFrame(rows)


def collect(run_dir, allow_partial: bool = False) -> pd.DataFrame:
    """The out-of-fold predictions, one row per product.

    Refuses an unfinished run unless `allow_partial=True`: a file
    holding four folds of five looks complete and is missing a fifth of
    the catalogue, chosen by hash rather than by anything visible.

    Columns, per level of the hierarchy:

        Actual <level>              the true label
        Predicted <level>           the assigned path
        <level> Confidence          that level's calibrated probability
                                    for the class predicted -- the one
                                    to threshold or to stack on
        <level> Prefix Probability  mass over paths sharing the prefix

    plus `fold`, the joint `Confidence` and `Path Evidence`, and the
    readability columns `predict.score` reports. Identifiers and labels
    come back as strings, as everywhere else in this repo.
    """
    run_dir = Path(run_dir)
    table = status(run_dir)
    run = _read_json(run_dir / RUN_FILE)
    unfinished = table.loc[table["status"] != "done", "fold"].tolist()
    if unfinished and not allow_partial:
        raise RuntimeError(
            f"Folds {unfinished} are not done, so these would not be "
            "out-of-fold predictions for the whole catalogue. Run "
            "out_of_fold_predictions() again to finish them, or pass "
            "allow_partial=True to look at what there is."
        )

    pieces = []
    for k in table.loc[table["status"] == "done", "fold"]:
        path = _fold_dir(run_dir, int(k)) / OOF_FILE
        header = pd.read_csv(path, nrows=0).columns
        as_text = {c: "str" for c in header
                   if c == run["id_col"] or c.startswith(("Actual ", "Predicted ", "Alt"))
                   and not c.endswith("Probability")}
        as_text["Review Reason"] = "str"
        pieces.append(pd.read_csv(path, dtype=as_text, keep_default_na=False,
                                  na_values={c: [""] for c in header
                                             if c not in as_text}))
    if not pieces:
        raise RuntimeError(f"No finished folds in {run_dir}.")
    return pd.concat(pieces, ignore_index=True)
