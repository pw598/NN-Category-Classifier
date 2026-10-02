"""The training loop, shared by both networks.

One loop, because the only thing that differs between the sparse and
dense networks is the shape of a batch, and `to_device` already
absorbs that. Two loops would be two places for the early-stopping
rule or the loss weighting to drift apart, and that drift would look
like a modelling result.

`fit` can also be stopped and picked up again. Give it a
`checkpoint_dir` and it saves its whole state at the end of an epoch
and, when called again with the same directory, carries on from the
epoch after -- same weights, same optimizer moments, same early-stopping
count, same random-number state. Without a `checkpoint_dir` nothing
about it has changed.
"""

from __future__ import annotations

import copy
import dataclasses
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .deps import auto_install

auto_install("torch", purpose="the neural networks")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from .config import NNConfig  # noqa: E402
from .paths import torch_device  # noqa: E402
from .torch_models import to_device  # noqa: E402


@dataclass
class History:
    """Per-epoch metrics, kept for the training curves and the run log."""

    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_acc: List[List[float]] = field(default_factory=list)
    val_acc: List[List[float]] = field(default_factory=list)
    lr: List[float] = field(default_factory=list)
    epoch_seconds: List[float] = field(default_factory=list)
    best_epoch: Optional[int] = None
    stopped_early: bool = False
    monitor: str = "accuracy"
    best_score: Optional[float] = None
    # What the *other* criterion would have chosen. Kept because the
    # gap between them is the thing worth seeing: it is how much
    # accuracy a loss-based restore would have discarded.
    best_epoch_by_loss: Optional[int] = None
    best_epoch_by_accuracy: Optional[int] = None
    # False when fit() returned because it ran out of time rather than
    # because training finished. The model then holds mid-training
    # weights -- not the best ones -- and must not be evaluated or
    # saved as though it were done. Call fit() again with the same
    # checkpoint_dir to continue.
    completed: bool = True
    # The epoch a resumed run picked up after, or None for a fresh one.
    resumed_from_epoch: Optional[int] = None

    def to_frame(self, level_columns: Optional[Sequence[str]] = None):
        import pandas as pd

        data = {
            "epoch": list(range(1, len(self.train_loss) + 1)),
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "lr": self.lr,
            "seconds": self.epoch_seconds,
        }
        n_levels = len(self.train_acc[0]) if self.train_acc else 0
        names = list(level_columns) if level_columns else [f"L{i+1}" for i in range(n_levels)]
        for i, name in enumerate(names[:n_levels]):
            data[f"train_acc[{name}]"] = [row[i] for row in self.train_acc]
            data[f"val_acc[{name}]"] = [row[i] for row in self.val_acc]
        return pd.DataFrame(data)


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def multi_head_loss(
    logits: Sequence[torch.Tensor],
    targets: torch.Tensor,
    criterion: nn.Module,
    weights: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """Weighted sum of the per-level cross-entropies.

    Equal weights by default. Raising the deepest level's weight is
    the usual first thing to try when L4 accuracy is the number that
    matters and the coarse heads are already saturated -- but do it
    knowing the coarse heads are part of what makes the trunk good,
    so pushing their weight to zero usually costs L4 accuracy rather
    than buying it.
    """
    n = len(logits)
    w = list(weights) if weights is not None else [1.0] * n
    if len(w) != n:
        raise ValueError(f"Got {len(w)} loss weights for {n} heads.")
    total = None
    for i, head_logits in enumerate(logits):
        part = criterion(head_logits, targets[:, i]) * w[i]
        total = part if total is None else total + part
    return total


@torch.no_grad()
def _accuracy(logits: Sequence[torch.Tensor], targets: torch.Tensor) -> List[int]:
    return [int((lg.argmax(dim=1) == targets[:, i]).sum().item())
            for i, lg in enumerate(logits)]


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: str,
    optimizer=None,
    loss_weights: Optional[Sequence[float]] = None,
):
    """One pass. Training when an optimizer is given, evaluation otherwise."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_rows = 0
    correct: Optional[np.ndarray] = None

    with torch.set_grad_enabled(training):
        for batch_x, batch_y in loader:
            batch_x = to_device(batch_x, device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = multi_head_loss(logits, batch_y, criterion, loss_weights)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            rows = batch_y.shape[0]
            total_loss += float(loss.item()) * rows
            total_rows += rows
            hits = np.asarray(_accuracy(logits, batch_y), dtype=np.int64)
            correct = hits if correct is None else correct + hits

    mean_loss = total_loss / max(total_rows, 1)
    acc = (correct / max(total_rows, 1)).tolist() if correct is not None else []
    return mean_loss, acc


class Optimizers:
    """AdamW for the dense parameters, SparseAdam for the sparse one.

    A pair rather than one object because the two cannot be merged:
    AdamW cannot consume a sparse gradient and SparseAdam cannot
    consume a dense one. Everything `run_epoch` asks of an optimizer
    -- `zero_grad` and `step` -- is forwarded to both.

    `param_groups` deliberately exposes only the dense optimizer's,
    because that is what `ReduceLROnPlateau` is attached to. Calling
    `sync_lr()` after each scheduler step copies the new rate across,
    so the two halves never drift apart.
    """

    def __init__(self, dense, sparse=None):
        self.dense = dense
        self.sparse = sparse

    @property
    def param_groups(self):
        return self.dense.param_groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.dense.zero_grad(set_to_none=set_to_none)
        if self.sparse is not None:
            self.sparse.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        self.dense.step()
        if self.sparse is not None:
            self.sparse.step()

    def sync_lr(self) -> None:
        """Copy the scheduled learning rate onto the sparse optimizer."""
        if self.sparse is None:
            return
        lr = self.dense.param_groups[0]["lr"]
        for group in self.sparse.param_groups:
            group["lr"] = lr


def build_optimizer(
    model: nn.Module, cfg: NNConfig, verbose: bool = True
) -> Optimizers:
    """Split the parameters by what kind of gradient they produce.

    A model built with `sparse_grad=True` reports its sparse
    parameters through `sparse_parameters()`; everything else stays on
    AdamW. A model that reports none -- which is every model by
    default -- gets a plain AdamW over all of it, so this is a no-op
    unless the flag is set.

    The mismatch worth catching is `cfg.sparse_grad=True` against a
    model constructed without it. Nothing would fail: it would train
    dense, at dense speed, and look like the optimisation simply did
    not help. So it warns.
    """
    sparse_params = []
    if hasattr(model, "sparse_parameters"):
        sparse_params = [p for p in model.sparse_parameters() if p.requires_grad]

    if getattr(cfg, "sparse_grad", False) and not sparse_params:
        print(
            "[fit] WARNING: cfg.sparse_grad is True but the model exposes no "
            "sparse parameters. Rebuild it with sparse_grad=True, or this "
            "trains dense at dense speed."
        )

    if sparse_params:
        from .torch_models import sparse_grad_is_supported

        ok, message = sparse_grad_is_supported()
        if not ok:
            raise RuntimeError(
                "sparse_grad=True, but this torch build will not produce a "
                f"sparse gradient for EmbeddingBag(sum) with per_sample_weights: "
                f"{message}. Rebuild the model with sparse_grad=False and set "
                "cfg.nn.sparse_grad = False; training is slower but identical "
                "in what it computes."
            )
        if verbose:
            print(f"[fit] {message}")

    sparse_ids = {id(p) for p in sparse_params}
    dense_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in sparse_ids
    ]

    dense = torch.optim.AdamW(
        dense_params, lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    sparse = (
        torch.optim.SparseAdam(sparse_params, lr=cfg.lr) if sparse_params else None
    )

    if verbose and sparse is not None:
        n_sparse = sum(p.numel() for p in sparse_params)
        n_dense = sum(p.numel() for p in dense_params)
        print(
            f"[fit] sparse gradients on: {n_sparse:,} parameters via SparseAdam "
            f"(no weight decay), {n_dense:,} via AdamW"
        )
    return Optimizers(dense, sparse)


def _model_signature(model: nn.Module) -> str:
    """A hash of the parameter names and shapes.

    What a checkpoint is checked against before it is loaded. The shapes
    depend on the vocabulary and the label tree, so a checkpoint from
    different data, a different fold or a different vectorizer setting
    has a different signature -- and loading it would either fail on a
    size mismatch or, worse, succeed on a coincidence.
    """
    import hashlib

    parts = [f"{name}:{tuple(t.shape)}" for name, t in model.state_dict().items()]
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]


# Settings that may change between sessions without making a resumed run
# a different experiment: how long to go on for, and how the batches are
# fetched. Everything else in NNConfig shapes the optimisation itself.
_RESUMABLE_CFG_FIELDS = frozenset({"epochs", "patience", "num_workers", "prefer_gpu"})


def _rng_state() -> dict:
    import random

    return {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _restore_rng_state(state: dict) -> None:
    import random

    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def fit(
    model: nn.Module,
    train_loader,
    val_loader,
    cfg: Optional[NNConfig] = None,
    level_columns: Optional[Sequence[str]] = None,
    device: Optional[str] = None,
    verbose: bool = True,
    checkpoint_dir=None,
    resume: bool = True,
    deadline: Optional[float] = None,
    checkpoint_every_seconds: float = 0.0,
    checkpoint_max_file_mb: int = 400,
) -> History:
    """Train, with early stopping and a best-weight restore.

    **Stopping and resuming.** With `checkpoint_dir` set, the full
    training state is saved at the end of an epoch (see
    `checkpoint.TrainingCheckpoint`). Calling `fit` again with the same
    directory -- same model, same loaders, a fresh process -- continues
    from the next epoch. The random-number state is part of what is
    saved, so a run that was stopped and resumed produces the same
    weights as one that was not -- wherever training is repeatable in
    the first place. With `sparse_grad=True` it is repeatable only on
    one thread (`torch.set_num_threads(1)`): two uninterrupted runs
    from the same seed already differ on several, by the order
    floating-point sums are taken in, and a resumed run differs from
    either by no more than they differ from each other.

      `deadline`  a `time.time()` value. Once it has passed, `fit`
                  finishes the epoch in progress, saves, and returns
                  with `history.completed = False`. The model is then
                  mid-training: do not evaluate it, call `fit` again.
      `resume`    False ignores and overwrites an existing checkpoint.
      `checkpoint_every_seconds`
                  the least time between saves. 0 saves every epoch,
                  so a kill costs at most the epoch in progress. Raise
                  it if the save is slow next to an epoch -- the time
                  each one takes is printed.

    A checkpoint is refused, loudly, if the model's parameter shapes do
    not match it: that means different data or a different vectorizer,
    and continuing would be training a different model from someone
    else's optimizer state.

    A hard kill (the cluster going away, an interrupt mid-epoch) loses
    only what came after the last save.

    `cfg.monitor` decides what "best" means, and on this problem the
    choice is worth real accuracy.

    The original version watched validation loss, on the reasoning
    that loss is the smoother signal and the one calibration depends
    on. That reasoning was wrong here. With 3,473 classes the two
    decouple: cross-entropy keeps getting worse on the examples the
    model is confidently wrong about, even while the argmax improves
    on others. Measured across four variants of notebook 04, loss
    plateaued around epoch 17-19 while L4 accuracy climbed for another
    six epochs, and restoring the loss-best weights discarded
    0.004-0.006 of accuracy every time.

    So `monitor="accuracy"` is the default, watching `monitor_level`
    (-1, the deepest). `monitor="val_loss"` restores the old
    behaviour.

    The LR scheduler still follows loss regardless. It is the
    smoother signal, and "stop improving" is a better trigger for
    slowing down than for stopping.

    Both criteria are recorded in the history either way, so the gap
    between them stays visible.
    """
    cfg = cfg or NNConfig()
    device = device or torch_device(cfg.prefer_gpu)
    set_seed(cfg.seed)
    model.to(device)

    if cfg.monitor not in ("val_loss", "accuracy"):
        raise ValueError(
            f"cfg.monitor={cfg.monitor!r}; expected 'val_loss' or 'accuracy'."
        )

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = build_optimizer(model, cfg, verbose=verbose)
    # Attached to the dense optimizer -- the scheduler type-checks its
    # argument, so it cannot take the pair. sync_lr() below carries
    # each change across to SparseAdam.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer.dense, mode="min", factor=cfg.scheduler_factor,
        patience=cfg.scheduler_patience,
    )

    history = History(monitor=cfg.monitor)
    best_score = float("inf") if cfg.monitor == "val_loss" else -float("inf")
    best_loss_seen = float("inf")
    best_acc_seen = -float("inf")
    best_state = None
    bad_epochs = 0

    names = list(level_columns) if level_columns else None

    # ---- resume ------------------------------------------------------
    ckpt = None
    start_epoch = 1
    finished = False
    saved_best_epoch = None          # the best_epoch the checkpoint holds
    signature = _model_signature(model)
    cfg_record = {k: (list(v) if isinstance(v, tuple) else v)
                  for k, v in dataclasses.asdict(cfg).items()}

    if checkpoint_dir is not None:
        from .checkpoint import TrainingCheckpoint

        ckpt = TrainingCheckpoint(checkpoint_dir, max_file_mb=checkpoint_max_file_mb)
        if ckpt.exists() and not resume:
            ckpt.clear()
        if ckpt.exists():
            state = ckpt.read_state()
            if state.get("model_signature") != signature:
                raise RuntimeError(
                    f"The checkpoint in {ckpt.directory} was written for a "
                    "model with different parameter shapes, so it comes from "
                    "different data, a different fold, or different "
                    "vectorizer or architecture settings. Resuming would "
                    "train this model from another one's optimizer state.\n"
                    "  Restore the settings that run used, or pass "
                    "resume=False (or delete the directory) to start again."
                )
            changed = sorted(
                k for k in set(cfg_record) | set(state.get("cfg", {}))
                if k not in _RESUMABLE_CFG_FIELDS
                and cfg_record.get(k) != state.get("cfg", {}).get(k)
            )
            if changed:
                raise RuntimeError(
                    f"The checkpoint in {ckpt.directory} was trained with "
                    f"different settings for {changed}:\n"
                    + "\n".join(
                        f"    {k}: checkpoint {state.get('cfg', {}).get(k)!r}, "
                        f"now {cfg_record.get(k)!r}" for k in changed)
                    + "\n  A run resumed under changed settings is neither "
                    "experiment. Restore them, or pass resume=False to start "
                    "again."
                )

            model.load_state_dict(ckpt.load_part("model", map_location=device))
            opt_state = ckpt.load_part("optim", map_location=device)
            optimizer.dense.load_state_dict(opt_state["dense"])
            if optimizer.sparse is not None and opt_state.get("sparse") is not None:
                optimizer.sparse.load_state_dict(opt_state["sparse"])
            scheduler.load_state_dict(opt_state["scheduler"])
            best_state = ckpt.load_part("best", map_location=device)

            history = History(**state["history"])
            best_score = state["best_score"]
            best_loss_seen = state["best_loss_seen"]
            best_acc_seen = state["best_acc_seen"]
            bad_epochs = state["bad_epochs"]
            finished = bool(state["finished"])
            start_epoch = int(state["epoch"]) + 1
            saved_best_epoch = history.best_epoch
            history.resumed_from_epoch = int(state["epoch"])
            history.completed = True

            # LengthBucketedBatches reshuffles from its own epoch counter.
            sampler = getattr(train_loader, "batch_sampler", None)
            if hasattr(sampler, "_epoch"):
                sampler._epoch = int(state["epoch"])
            # Last, so nothing above consumes from the restored stream.
            _restore_rng_state(opt_state["rng"])

            if verbose:
                what = ("training had already finished"
                        if finished else f"continuing at epoch {start_epoch}")
                print(f"[fit] resumed from {ckpt.directory}: epoch "
                      f"{state['epoch']} done, best so far epoch "
                      f"{history.best_epoch} -- {what}")

    last_save = time.time()

    def _save_checkpoint(epoch: int, is_finished: bool) -> None:
        nonlocal saved_best_epoch, last_save
        parts = {
            "model": model.state_dict(),
            "optim": {
                "dense": optimizer.dense.state_dict(),
                "sparse": (optimizer.sparse.state_dict()
                           if optimizer.sparse is not None else None),
                "scheduler": scheduler.state_dict(),
                "rng": _rng_state(),
            },
        }
        carry = []
        if best_state is not None:
            if history.best_epoch != saved_best_epoch:
                parts["best"] = best_state
            else:
                carry.append("best")
        info = ckpt.save(
            {
                "epoch": epoch,
                "finished": is_finished,
                "model_signature": signature,
                "cfg": cfg_record,
                "history": dataclasses.asdict(history),
                "best_score": best_score,
                "best_loss_seen": best_loss_seen,
                "best_acc_seen": best_acc_seen,
                "bad_epochs": bad_epochs,
            },
            parts, carry=carry,
        )
        saved_best_epoch = history.best_epoch
        last_save = time.time()
        if verbose:
            print(f"[fit] checkpoint saved after epoch {epoch} "
                  f"({info['bytes'] / 1e6:,.0f} MB in {info['seconds']:.1f}s)")

    va_acc: List[float] = history.val_acc[-1] if history.val_acc else []

    for epoch in range(start_epoch, (0 if finished else cfg.epochs) + 1):
        started = time.time()
        tr_loss, tr_acc = run_epoch(
            model, train_loader, criterion, device, optimizer, cfg.level_loss_weights
        )
        va_loss, va_acc = run_epoch(
            model, val_loader, criterion, device, None, cfg.level_loss_weights
        )
        scheduler.step(va_loss)
        optimizer.sync_lr()
        elapsed = time.time() - started

        history.train_loss.append(tr_loss)
        history.val_loss.append(va_loss)
        history.train_acc.append(tr_acc)
        history.val_acc.append(va_acc)
        history.lr.append(optimizer.param_groups[0]["lr"])
        history.epoch_seconds.append(elapsed)

        monitored_acc = va_acc[cfg.monitor_level] if va_acc else 0.0

        # Track both, whichever is being acted on.
        if va_loss < best_loss_seen - 1e-6:
            best_loss_seen = va_loss
            history.best_epoch_by_loss = epoch
        if monitored_acc > best_acc_seen + 1e-6:
            best_acc_seen = monitored_acc
            history.best_epoch_by_accuracy = epoch

        if cfg.monitor == "val_loss":
            improved = va_loss < best_score - 1e-6
            if improved:
                best_score = va_loss
        else:
            improved = monitored_acc > best_score + 1e-6
            if improved:
                best_score = monitored_acc

        if improved:
            best_state = copy.deepcopy(model.state_dict())
            history.best_epoch = epoch
            history.best_score = best_score
            bad_epochs = 0
        else:
            bad_epochs += 1

        if verbose:
            acc_str = "  ".join(
                f"{(names[i] if names else f'L{i+1}')}={a:.3f}"
                for i, a in enumerate(va_acc)
            )
            print(
                f"[fit] epoch {epoch:>3}  train_loss={tr_loss:.4f}  "
                f"val_loss={va_loss:.4f}  {acc_str}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}  {elapsed:.1f}s"
                + ("  *" if improved else "")
            )

        stop_early = bad_epochs >= cfg.patience
        done = stop_early or epoch >= cfg.epochs
        out_of_time = (deadline is not None and time.time() >= deadline
                       and not done)
        if stop_early:
            history.stopped_early = True

        if ckpt is not None and (
            done or out_of_time
            or time.time() - last_save >= checkpoint_every_seconds
        ):
            _save_checkpoint(epoch, done)

        if stop_early:
            if verbose:
                print(
                    f"[fit] stopping at epoch {epoch}; no improvement in "
                    f"{cfg.patience} epochs (best was epoch {history.best_epoch})"
                )
            break

        if out_of_time:
            history.completed = False
            if verbose:
                where = (f" Saved to {ckpt.directory}; call fit() again with "
                         "the same checkpoint_dir to continue."
                         if ckpt is not None else
                         " No checkpoint_dir was given, so this run cannot "
                         "be resumed.")
                print(f"[fit] out of time after epoch {epoch} of at most "
                      f"{cfg.epochs}.{where}")
            return history

    if best_state is not None:
        model.load_state_dict(best_state)
        if verbose:
            level_name = (names[cfg.monitor_level] if names
                          else f"L{len(va_acc) + cfg.monitor_level + 1}")
            what = "val_loss" if cfg.monitor == "val_loss" else f"{level_name} accuracy"
            print(f"[fit] restored weights from epoch {history.best_epoch} "
                  f"(best {what} = {history.best_score:.4f})")

            # The gap between the two criteria, made visible. When they
            # disagree, one of them is throwing something away.
            by_loss = history.best_epoch_by_loss
            by_acc = history.best_epoch_by_accuracy
            if by_loss != by_acc and by_loss and by_acc:
                acc_at_loss_best = history.val_acc[by_loss - 1][cfg.monitor_level]
                acc_at_acc_best = history.val_acc[by_acc - 1][cfg.monitor_level]
                print(
                    f"[fit] loss was best at epoch {by_loss} "
                    f"({level_name} {acc_at_loss_best:.4f}), accuracy at epoch "
                    f"{by_acc} ({level_name} {acc_at_acc_best:.4f}) -- "
                    f"a {abs(acc_at_acc_best - acc_at_loss_best):+.4f} spread"
                )
    return history


@torch.no_grad()
def collect_logits(
    model: nn.Module,
    loader,
    device: Optional[str] = None,
    return_targets: bool = True,
):
    """Every level's raw logits for a whole loader.

    Logits, not probabilities, and that matters: temperature scaling
    divides logits by a scalar before the softmax, so a function that
    returned probabilities would have thrown away exactly what the
    calibration step needs. The loader must not shuffle, or the
    targets will not line up with the rows.
    """
    device = device or torch_device()
    model.to(device)
    model.eval()

    chunks: Optional[List[List[torch.Tensor]]] = None
    targets: List[torch.Tensor] = []

    for batch_x, batch_y in loader:
        logits = model(to_device(batch_x, device))
        if chunks is None:
            chunks = [[] for _ in logits]
        for i, lg in enumerate(logits):
            chunks[i].append(lg.detach().cpu())
        if return_targets:
            targets.append(batch_y.detach().cpu())

    level_logits = [torch.cat(c) for c in (chunks or [])]
    y = torch.cat(targets) if (return_targets and targets) else None
    return (level_logits, y) if return_targets else level_logits


def softmax_probs(level_logits: Sequence[torch.Tensor]) -> List[np.ndarray]:
    """Logits to per-level probability matrices, as numpy for the hierarchy."""
    return [
        torch.softmax(lg.float(), dim=1).numpy().astype(np.float32)
        for lg in level_logits
    ]
