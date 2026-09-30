"""The training loop, shared by both networks.

One loop, because the only thing that differs between the sparse and
dense networks is the shape of a batch, and `to_device` already
absorbs that. Two loops would be two places for the early-stopping
rule or the loss weighting to drift apart, and that drift would look
like a modelling result.
"""

from __future__ import annotations

import copy
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


def fit(
    model: nn.Module,
    train_loader,
    val_loader,
    cfg: Optional[NNConfig] = None,
    level_columns: Optional[Sequence[str]] = None,
    device: Optional[str] = None,
    verbose: bool = True,
) -> History:
    """Train, with early stopping and a best-weight restore.

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

    for epoch in range(1, cfg.epochs + 1):
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

        if bad_epochs >= cfg.patience:
            history.stopped_early = True
            if verbose:
                print(
                    f"[fit] stopping at epoch {epoch}; no improvement in "
                    f"{cfg.patience} epochs (best was epoch {history.best_epoch})"
                )
            break

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
