"""The two networks, both multi-head.

One shared trunk, one softmax head per level. That choice is what makes
the hierarchy work without training four models: the trunk learns a
single representation of what the product *is*, and each head reads a
different granularity off it. Levels 1 through 3 are, in effect, free
supervision for level 4 -- the coarse heads are easy to get right, and
the gradient they send back shapes the trunk in a way that helps the
hard head too.

Only the input side differs between the two:

  MultiHeadSparseMLP   a BOW / TF-IDF row straight into a Linear
  MultiHeadEmbedMLP    token indices through an EmbeddingBag, then the
                       same trunk (fastText-style: mean of learned word
                       vectors, no sequence model)

Both produce a list of logit tensors, one per active level, in the same
order as `level_columns`. Everything downstream -- calibration,
hierarchy, prediction -- takes that list and does not care which
network produced it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .deps import auto_install

# torch ships with the Databricks ML runtimes and not the standard ones,
# so this is the first thing that runs. A no-op when torch is present.
#
# On a cluster with no GPU the default wheel is mostly CUDA you will
# never use; install the CPU-only build yourself beforehand and this
# will leave it alone:
#
#   nn_classifier.ensure_packages(
#       "torch", index_url="https://download.pytorch.org/whl/cpu")
auto_install("torch", purpose="the neural networks")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from scipy import sparse  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class SparseMatrixDataset(Dataset):
    """Rows of a CSR matrix, densified one batch at a time.

    The dataset hands back row *indices*, not rows. Densifying in
    `__getitem__` would build one dense vector per item and then stack
    them; densifying in the collate function slices the CSR once per
    batch, which is both faster and the difference between a 16,000-
    feature matrix fitting in memory and not.
    """

    def __init__(self, X: sparse.csr_matrix, y: np.ndarray):
        self.X = X.tocsr()
        self.y = torch.as_tensor(np.asarray(y), dtype=torch.long)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return idx, self.y[idx]


class SparseCollate:
    """Densify one batch of CSR rows.

    A class rather than a closure so it survives pickling. On Linux a
    DataLoader worker inherits it by fork and a closure would have
    worked; on spawn -- Windows, or a future torch default -- it would
    not, and the failure is an opaque pickling error a long way from
    its cause. `num_workers > 0` is worth having here because
    densifying a `(batch x n_features)` block is most of this
    network's CPU time.
    """

    def __init__(self, X: sparse.csr_matrix):
        self.X = X.tocsr()

    def __call__(self, batch):
        idx = [b[0] for b in batch]
        ys = torch.stack([b[1] for b in batch])
        # toarray() rather than todense(): the latter returns np.matrix
        # and, on a float64 CSR, forces a second full-size cast. Store
        # X as float32 and this is one allocation instead of two.
        dense = torch.from_numpy(np.asarray(self.X[idx].toarray(), dtype=np.float32))
        return dense, ys


def make_sparse_collate(X: sparse.csr_matrix) -> "SparseCollate":
    return SparseCollate(X)


class CSRTripleDataset(Dataset):
    """CSR rows kept sparse all the way into the model.

    `SparseMatrixDataset` densifies each batch to
    `(batch x n_features)`. With bigrams that is 512 x 40,000 x 8 bytes
    of float64 from `.todense()`, plus an 82 MB float32 cast, 1,029
    times per epoch -- 235 GB of allocation churn to carry about six
    non-zeros per row. That is what took notebook 03 from 132 seconds
    an epoch to over an hour.

    This hands the model `(indices, offsets, values)` instead and lets
    the first layer do a sparse gather. Per batch it moves 25 KB.
    """

    def __init__(self, X: sparse.csr_matrix, y: np.ndarray):
        self.X = X.tocsr()
        self.y = torch.as_tensor(np.asarray(y), dtype=torch.long)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return idx, self.y[idx]


class CSRTripleCollate:
    """Slice a CSR block and hand back its raw sparse structure."""

    def __init__(self, X: sparse.csr_matrix):
        self.X = X.tocsr()

    def __call__(self, batch):
        rows = [b[0] for b in batch]
        ys = torch.stack([b[1] for b in batch])
        block = self.X[rows]
        # indptr[:-1] is where each row starts, which is exactly what
        # EmbeddingBag wants as `offsets`. Empty rows repeat an offset,
        # which it handles by returning zeros for that bag.
        indices = torch.from_numpy(block.indices.astype(np.int64, copy=False))
        offsets = torch.from_numpy(block.indptr[:-1].astype(np.int64, copy=False))
        values = torch.from_numpy(block.data.astype(np.float32, copy=False))
        return (indices, offsets, values), ys


class SparseLinear(nn.Module):
    """`y = x @ W + b` evaluated only at the non-zeros of x.

    An `EmbeddingBag` in sum mode with `per_sample_weights` *is* a
    linear layer over a sparse input: it gathers the rows of W named by
    the non-zero feature indices, scales each by its value, and sums.
    The dense equivalent multiplies 40,000 numbers per row to use six
    of them.

    The bias is separate because EmbeddingBag has none.

    `sparse=True` changes the *backward* pass, not the forward one.
    This matters more than it sounds, and measuring it is what found
    it: across the 03a vocabulary sweep, epoch time tracked the
    feature count almost exactly -- 23,319 features took 108s and
    80,000 took 374s -- while the non-zeros per row stayed flat at
    5.3-5.7. The forward gather was never the cost.

    The cost is that a dense-gradient `EmbeddingBag` produces a full
    `(in_features, out_features)` gradient on every step, and AdamW
    then rewrites all of it plus two moment buffers -- 41M parameters
    times four tensors, 1,029 times an epoch, to touch six rows per
    example. `sparse=True` emits a sparse gradient covering only the
    rows actually used, which is what makes the cost scale with
    non-zeros the way the forward pass already does.

    The catch is that it needs an optimizer that understands sparse
    gradients. `training.fit` handles that by splitting the parameter
    groups; see its docstring. Nothing about the forward arithmetic
    changes, so a model trained either way scores identically and the
    saved weights are interchangeable.

    `feature_dropout` drops whole input features during training, and
    it exists because SparseAdam takes no weight decay -- so the
    52.4M numbers in this table, 98% of the model, currently train
    with no penalty at all. The dropout already in the trunk sits
    *after* this layer, on the 512-wide output, and does nothing to
    stop a rare feature memorising the handful of rows it appears in.

    Implemented by zeroing entries of `per_sample_weights` rather
    than by reindexing: a weight of zero contributes nothing to the
    sum and receives no gradient, which is exactly dropping the
    token, and it keeps the offsets valid. Survivors are scaled by
    1/(1-p) so the expected sum is unchanged and inference needs no
    correction.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sparse: bool = False,
        feature_dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.sparse = bool(sparse)
        self.feature_dropout = float(feature_dropout)
        if not 0.0 <= self.feature_dropout < 1.0:
            raise ValueError(
                f"feature_dropout={feature_dropout}; expected 0.0 <= p < 1.0."
            )
        self.weight = nn.EmbeddingBag(
            in_features, out_features, mode="sum", sparse=self.sparse
        )
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, indices, offsets, values):
        if self.training and self.feature_dropout > 0.0:
            keep = (torch.rand_like(values) >= self.feature_dropout).to(values.dtype)
            # A row can lose every token, leaving only the bias. That is
            # the same thing standard dropout does to a narrow layer and
            # is left alone deliberately: with ~7 tokens per row at
            # p=0.15 it happens about once in three million.
            values = values * keep / (1.0 - self.feature_dropout)
        gathered = self.weight(indices, offsets, per_sample_weights=values)
        return gathered + self.bias

    def sparse_parameters(self) -> List[nn.Parameter]:
        """The parameters whose gradients come back sparse, if any.

        Empty when `sparse=False`, so a caller can split parameter
        groups by calling this unconditionally.
        """
        return [self.weight.weight] if self.sparse else []


_SPARSE_GRAD_CHECK: Optional[Tuple[bool, str]] = None


def sparse_grad_is_supported() -> Tuple[bool, str]:
    """Does this torch build produce sparse gradients for our exact call?

    `EmbeddingBag(sparse=True)` and `per_sample_weights` are each
    documented, but their *combination* is the narrow path -- sum mode
    with per-sample weights and a sparse gradient -- and support for
    it has moved between versions and between CPU and CUDA. Rather
    than assume, this runs the real thing on three rows and looks at
    whether the gradient came back sparse.

    Cheap enough to run once and cached, because the answer cannot
    change inside a process.

    Returns `(ok, message)`. A False here is not a bug in the caller;
    it means this torch build wants the dense path.
    """
    global _SPARSE_GRAD_CHECK
    if _SPARSE_GRAD_CHECK is not None:
        return _SPARSE_GRAD_CHECK

    try:
        bag = nn.EmbeddingBag(10, 4, mode="sum", sparse=True)
        indices = torch.tensor([0, 3, 7, 2], dtype=torch.long)
        offsets = torch.tensor([0, 2, 3], dtype=torch.long)
        values = torch.tensor([1.0, 0.5, 2.0, 1.5], dtype=torch.float32)
        out = bag(indices, offsets, per_sample_weights=values)
        out.sum().backward()

        grad = bag.weight.grad
        if grad is None:
            result = (False, "backward produced no gradient at all")
        elif not grad.is_sparse:
            result = (
                False,
                "gradient came back dense despite sparse=True, so SparseAdam "
                "would reject it and nothing would be saved",
            )
        else:
            result = (True, f"sparse gradient confirmed ({torch.__version__})")
    except Exception as exc:                               # noqa: BLE001
        result = (False, f"{type(exc).__name__}: {exc}")

    _SPARSE_GRAD_CHECK = result
    return result


class TokenIndexDataset(Dataset):
    """Variable-length token-index lists, for EmbeddingBag."""

    def __init__(self, encoded: Sequence[Sequence[int]], y: np.ndarray):
        self.encoded = [list(e) for e in encoded]
        self.y = torch.as_tensor(np.asarray(y), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int):
        return self.encoded[idx], self.y[idx]


class OffsetsCollate:
    """Flat token tensor plus offsets -- no padding, so no wasted compute.

    EmbeddingBag takes the whole batch as one 1-D tensor with an index
    of where each document starts. Padding to the longest description
    in the batch would mean most of the work is done on pad tokens,
    and product descriptions vary enormously in length.

    Picklable, for the same reason as SparseCollate.
    """

    def __init__(self, unk_index: int = 1):
        self.unk_index = unk_index

    def __call__(self, batch):
        seqs = [b[0] if len(b[0]) else [self.unk_index] for b in batch]
        ys = torch.stack([b[1] for b in batch])
        offsets = torch.tensor(
            [0] + [len(s) for s in seqs[:-1]], dtype=torch.long
        ).cumsum(dim=0)
        flat = torch.tensor([t for s in seqs for t in s], dtype=torch.long)
        return (flat, offsets), ys


def make_offsets_collate(unk_index: int = 1) -> "OffsetsCollate":
    return OffsetsCollate(unk_index)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def _build_trunk(input_dim: int, hidden_dims: Sequence[int], dropout: float) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = input_dim
    for dim in hidden_dims:
        layers += [
            nn.Linear(prev, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        prev = dim
    return nn.Sequential(*layers)


class MultiHeadSparseMLP(nn.Module):
    """BOW / TF-IDF in, one logit tensor per level out."""

    def __init__(
        self,
        input_dim: int,
        n_classes_per_level: Sequence[int],
        hidden_dims: Sequence[int] = (512, 256),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.n_classes_per_level = list(n_classes_per_level)
        self.hidden_dims = list(hidden_dims)
        self.dropout = float(dropout)

        self.trunk = _build_trunk(input_dim, hidden_dims, dropout)
        out_dim = hidden_dims[-1] if hidden_dims else input_dim
        self.heads = nn.ModuleList([nn.Linear(out_dim, n) for n in n_classes_per_level])

    def forward(self, x) -> List[torch.Tensor]:
        z = self.trunk(x)
        return [head(z) for head in self.heads]

    def init_kwargs(self) -> Dict:
        return dict(
            input_dim=self.input_dim,
            n_classes_per_level=self.n_classes_per_level,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        )


class MultiHeadEmbedMLP(nn.Module):
    """Learned word vectors, mean-pooled per description, then the trunk."""

    def __init__(
        self,
        vocab_size: int,
        n_classes_per_level: Sequence[int],
        embed_dim: int = 150,
        hidden_dims: Sequence[int] = (512, 256),
        dropout: float = 0.2,
        padding_idx: int = 0,
        pooling_mode: str = "mean",
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.n_classes_per_level = list(n_classes_per_level)
        self.hidden_dims = list(hidden_dims)
        self.dropout = float(dropout)
        self.padding_idx = int(padding_idx)
        self.pooling_mode = str(pooling_mode)

        if self.pooling_mode not in ("mean", "sum", "max"):
            raise ValueError(
                f"pooling_mode={pooling_mode!r}; expected 'mean', 'sum' or 'max'."
            )

        # torch refuses padding_idx with max pooling -- there is no
        # sensible identity element for a maximum.
        bag_kwargs = {} if self.pooling_mode == "max" else {"padding_idx": padding_idx}
        self.embedding = nn.EmbeddingBag(
            vocab_size, embed_dim, mode=self.pooling_mode, **bag_kwargs
        )
        self.trunk = _build_trunk(embed_dim, hidden_dims, dropout)
        out_dim = hidden_dims[-1] if hidden_dims else embed_dim
        self.heads = nn.ModuleList([nn.Linear(out_dim, n) for n in n_classes_per_level])

    def forward(self, x) -> List[torch.Tensor]:
        flat, offsets = x
        z = self.trunk(self.embedding(flat, offsets))
        return [head(z) for head in self.heads]

    def load_pretrained_embeddings(
        self, matrix: np.ndarray, freeze: bool = False
    ) -> None:
        """Seed the embedding matrix from Word2Vec.

        Worth doing when the catalogue is large but the labelled part
        of it is small: Word2Vec learns from every description,
        labelled or not, so it brings in evidence the supervised loss
        never sees.

        `freeze=True` only makes sense when the Word2Vec corpus was
        much larger than the training set. Otherwise fine-tuning wins,
        because the task-specific signal is what separates two
        near-synonyms that matter to the taxonomy.
        """
        weights = torch.as_tensor(np.asarray(matrix, dtype=np.float32))
        if weights.shape != self.embedding.weight.shape:
            raise ValueError(
                f"Embedding matrix is {tuple(weights.shape)} but the layer is "
                f"{tuple(self.embedding.weight.shape)}."
            )
        with torch.no_grad():
            self.embedding.weight.copy_(weights)
            if self.pooling_mode != "max":
                # Under max pooling there is no padding_idx and a zeroed
                # row could win the maximum on negative dimensions.
                self.embedding.weight[self.padding_idx].zero_()
        self.embedding.weight.requires_grad_(not freeze)

    def init_kwargs(self) -> Dict:
        return dict(
            vocab_size=self.vocab_size,
            n_classes_per_level=self.n_classes_per_level,
            embed_dim=self.embed_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            padding_idx=self.padding_idx,
            pooling_mode=self.pooling_mode,
        )


class MultiHeadSparseInputMLP(nn.Module):
    """The sparse network, without ever densifying a batch.

    Identical to `MultiHeadSparseMLP` in what it computes -- the first
    layer is still an affine map of the TF-IDF vector -- but it
    evaluates that map over the non-zeros instead of materialising
    40,000 mostly-zero numbers per row. Everything after the first
    layer is the same dense trunk, because by then the representation
    genuinely is dense.

    Worth being clear that this is not an approximation. `SparseLinear`
    computes exactly `x @ W + b`; the weights are the same weights and
    the gradients are the same gradients. Only the arithmetic that
    multiplies by zero is skipped.
    """

    def __init__(
        self,
        input_dim: int,
        n_classes_per_level: Sequence[int],
        hidden_dims: Sequence[int] = (512, 256),
        dropout: float = 0.2,
        sparse_grad: bool = False,
        feature_dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.n_classes_per_level = list(n_classes_per_level)
        self.hidden_dims = list(hidden_dims)
        self.dropout = float(dropout)
        self.sparse_grad = bool(sparse_grad)
        self.feature_dropout = float(feature_dropout)

        if not hidden_dims:
            raise ValueError(
                "MultiHeadSparseInputMLP needs at least one hidden layer; "
                "the sparse gather is the first one."
            )

        first = hidden_dims[0]
        self.first = SparseLinear(input_dim, first, sparse=self.sparse_grad,
                                  feature_dropout=self.feature_dropout)
        self.first_norm = nn.BatchNorm1d(first)
        self.first_act = nn.Sequential(nn.ReLU(), nn.Dropout(dropout))

        # The rest of the trunk sees a dense `first`-wide vector.
        self.rest = _build_trunk(first, hidden_dims[1:], dropout)
        out_dim = hidden_dims[-1]
        self.heads = nn.ModuleList([nn.Linear(out_dim, n)
                                    for n in n_classes_per_level])

    def forward(self, x) -> List[torch.Tensor]:
        indices, offsets, values = x
        z = self.first(indices, offsets, values)
        z = self.first_act(self.first_norm(z))
        z = self.rest(z)
        return [head(z) for head in self.heads]

    def init_kwargs(self) -> Dict:
        return dict(
            input_dim=self.input_dim,
            n_classes_per_level=self.n_classes_per_level,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            sparse_grad=self.sparse_grad,
            feature_dropout=self.feature_dropout,
        )

    def sparse_parameters(self) -> List[nn.Parameter]:
        """Parameters needing a sparse-gradient optimizer. Empty unless enabled."""
        return self.first.sparse_parameters()


class MultiHeadAttentionMLP(nn.Module):
    """Self-attention over the token bag, instead of summing it.

    The motivation is a specific, measured failure. Level-4 top-5
    accuracy is 0.918 against an information ceiling of 0.949, so the
    model finds the right neighbourhood and loses on the final pick.
    Look at what those picks are:

        insert coroturn  -> Turning Inserts | Milling Inserts | Cutting Inserts
        insert coromill  -> Milling Inserts | Ball Nose End Mills | ...
        insert corocut   -> Parting & Grooving | Turning Inserts

    `insert` means nothing on its own; the neighbouring token decides.
    `SparseLinear` computes a plain sum, so `W[insert] + W[coroturn]`
    collapses to one point and any interaction has to be recovered by
    the ReLU layers downstream. The bias is additive: the model must
    *learn* that a combination is special rather than being built to
    notice combinations.

    Attention makes each token's vector depend on the others present,
    so `insert` beside `coroturn` is genuinely a different vector from
    `insert` beside `coromill` -- which is the thing a bag of features
    cannot express.

    Takes the same `(indices, offsets, values)` triple as
    `MultiHeadSparseInputMLP`, so `csr_loaders` needs no changes. The
    padding is rebuilt per batch inside `forward`.

    Cost, honestly: attention is O(L^2) in tokens per row, and the
    padding is to the *longest* row in the batch. At ~7 tokens
    average a single long row pads everything, so `max_tokens` caps
    it. Expect several times the sum model's epoch time on CPU.
    """

    def __init__(
        self,
        input_dim: int,
        n_classes_per_level: Sequence[int],
        embed_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 1,
        hidden_dims: Sequence[int] = (512, 256),
        dropout: float = 0.2,
        max_tokens: int = 32,
        sparse_grad: bool = False,
    ):
        super().__init__()
        if embed_dim % n_heads:
            raise ValueError(
                f"embed_dim={embed_dim} must divide by n_heads={n_heads}."
            )
        self.input_dim = int(input_dim)
        self.n_classes_per_level = list(n_classes_per_level)
        self.embed_dim = int(embed_dim)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.hidden_dims = list(hidden_dims)
        self.dropout = float(dropout)
        self.max_tokens = int(max_tokens)
        self.sparse_grad = bool(sparse_grad)

        # input_dim + 1: the extra row is a learned "nothing readable
        # here" token, placed in rows that have no features at all.
        # Without it such a row is entirely padding, attention softmaxes
        # over a row of -inf, and the result is NaN -- which propagates
        # through the batch and makes every loss NaN.
        #
        # Not a hypothetical. Token pruning left 3,633 training rows
        # empty in the last run, and a row whose text prunes away to
        # nothing vectorizes to an all-zero CSR row with no non-zeros.
        # `TokenVocabulary.encode` solves the same problem the same way,
        # returning [<unk>] rather than an empty bag.
        self.empty_index = int(input_dim)
        self.embed = nn.Embedding(input_dim + 1, embed_dim, sparse=self.sparse_grad)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=4 * embed_dim, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.trunk = _build_trunk(embed_dim, hidden_dims, dropout)
        out_dim = hidden_dims[-1] if hidden_dims else embed_dim
        self.heads = nn.ModuleList([nn.Linear(out_dim, n)
                                    for n in n_classes_per_level])

    def _pad(self, indices, offsets):
        """CSR triple -> (padded indices, pad mask).

        `offsets` gives each row's start in the flat index array, so
        the lengths are its differences with the total appended.
        Rows are truncated at `max_tokens`; with a mean of ~7 that
        affects very few, and it stops one 300-token outlier from
        padding an entire batch to 300.

        A row with no features at all gets the `empty_index` token in
        position 0, so it is never fully masked. Attention over an
        entirely masked row softmaxes a vector of -inf and returns
        NaN; one such row poisons the whole batch through the loss.
        """
        n_rows = offsets.numel()
        total = indices.new_tensor([indices.numel()])
        lengths = torch.diff(torch.cat([offsets, total])).clamp(max=self.max_tokens)
        width = int(lengths.max().item()) if n_rows else 1
        width = max(width, 1)

        ar = torch.arange(width, device=indices.device)
        valid = ar[None, :] < lengths[:, None]
        positions = offsets[:, None] + ar[None, :]

        padded = torch.full((n_rows, width), self.empty_index,
                            dtype=torch.long, device=indices.device)
        padded[valid] = indices[positions[valid]]

        # Give empty rows one real position to attend to.
        empty = lengths == 0
        if bool(empty.any()):
            valid[empty, 0] = True          # padded[empty, 0] is already empty_index

        # True marks padding, which is what key_padding_mask expects.
        return padded, ~valid

    def forward(self, x) -> List[torch.Tensor]:
        indices, offsets, _values = x
        padded, pad_mask = self._pad(indices, offsets)

        h = self.embed(padded)
        h = self.encoder(h, src_key_padding_mask=pad_mask)

        # Masked mean. Padding positions carry a real embedding (index
        # 0 is a legitimate feature), so they have to be zeroed before
        # averaging rather than trusted to be harmless.
        keep = (~pad_mask).unsqueeze(-1).to(h.dtype)
        pooled = (h * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)

        z = self.trunk(pooled)
        return [head(z) for head in self.heads]

    def sparse_parameters(self) -> List[nn.Parameter]:
        return [self.embed.weight] if self.sparse_grad else []

    def init_kwargs(self) -> Dict:
        return dict(
            input_dim=self.input_dim,
            n_classes_per_level=self.n_classes_per_level,
            embed_dim=self.embed_dim,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            max_tokens=self.max_tokens,
            sparse_grad=self.sparse_grad,
        )

    @torch.no_grad()
    def self_test(self, verbose: bool = True) -> bool:
        """A four-row forward before committing to an epoch.

        Written because the padding arithmetic is the kind of code
        that produces plausible garbage rather than an exception:
        an off-by-one in the offsets silently shifts every row's
        tokens and the model just trains badly.
        """
        was_training = self.training
        self.eval()
        try:
            # Rows of length 3, 1, 4, 0 -- including an empty one.
            lengths = [3, 1, 4, 0]
            flat = torch.arange(sum(lengths)) % self.input_dim
            offsets = torch.tensor([0, 3, 4, 8], dtype=torch.long)
            values = torch.ones(flat.numel(), dtype=torch.float32)

            padded, mask = self._pad(flat, offsets)
            assert padded.shape[0] == 4, padded.shape
            assert padded[0, :3].tolist() == flat[:3].tolist()
            assert padded[2, :4].tolist() == flat[4:8].tolist()

            # Rows 0-2 keep their own lengths; row 3 is empty and must
            # come back with exactly one valid position, carrying the
            # empty token. A fully masked row is the NaN case.
            kept = (~mask).sum(dim=1).tolist()
            assert kept[:3] == lengths[:3], kept
            assert kept[3] == 1, f"empty row has {kept[3]} valid positions"
            assert padded[3, 0].item() == self.empty_index
            assert not bool(mask.all(dim=1).any()), "a row is entirely masked"

            logits = self((flat, offsets, values))
            for lg, n in zip(logits, self.n_classes_per_level):
                assert lg.shape == (4, n), (lg.shape, n)
                assert torch.isfinite(lg).all(), (
                    "non-finite logits -- check for fully masked rows"
                )

            if verbose:
                print(f"[self_test] ok -- {len(logits)} heads, padded to "
                      f"{padded.shape[1]} tokens, empty row -> empty token")
            return True
        finally:
            self.train(was_training)


MODEL_CLASSES = {
    "MultiHeadSparseMLP": MultiHeadSparseMLP,
    "MultiHeadSparseInputMLP": MultiHeadSparseInputMLP,
    "MultiHeadEmbedMLP": MultiHeadEmbedMLP,
    "MultiHeadAttentionMLP": MultiHeadAttentionMLP,
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def shm_free_bytes() -> Optional[int]:
    """Free space in /dev/shm, or None where that does not apply."""
    import os

    try:
        st = os.statvfs("/dev/shm")
    except (OSError, AttributeError, ValueError):
        return None
    return st.f_bavail * st.f_frsize


def check_worker_shm(
    batch_size: int,
    n_features: int,
    num_workers: int,
    prefetch_factor: int = 2,
    verbose: bool = True,
) -> int:
    """Return a worker count that will not exhaust shared memory.

    DataLoader workers hand finished batches back through /dev/shm, and
    a densified sparse batch is `batch_size x n_features` float32 --
    16 MB at 512 x 8130. With four workers prefetching two batches each
    that is 133 MB in flight, against a container that typically gives
    /dev/shm 64 MB. The failure is not a clean OOM; it is

        RuntimeError: unable to allocate shared memory (shm) ...
        No space left on device

    thrown mid-epoch, after the time has already been spent. Hence
    checking up front rather than letting it happen: the cost of being
    wrong in one direction is a slower epoch, and in the other, a lost
    training run.
    """
    if num_workers <= 0:
        return 0

    per_batch = batch_size * n_features * 4
    needed = per_batch * num_workers * prefetch_factor
    free = shm_free_bytes()

    if free is None:
        return num_workers
    if needed <= free * 0.8:
        if verbose:
            print(f"[loader] {num_workers} workers need ~{needed / 1e6:.0f} MB of "
                  f"/dev/shm; {free / 1e6:.0f} MB free")
        return num_workers

    safe = int((free * 0.8) // (per_batch * prefetch_factor))
    if verbose:
        print(
            f"[loader] {num_workers} workers would need ~{needed / 1e6:.0f} MB of "
            f"/dev/shm but only {free / 1e6:.0f} MB is free "
            f"({per_batch / 1e6:.1f} MB per densified batch). "
            f"Falling back to num_workers={max(safe, 0)}.\n"
            f"[loader] To use workers, cut batch_size or n_features -- or run "
            f"single-process, which needs no shared memory at all."
        )
    return max(safe, 0)


def sparse_loaders(
    X_train: sparse.csr_matrix,
    y_train: np.ndarray,
    X_val: Optional[sparse.csr_matrix] = None,
    y_val: Optional[np.ndarray] = None,
    batch_size: int = 256,
    num_workers: int = 0,
    shuffle_train: bool = True,
    check_shm: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    if check_shm:
        num_workers = check_worker_shm(
            batch_size, X_train.shape[1], num_workers
        )

    train_loader = DataLoader(
        SparseMatrixDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=shuffle_train,
        collate_fn=make_sparse_collate(X_train),
        num_workers=num_workers,
    )
    val_loader = None
    if X_val is not None:
        val_loader = DataLoader(
            SparseMatrixDataset(X_val, y_val),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=make_sparse_collate(X_val),
            num_workers=num_workers,
        )
    return train_loader, val_loader


class LengthBucketedBatches:
    """Batches of similarly-sized rows, so padding stops dominating.

    `MultiHeadAttentionMLP` pads each batch to its longest row. With
    256 random rows per batch you almost always draw one near the
    32-token cap, so the model does 32 tokens of work per row when
    the mean is 6.8 -- a measured 4.7x of pure waste, and the reason
    the attention arm ran 10x slower than the sum model rather than
    the 2-3x the arithmetic predicts.

    Grouping similar lengths together makes the padded width track
    the actual width. Randomness is kept by shuffling first, sorting
    only *within* pools of `pool_factor` batches, and then shuffling
    the batch order -- so batches differ every epoch and are not
    ordered short-to-long, which would correlate every gradient step
    with description length.

    Deliberately not a `torch.utils.data.Sampler` subclass: a plain
    iterable of index lists is all `DataLoader(batch_sampler=...)`
    requires, and keeping torch out of it means the logic is testable
    without torch installed.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        shuffle: bool = True,
        pool_factor: int = 64,
        seed: int = 0,
        drop_last: bool = False,
    ):
        self.lengths = np.asarray(lengths)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.pool_factor = int(pool_factor)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self._epoch = 0

    def __len__(self) -> int:
        n = len(self.lengths)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.lengths)
        order = np.arange(n)
        if self.shuffle:
            # A different shuffle each epoch, so the pools -- and
            # therefore the batches -- are not the same every time.
            rng = np.random.default_rng(self.seed + self._epoch)
            rng.shuffle(order)
            self._epoch += 1

        pool = max(self.batch_size * self.pool_factor, self.batch_size)
        batches = []
        for start in range(0, n, pool):
            chunk = order[start:start + pool]
            chunk = chunk[np.argsort(self.lengths[chunk], kind="stable")]
            for lo in range(0, len(chunk), self.batch_size):
                batch = chunk[lo:lo + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch.tolist())

        if self.shuffle:
            # Without this the epoch runs short rows first and every
            # gradient step early on sees only short descriptions.
            np.random.default_rng(self.seed + self._epoch + 10_000).shuffle(batches)
        return iter(batches)

    def mean_padded_width(self) -> float:
        """Average width these batches will pad to. The diagnostic."""
        widths = [int(self.lengths[b].max()) for b in self]
        sizes = [len(b) for b in self]
        return float(np.average(widths, weights=sizes)) if widths else 0.0


def csr_loaders(
    X_train: sparse.csr_matrix,
    y_train: np.ndarray,
    X_val: Optional[sparse.csr_matrix] = None,
    y_val: Optional[np.ndarray] = None,
    batch_size: int = 512,
    num_workers: int = 0,
    shuffle_train: bool = True,
    persistent_workers: bool = True,
    bucket_by_length: bool = False,
    pool_factor: int = 64,
    seed: int = 42,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Loaders for `MultiHeadSparseInputMLP`.

    No `/dev/shm` check here, unlike `sparse_loaders`: a batch carries
    its non-zeros rather than a dense block, so it is kilobytes rather
    than tens of megabytes and worker processes are safe.

    `persistent_workers` keeps them alive between epochs. Without it
    the workers are forked and torn down every epoch, which on
    Databricks also reprints the gRPC fork warnings each time -- the
    four `skipping fork() handlers` lines are harmless in themselves
    (the children never touch gRPC) but they are a symptom of paying
    the respawn cost 25 times over.
    """
    persist = bool(persistent_workers) and num_workers > 0

    def _loader(X, y, shuffle):
        common = dict(collate_fn=CSRTripleCollate(X),
                      num_workers=num_workers, persistent_workers=persist)
        if bucket_by_length:
            # batch_sampler is mutually exclusive with batch_size and
            # shuffle, so they are not passed alongside it.
            sampler = LengthBucketedBatches(
                np.diff(X.indptr), batch_size, shuffle=shuffle,
                pool_factor=pool_factor, seed=seed)
            return DataLoader(CSRTripleDataset(X, y),
                              batch_sampler=sampler, **common)
        return DataLoader(CSRTripleDataset(X, y), batch_size=batch_size,
                          shuffle=shuffle, **common)

    train_loader = _loader(X_train, y_train, shuffle_train)
    val_loader = _loader(X_val, y_val, False) if X_val is not None else None
    return train_loader, val_loader


def token_loaders(
    enc_train: Sequence[Sequence[int]],
    y_train: np.ndarray,
    enc_val: Optional[Sequence[Sequence[int]]] = None,
    y_val: Optional[np.ndarray] = None,
    batch_size: int = 256,
    num_workers: int = 0,
    unk_index: int = 1,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    collate = make_offsets_collate(unk_index)
    train_loader = DataLoader(
        TokenIndexDataset(enc_train, y_train),
        batch_size=batch_size,
        shuffle=shuffle_train,
        collate_fn=collate,
        num_workers=num_workers,
    )
    val_loader = None
    if enc_val is not None:
        val_loader = DataLoader(
            TokenIndexDataset(enc_val, y_val),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=num_workers,
        )
    return train_loader, val_loader


def to_device(batch_x, device: str):
    """Move either batch shape to the device."""
    if isinstance(batch_x, (tuple, list)):
        return tuple(t.to(device) for t in batch_x)
    return batch_x.to(device)
