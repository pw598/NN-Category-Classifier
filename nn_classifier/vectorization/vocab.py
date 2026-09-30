"""The integer-index vocabulary behind the dense network.

`nn.EmbeddingBag` wants token *indices*, not a sparse matrix. This is
the piece that turns cleaned text into those indices and back again.

Built on document frequency measured on the training split only, for
the same reason everything else in this repo is: a vocabulary that has
seen the validation set makes the validation score a fiction.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

from ..cleaning.tokens import tokenize
from ..config import TokenVocabConfig


@dataclass
class TokenVocabulary:
    """A picklable string <-> index map, saved alongside the model.

    Index 0 is always the pad token and index 1 the unknown token. That
    is a contract with the model: `padding_idx=0` in the embedding
    layer depends on it, and so does the empty-document guard in the
    collate function.
    """

    itos: List[str]
    stoi: Dict[str, int] = field(default_factory=dict)
    lowercase: bool = True
    token_pattern: str = r"\b\w\w+\b"
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"

    def __post_init__(self):
        if not self.stoi:
            self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    @property
    def pad_index(self) -> int:
        return self.stoi[self.pad_token]

    @property
    def unk_index(self) -> int:
        return self.stoi[self.unk_token]

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> List[int]:
        """Text to indices. Never empty: an empty result becomes [<unk>].

        EmbeddingBag with mean pooling divides by the bag size, so a
        genuinely empty bag produces NaN rather than an error, and the
        NaN then poisons the whole batch through the loss. One <unk> is
        a cheap way to keep that from happening, and it also gives the
        model a consistent representation for 'nothing readable here'.
        """
        tokens = tokenize(text, lowercase=self.lowercase, pattern=self.token_pattern)
        idx = [self.stoi[t] for t in tokens if t in self.stoi]
        return idx or [self.unk_index]

    def encode_all(self, texts: Iterable[str]) -> List[List[int]]:
        return [self.encode(t) for t in texts]

    def oov_rate(self, texts: Iterable[str]) -> float:
        """Share of tokens that fall outside the vocabulary.

        The single most useful diagnostic when scoring quality drops
        without the code changing: a rise here means the cleaning or
        the input distribution moved, not the model.
        """
        total = hits = 0
        for text in texts:
            for tok in tokenize(text, self.lowercase, self.token_pattern):
                total += 1
                hits += tok in self.stoi
        return 0.0 if total == 0 else 1.0 - hits / total


def build_vocabulary(
    train_texts: Sequence[str],
    cfg: Optional[TokenVocabConfig] = None,
    verbose: bool = True,
) -> TokenVocabulary:
    """Document-frequency vocabulary over the training texts.

    Document frequency, not term frequency: a token repeated five
    times in one description is one description's worth of evidence.
    """
    cfg = cfg or TokenVocabConfig()
    df_counts: Counter = Counter()
    for text in train_texts:
        df_counts.update(set(tokenize(text, cfg.lowercase, cfg.token_pattern)))

    kept = [tok for tok, c in df_counts.items() if c >= cfg.min_df]
    kept.sort(key=lambda t: (-df_counts[t], t))
    if cfg.vocab_size:
        kept = kept[: cfg.vocab_size]

    itos = [cfg.pad_token, cfg.unk_token] + kept
    vocab = TokenVocabulary(
        itos=itos,
        lowercase=cfg.lowercase,
        token_pattern=cfg.token_pattern,
        pad_token=cfg.pad_token,
        unk_token=cfg.unk_token,
    )
    if verbose:
        print(
            f"[vocab] {len(vocab):,} entries "
            f"({len(kept):,} tokens with min_df={cfg.min_df}, "
            f"{len(df_counts):,} seen in total)"
        )
    return vocab
