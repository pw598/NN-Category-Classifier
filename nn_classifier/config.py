"""Every tunable in one place.

The rule the old repo followed and this one keeps: no column name, file
path or hyperparameter is hard-coded anywhere else in the package. A
notebook builds one of these, prints it, and passes it down. When a run
is saved, `to_dict()` is what gets written alongside the weights, so a
result can always be traced back to the settings that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import paths

# The four-level hierarchy, by ID. IDs rather than names on purpose: a
# category name can repeat under different parents ('ACCESSORIES' lives
# in half a dozen places), so a path built from names is not unique and
# joint-path scoring would silently merge branches. Names come back at
# predict time from the saved hierarchy lookup.
DEFAULT_LEVEL_COLUMNS: Tuple[str, ...] = (
    "Level 1 ID",
    "Level 2 ID",
    "Level 3 ID",
    "Level 4 ID",
)

DEFAULT_NAME_COLUMNS: Tuple[str, ...] = (
    "Level 1 Name",
    "Level 2 Name",
    "Level 3 Name",
    "Level 4 Name",
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Where the rows come from and what the columns are called."""

    path: str = field(default_factory=lambda: paths.data_path("cleaned_data.txt"))
    file_format: str = "csv"           # csv | excel | delta | spark_table
    delimiter: str = "|"
    text_col: str = "FullDesc"
    id_col: Optional[str] = "ID"
    # The SUPPLIER, not the item. This defaulted to "Vallen ID" for a
    # long time, which is `erp_product_number` in the pull -- one value
    # per SKU. `attach_vendor=True` therefore prepended an item
    # identifier to every description, which memorises perfectly and
    # generalises to nothing. Notebook 11a caught it only after
    # building the dataset ("mean rows per vendor: 1").
    #
    # "Global Vendno" is `global_vendor_number`, which is what the
    # marker was always meant to carry.
    vendor_col: Optional[str] = "Global Vendno"
    level_columns: Tuple[str, ...] = DEFAULT_LEVEL_COLUMNS
    name_columns: Tuple[str, ...] = DEFAULT_NAME_COLUMNS
    drop_null_text: bool = True
    drop_null_labels: bool = True


@dataclass
class CleaningConfig:
    """The description-cleaning recipe.

    Mirrors the old repo's procedure, plus the steps that had only ever
    lived in notebooks: misspelling normalisation, the vocabulary filter,
    and vendor-ID attachment.
    """

    strip_leading_code: bool = True
    uppercase: bool = False

    # Misspelling / variant normalisation from to_normalize.csv, applied
    # before the regex pass so the corrected forms are what the later
    # steps and the vocabulary filter actually see.
    normalize_misspellings: bool = False

    # Abbreviation expansion from abbrevs_to_expand.csv, applied last.
    expand_abbrevs: bool = False

    # Keep only tokens that are recognisable: an English word, or a
    # designated good token, or an abbreviation we know about. Off by
    # default because it is lossy and worth switching on deliberately.
    filter_to_known_tokens: bool = False
    keep_abbrev_keys: bool = True      # abbreviations are known tokens too
    min_token_length: int = 2

    # Prepend a vendor marker (VND_<id>) to each description. Pair this
    # with count/binary vectorization rather than TF-IDF -- IDF would
    # down-weight exactly the signal the marker is there to carry.
    attach_vendor: bool = False
    vendor_token_prefix: str = "VND_"

    # Extra categorical codes folded in as tokens, the same way the
    # vendor is. Maps a source column to how it is emitted:
    #
    #   {"ICSP Prodcat": {"prefix": "CAT_"},
    #    "UNSPSC": {"prefix": "UNSPSC", "levels": (2, 4, 6, 8)},
    #    "Prodline": {"prefix": "PLINE_"}}
    #
    # `levels` is for hierarchical codes -- see attach_marker_tokens.
    # Scoring reads this back out of the saved config, so a bundle
    # cannot be fed differently from how it was trained.
    marker_columns: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class FilterConfig:
    """Row- and token-level filters applied after cleaning."""

    # Drop rows whose deepest-level class appears fewer than this many
    # times. Also what makes a stratified split possible.
    min_class_count: int = 6
    # Drop rows whose full L1..L4 path appears fewer than this many times.
    min_path_count: int = 2
    # Drop tokens appearing in fewer than this many descriptions. Counted
    # on the training split only, then applied to both, to avoid leakage.
    min_token_count: int = 1
    # Drop tokens that appear under fewer than this many distinct classes
    # at the deepest level. 0 disables.
    min_token_classes: int = 0
    ngram_range_for_counts: Tuple[int, int] = (1, 1)


@dataclass
class SplitConfig:
    test_size: float = 0.1
    random_state: int = 42
    stratify: bool = True


# ---------------------------------------------------------------------------
# Vectorization
# ---------------------------------------------------------------------------

@dataclass
class SparseVectorizerConfig:
    """BOW / TF-IDF, for the sparse neural network and any sklearn model."""

    kind: str = "tfidf"                # tfidf | count
    max_features: Optional[int] = 16000
    ngram_range: Tuple[int, int] = (1, 1)
    min_df: int | float = 2
    max_df: int | float = 0.95
    stop_words: Optional[str] = None
    lowercase: bool = True
    binary: bool = False
    sublinear_tf: bool = True          # tfidf only
    norm: Optional[str] = "l2"         # tfidf only
    use_idf: bool = True               # tfidf only
    # Fold bigrams to an order-insensitive form, so 'ball valve' and
    # 'valve ball' are one feature. Unigrams and trigrams keep order.
    unordered_bigrams: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenVocabConfig:
    """The integer-index vocabulary behind the dense (EmbeddingBag) network."""

    vocab_size: Optional[int] = 16000
    min_df: int = 2
    lowercase: bool = True
    token_pattern: str = r"\b\w\w+\b"
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"


@dataclass
class Word2VecConfig:
    """Standalone Word2Vec, for initialising embeddings or pooling doc vectors."""

    backend: str = "gensim"            # gensim | spark
    vector_size: int = 150
    window: int = 5
    min_count: int = 2
    epochs: int = 5
    sg: int = 0                        # gensim: 0=CBOW, 1=skip-gram
    workers: int = 4
    seed: int = 42
    # spark backend only
    num_partitions: int = 4
    step_size: float = 0.025
    # Document pooling for the sklearn models.
    pooling: str = "mean"              # mean | sum | max
    normalize_docvecs: bool = False


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass
class NNConfig:
    """Shared settings for both multi-head networks."""

    hidden_dims: Tuple[int, ...] = (512, 256)
    dropout: float = 0.2
    embed_dim: int = 150               # dense network only
    batch_size: int = 256
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-5
    label_smoothing: float = 0.05
    patience: int = 4                  # early stopping
    scheduler_patience: int = 2
    scheduler_factor: float = 0.5

    # What early stopping and the weight restore watch.
    #   "val_loss"  -- validation cross-entropy.
    #   "accuracy"  -- accuracy at `monitor_level` (-1 = deepest).
    #
    # "accuracy" is the default because on this problem the two
    # decouple. With 3,473 classes, cross-entropy keeps worsening on
    # the examples the model is confidently wrong about even while the
    # argmax improves elsewhere -- so validation loss plateaus while
    # accuracy is still climbing. Every 04 variant lost 0.004-0.006 of
    # L4 accuracy to a restore triggered by loss.
    #
    # The LR scheduler still follows loss, which is the smoother
    # signal and the right one for deciding when to slow down.
    monitor: str = "accuracy"          # "val_loss" | "accuracy"
    monitor_level: int = -1

    # Sparse network only. Emit a sparse gradient from the first layer
    # and update it with SparseAdam, leaving everything else on AdamW.
    #
    # Measured, not assumed. In the 03a vocabulary sweep epoch time
    # tracked the feature count almost exactly (23,319 features ->
    # 108s, 80,000 -> 374s) while non-zeros per row stayed at 5.3-5.7.
    # The forward gather was never the cost: a dense-gradient
    # EmbeddingBag hands back a full (V x hidden) gradient every step,
    # and AdamW rewrites all of it plus two moment buffers whether or
    # not a row was touched.
    #
    # Off by default because it changes the optimizer, and the two
    # paths are not bit-identical run to run even though the forward
    # arithmetic is the same. Turn it on for large vocabularies.
    # One real difference: SparseAdam takes no weight_decay, so the
    # first layer trains unregularised while the rest keeps
    # `weight_decay`. At 1e-5 that is immaterial, and dropout plus
    # early stopping are doing the regularising here -- but it is a
    # difference, not an equivalence, so it is written down rather
    # than papered over.
    sparse_grad: bool = False

    # Sparse network only. Drop whole input tokens during training,
    # before the gather. The `dropout` above acts on the 512-wide
    # output of the first layer and cannot stop a rare feature
    # memorising the few rows it appears in; this can. Matters
    # because SparseAdam applies no weight decay, leaving 98% of the
    # parameters otherwise unregularised.
    feature_dropout: float = 0.0

    seed: int = 42
    prefer_gpu: bool = True
    num_workers: int = 0
    # Per-level loss weights; None means equal weight on every active level.
    level_loss_weights: Optional[Sequence[float]] = None
    # Dense network: seed the embedding matrix from a trained Word2Vec.
    init_from_word2vec: bool = False
    freeze_embeddings: bool = False

    # How EmbeddingBag pools a description's word vectors.
    #   "mean" -- the average. Scale-invariant, so a one-token and a
    #             six-token description produce vectors of the same
    #             magnitude. On a catalogue averaging ~3.5 tokens that
    #             discards a real signal: how much the model had to go on.
    #   "sum"  -- keeps it. Magnitude now carries token count, which the
    #             first BatchNorm partly but not wholly removes.
    #   "max"  -- per-dimension maximum. Does not support padding_idx.
    pooling_mode: str = "mean"


@dataclass
class SklearnModelConfig:
    """One of the registered sklearn estimators, fed pooled dense vectors."""

    name: str = "sgd_logistic"         # see sklearn_models.REGISTRY
    params: Dict[str, Any] = field(default_factory=dict)
    scale_features: bool = True
    n_folds: int = 5                   # out-of-fold cross-validation
    cross_validate: bool = True
    random_state: int = 42


# ---------------------------------------------------------------------------
# Hierarchy and calibration
# ---------------------------------------------------------------------------

@dataclass
class HierarchyConfig:
    """How the four levels are combined into one answer.

    joint_mode:
      'independent'  -- multiply the per-level probabilities, restricted
                        to paths seen in training. Assumes the levels are
                        independent, which they are not, but it is cheap
                        and works surprisingly well.
      'conditional'  -- chain rule: P(L1) * P(L2|L1) * P(L3|L1,L2) * ...
                        estimated by renormalising each level's head over
                        only the children of the chosen parent.
      None           -- no joining at all; score each level on its own.

    predict_all_levels=False emits only the deepest active level.
    """

    levels: Tuple[str, ...] = DEFAULT_LEVEL_COLUMNS
    joint_mode: Optional[str] = "independent"
    predict_all_levels: bool = True
    # Renormalise the joint score over valid paths so it reads as a
    # probability. False leaves the raw product, which is a likelihood.
    normalise: bool = True
    top_k: int = 3
    batch_size: int = 4096

    def active_levels(self) -> List[str]:
        return list(self.levels) if self.predict_all_levels else [self.levels[-1]]


@dataclass
class CalibrationConfig:
    """Turning a confidence number into an accuracy you can act on.

    Section D of the draft notebooks, generalised: fit a calibrator on
    validation, then characterise confidence -> correctness with a
    reliability diagram, ECE, and a coverage/accuracy table.
    """

    method: Optional[str] = "temperature"   # temperature | isotonic | None
    n_bins: int = 10
    thresholds: Tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    target_accuracy: float = 0.95
    # Carry margin and entropy through onto predictions alongside
    # confidence; useful for routing a review queue.
    include_margin_entropy: bool = True
    default_threshold: float = 0.60
    max_iter: int = 100                     # temperature scaling, LBFGS


# ---------------------------------------------------------------------------
# The whole run
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    """One object to print at the top of a notebook and save with the run."""

    data: DataConfig = field(default_factory=DataConfig)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    sparse: SparseVectorizerConfig = field(default_factory=SparseVectorizerConfig)
    vocab: TokenVocabConfig = field(default_factory=TokenVocabConfig)
    word2vec: Word2VecConfig = field(default_factory=Word2VecConfig)
    nn: NNConfig = field(default_factory=NNConfig)
    sklearn: SklearnModelConfig = field(default_factory=SklearnModelConfig)
    hierarchy: HierarchyConfig = field(default_factory=HierarchyConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    notes: str = ""

    def __post_init__(self):
        # One source of truth for the level columns.
        if tuple(self.hierarchy.levels) != tuple(self.data.level_columns):
            self.hierarchy.levels = tuple(self.data.level_columns)
        self.validate()

    def validate(self, raise_on_error: bool = False) -> List[str]:
        """Catch the config mistakes that do not raise on their own.

        Every one of these produces a run that completes and is wrong,
        which is the category worth spending code on. A config that
        crashes tells you about itself.
        """
        problems: List[str] = []

        if self.cleaning.attach_vendor and self.sparse.kind == "tfidf":
            problems.append(
                "attach_vendor is on but sparse.kind is 'tfidf'. IDF "
                "down-weights a marker that appears on every one of a large "
                "vendor's rows, suppressing exactly the vendors there is most "
                "data for. Use kind='count' (optionally binary=True)."
            )
        # The marker must not be the item identifier. This cannot be
        # checked from the config alone -- it needs the data -- but the
        # column name is a strong tell, and the mistake cost a full
        # experiment before anyone noticed.
        if (self.cleaning.attach_vendor and self.data.vendor_col
                and self.data.vendor_col.strip().lower() in
                {"vallen id", "id", "sku", "erp_product_number", "product number"}):
            problems.append(
                f"attach_vendor is on with vendor_col={self.data.vendor_col!r}, "
                "which looks like an item identifier rather than a supplier. "
                "One value per row memorises perfectly and generalises to "
                "nothing. 'Global Vendno' is the supplier column in the pull."
            )

        if self.cleaning.attach_vendor and not self.data.vendor_col:
            problems.append(
                "attach_vendor is on but data.vendor_col is unset, so there "
                "is no column to build the marker from."
            )
        if self.hierarchy.joint_mode not in ("independent", "conditional", None):
            problems.append(
                f"hierarchy.joint_mode={self.hierarchy.joint_mode!r}; expected "
                "'independent', 'conditional' or None."
            )
        if self.calibration.method not in ("temperature", "isotonic", None, "none"):
            problems.append(
                f"calibration.method={self.calibration.method!r}; expected "
                "'temperature', 'isotonic' or None."
            )
        if self.sklearn.cross_validate and self.filters.min_class_count < self.sklearn.n_folds:
            problems.append(
                f"filters.min_class_count={self.filters.min_class_count} is below "
                f"sklearn.n_folds={self.sklearn.n_folds}; a stratified k-fold "
                "needs at least n_folds members in every class."
            )
        if self.nn.pooling_mode not in ("mean", "sum", "max"):
            problems.append(
                f"nn.pooling_mode={self.nn.pooling_mode!r}; expected 'mean', "
                "'sum' or 'max'."
            )
        if self.nn.freeze_embeddings and not self.nn.init_from_word2vec:
            problems.append(
                "nn.freeze_embeddings is on without init_from_word2vec, which "
                "freezes a randomly initialised embedding matrix -- the network "
                "would train a classifier on noise."
            )
        if (self.nn.level_loss_weights is not None
                and len(self.nn.level_loss_weights) != len(self.data.level_columns)):
            problems.append(
                f"nn.level_loss_weights has {len(self.nn.level_loss_weights)} "
                f"entries for {len(self.data.level_columns)} levels."
            )

        if problems:
            message = "\n".join(f"  - {p}" for p in problems)
            if raise_on_error:
                raise ValueError("Invalid RunConfig:\n" + message)
            print("[config] WARNING:\n" + message)
        return problems

    def active_levels(self) -> List[str]:
        return self.hierarchy.active_levels()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def describe(self) -> str:
        import json

        return json.dumps(self.to_dict(), indent=2, default=str)


def save_config(cfg: RunConfig, path=None) -> Any:
    """Persist a config as both a pickle and readable JSON.

    Two formats because they are read by different things. The pickle
    is what notebooks 03-05 load, so they cannot silently diverge from
    the decisions made in notebook 02. The JSON is what a person reads
    six months later when asking what settings produced a result, and
    it stays readable when the pickle stops loading.
    """
    import joblib

    from . import paths as _paths

    path = _paths.output_path("run_config.joblib") if path is None else path
    joblib.dump(cfg, path)
    json_path = str(path).rsplit(".", 1)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as fh:
        fh.write(cfg.describe())
    print(f"[config] wrote {path} and {json_path}")
    return path


def load_config(path=None, default: Optional[RunConfig] = None) -> RunConfig:
    """Load the saved config, falling back to a fresh default.

    The fallback exists so a model notebook can be run on its own
    before notebook 02 has been executed. It prints which it used --
    a run that silently fell back to defaults and a run that loaded
    the tuned config look identical otherwise, and only one of them
    is comparable to the others.
    """
    import joblib

    from . import paths as _paths

    path = _paths.output_dir() / "run_config.joblib" if path is None else path
    try:
        cfg = joblib.load(path)
        print(f"[config] loaded {path}")
        return cfg
    except (FileNotFoundError, OSError):
        print(f"[config] no saved config at {path}; using defaults")
        return default or RunConfig()


def set_dotted(cfg: RunConfig, key: str, value: Any) -> None:
    """Set 'nn.dropout' style paths, for notebook-side sweeps."""
    obj = cfg
    parts = key.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    if not hasattr(obj, parts[-1]):
        raise AttributeError(f"No such config field: {key!r}")
    setattr(obj, parts[-1], value)


def apply_params(cfg: RunConfig, params: Dict[str, Any]) -> RunConfig:
    """Apply a {dotted_key: value} mapping in place, returning the config."""
    for key, value in params.items():
        set_dotted(cfg, key, value)
    return cfg
