"""Text cleaning: the same recipe for every model downstream.

Three networks and an open-ended set of sklearn models all come through
here. That is the point -- if the sparse network and the dense one clean
their text even slightly differently, their numbers are not comparable,
and the difference will look like a modelling result.

`descriptions` holds the full procedure ported from the old repo, plus
the steps that had only ever lived in notebooks. `text` is the lighter,
configurable recipe. `tokens` holds the counting and filtering that
happens after cleaning.
"""

from .lists import (  # noqa: F401
    load_cleaning_lists,
    load_expansions,
    load_good_tokens,
    load_normalizations,
    sort_by_length,
    upper,
)
from .descriptions import (  # noqa: F401
    DescriptionCleaner,
    add_first_word_real,
    apply_patterns,
    attach_marker_tokens,
    attach_vendor_token,
    build_cleaner,
    build_known_tokens,
    build_patterns,
    clean_descriptions,
    clean_descriptions_with,
    cleaner_from_config,
    expand_abbreviations,
    filter_known_tokens,
    first_word_is_real,
    load_english_vocabulary,
    normalize_misspellings,
    regex_cleaning_proc,
    strip_first_term,
)
from .text import (  # noqa: F401
    CleaningConfig as SimpleCleaningConfig,
    clean_dataframe,
    clean_series,
)
from .tokens import (  # noqa: F401
    count_ngrams,
    drop_one_off_tokens,
    drop_rare_classes,
    drop_rare_paths,
    prune_to_vocabulary,
    token_report,
    tokenize,
)

__all__ = [
    "DescriptionCleaner",
    "SimpleCleaningConfig",
    "add_first_word_real",
    "apply_patterns",
    "attach_marker_tokens",
    "attach_vendor_token",
    "build_cleaner",
    "build_known_tokens",
    "build_patterns",
    "clean_dataframe",
    "clean_descriptions",
    "clean_descriptions_with",
    "clean_series",
    "cleaner_from_config",
    "count_ngrams",
    "drop_one_off_tokens",
    "drop_rare_classes",
    "drop_rare_paths",
    "expand_abbreviations",
    "filter_known_tokens",
    "first_word_is_real",
    "load_cleaning_lists",
    "load_english_vocabulary",
    "load_expansions",
    "load_good_tokens",
    "load_normalizations",
    "normalize_misspellings",
    "prune_to_vocabulary",
    "regex_cleaning_proc",
    "sort_by_length",
    "strip_first_term",
    "token_report",
    "tokenize",
    "upper",
]
