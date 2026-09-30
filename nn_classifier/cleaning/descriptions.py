"""The description-cleaning procedure used to build the golden set.

A faithful port of the original notebook procedure, which is considerably
more aggressive than cleaning.text.CleaningConfig: as well as stripping
colours, sizes and units, it removes pack counts, dimension strings,
fractions, standalone numbers and all remaining punctuation. What survives
is close to bare nouns and adjectives, which is the point -- a category is
determined by what a thing *is*, not by how big it is or how many come in
the box.

Two steps in here have no equivalent in CleaningConfig:

first-word stripping
    Descriptions frequently open with a vendor code or part number
    ('QJX8A TORQUE HEAD BOXEND'). If the first token is not a recognisable
    English word it is dropped, on the reasoning that it identifies the
    individual product rather than its category -- and a token unique to
    one SKU is noise the model would otherwise try to learn from.

the two-phase regex pass
    Order matters and the phases are not interchangeable. Phase 1 removes
    the compound patterns while the punctuation that delimits them is
    still present -- '1-1/4X3-1/2' is only recognisable as a dimension
    while it still has its slashes. Phase 2 then removes the reference-list
    terms and finally the punctuation itself. Run the other way round, the
    dimension strings fragment into stray digits that the earlier patterns
    would no longer match.

Note the text is *not* uppercased here, matching the original. The
patterns and the reference lists are uppercase, so this assumes the source
descriptions already are -- true of the ERP extract. Pass
uppercase=True if that stops holding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

from .lists import (
    load_cleaning_lists,
    load_expansions,
    load_good_tokens,
    load_normalizations,
    upper,
)

DEFAULT_ORIG_COL = "FullDesc_orig"
FIRST_WORD_COL = "FirstWordReal"

# Units that may legitimately trail a number inside a dimension string, as
# opposed to the tokens that make up the dimension itself.
SPEC_UNITS = ["FT", "IN", "CM", "MM", "MIL", "TPI", "TP", "T"]


# ---------------------------------------------------------------------------
# Primitives, on a whole frame
# ---------------------------------------------------------------------------

def regex_deletion(df: pd.DataFrame, column: str, patterns: Sequence[str]) -> pd.DataFrame:
    """Delete every match of every pattern, in order, in place on a copy."""
    out = df.copy()
    series = out[column].astype(str)
    for pattern in patterns:
        series = series.str.replace(pattern, "", regex=True)
    out[column] = series
    return out


def regex_replacement(
    df: pd.DataFrame, column: str, pattern_replacements: Dict[str, str]
) -> pd.DataFrame:
    """Apply {pattern: replacement} substitutions in dict order."""
    out = df.copy()
    series = out[column].astype(str)
    for pattern, replacement in pattern_replacements.items():
        series = series.str.replace(pattern, replacement, regex=True)
    out[column] = series
    return out


# ---------------------------------------------------------------------------
# First-word stripping
# ---------------------------------------------------------------------------

def load_english_vocabulary(download: bool = True) -> set:
    """The NLTK English word list, lowercased.

    Two separate things have to be present, and only the first is a pip
    install: the nltk library, and the 'words' corpus, which ships as
    data rather than with the package. Both are handled here, at the
    point of use, so no notebook needs a setup cell.

    Downloading the corpus needs network access and a writable NLTK
    data directory, neither of which is guaranteed on a cluster. Pass
    your own set to first_word_is_real / add_first_word_real to avoid
    the dependency entirely -- and note that build_cleaner pickles the
    resolved word list into the cleaner, so scoring never repeats this.
    """
    from ..deps import auto_install

    auto_install("nltk", purpose="the leading-code test in cleaning")

    import nltk
    from nltk.corpus import words

    if download:
        try:
            nltk.data.find("corpora/words")
        except LookupError:
            nltk.download("words", quiet=True)
    return {w.lower() for w in words.words()}


def first_word_is_real(text, vocabulary: set) -> int:
    """1 when the leading token looks like an English word, else 0.

    Three ways to score 0, and each is deliberate:
      * two characters or fewer -- too short to be a meaningful noun, and
        usually a code fragment;
      * contains a digit -- part numbers and sizes;
      * not in the vocabulary -- vendor codes, abbreviations, model names.
    """
    if not isinstance(text, str) or not text.strip():
        return 0
    first_word = text.split(maxsplit=1)[0]
    first_clean = first_word.strip('.,;:!?"\'()[]{}').lower()
    if len(first_clean) <= 2:
        return 0
    if any(char.isdigit() for char in first_clean):
        return 0
    return 1 if first_clean in vocabulary else 0


def add_first_word_real(
    df: pd.DataFrame, column: str, vocabulary: set, out_col: str = FIRST_WORD_COL
) -> pd.DataFrame:
    """Add the 0/1 flag as a column, keeping it for later inspection."""
    out = df.copy()
    out[out_col] = out[column].apply(lambda t: first_word_is_real(t, vocabulary))
    return out


def strip_first_term(
    df: pd.DataFrame, column: str, flag_col: str = FIRST_WORD_COL
) -> pd.DataFrame:
    """Drop the leading token wherever the flag says it is not a real word.

    Only the first token, and only once -- a description opening with two
    codes keeps the second. That is the original behaviour; whether it
    should recurse is a real question, and one worth answering with the
    counts rather than by assumption.
    """
    out = df.copy()

    def _strip(row):
        if row[flag_col] == 0 and isinstance(row[column], str):
            parts = row[column].split(maxsplit=1)
            return parts[1] if len(parts) > 1 else ""
        return row[column]

    out[column] = out.apply(_strip, axis=1)
    return out


# ---------------------------------------------------------------------------
# The regex procedure
# ---------------------------------------------------------------------------

def build_patterns(lists: Dict[str, List[str]]) -> Dict[str, object]:
    """The four pattern groups, built from the reference lists.

    Returned rather than applied so they can be printed and inspected --
    a wall of regex that only ever runs is a wall of regex nobody checks.

    Note the lists are interpolated raw, not regex-escaped, except for
    `sizes` in the SZ pattern. That is how the original was written; terms
    containing regex metacharacters would therefore behave as patterns
    rather than literals.
    """
    colors = lists["colors"]
    color_abbrevs = lists["color_abbrevs"]
    sizes = lists["sizes"]
    units = lists["units"]
    unit_abbrevs = lists["unit_abbrevs"]
    unit_plurals = lists["unit_plurals"]

    spec_units_pattern = "|".join(SPEC_UNITS)
    non_spec_token = r"\d+|\.|/|-|X"
    all_tokens = rf"{non_spec_token}|{spec_units_pattern}"
    sizes_pattern = "|".join(re.escape(s) for s in sizes)

    deletion_1 = [
        # DV or DC indicators
        r"\(DV\)|\[DV\]|\(DC\)|\[DC\]",
        # number followed by a slash and then PK, etc.
        r"(?<=\s)\d+/(CT|CNT|CA|PK|BX|BG|CASE|PACK|BOX|BAG|RL|ROLL)",
        # number (or number followed by a dash) followed by CNT/PK, etc.
        r"(?<=\s)\d+(?:-)?/(CNT|CT)/(CA|PK|BX|BG|RL|ROLL)",
        # REMOVED 2026-09-28 -- order 4, rank 5.
        # Matched any run of two or more digits, so it deleted carbide
        # grades (4325), series numbers (390) and model numbers (3430)
        # along with the dimensions it was written for. It could not
        # tell a magnitude from an identity code.
        #   rf"(?<=\s)(?:{non_spec_token})(?:{all_tokens}){{1,}}(?=\s|$)",
        # periods
        r"[.]",
        r"(?:\d+-)?\d+/\d+[Xx]\d+",
        r"\d+/\d+[Xx]\d+-\d+/\d+",
        r"\d+[Xx]\d+-\d+/\d+",
        # #/#X#
        r"\d+/\d+[Xx]\d+",
        r"\d+-\d+/\d+",
        # #/#
        r"\d+/\d+",
    ]

    replacement_1 = {
        # replace apostrophe with a space
        r"[']": " ",
    }

    deletion_2 = [
        # delete #X#<unit> or #X#X#<unit>
        r"(?<=\s)\d+(?:(?:X\d+)|(" + "|".join(unit_abbrevs) + r"))*("
        + "|".join(unit_abbrevs) + r")(?=\s|$)",
        # numbers followed by unit abbreviation, if followed by space or end
        r"(?<=\s)\d+(" + "|".join(unit_abbrevs) + r")(?=\s|$)",
        # REMOVED 2026-09-28 -- order 15, rank 5.
        # Pressure and power ratings are category signals, not sizes:
        # a 3000PSI hydraulic hose and a 150PSI air hose are different
        # categories.
        #   r"(?<=\s)(?:\d+\s*(?:PSI|HP)|PSI|HP)",
        # color abbrevs
        r"(?<=\s)(" + "|".join(color_abbrevs) + r")(?=\s|$)",
        # colors
        r"(?<=\s)(" + "|".join(colors) + r")(?=\s|$)",
        # sizes
        r"(?:^|\s)(" + "|".join(sizes) + r")/(" + "|".join(sizes) + r")(?=\s|$)",
        r"(?<=\s)(" + "|".join(sizes) + r")(?=\s|$)",
        # REMOVED 2026-09-28 -- orders 20, 21 and 22 (ranks 3, 3, 4).
        # The magnitude is the noisy half; the unit is not. There are
        # only ~31 unit abbreviations, so no cardinality problem, and
        # they say what kind of thing the item is: GAL implies a
        # liquid, FT a hose or cable, LB bulk material. Order 22 also
        # over-matched -- it deleted the "LB" from "CROUSE-HINDS COND
        # BODY LB", where LB is a conduit body type.
        #   r"(?<=\s)(" + "|".join(unit_plurals) + r")(?=\s|$)",
        #   r"(?<=\s)(" + "|".join(units) + r")(?=\s|$)",
        #   r"(?<=\s)(" + "|".join(unit_abbrevs) + r")(?=\s|$)",
        # numbers followed by CT, CNT, CA, etc.
        r"(?<=\s)\d+(CT|CNT|CA|PK|BX|BG)(?=\s|$)",
        # instances of "W/"
        r"W/",
        # instances of #X# mixed with slashes and/or dashes
        r"(?i)(?<=\s)(?:(?:\d+(?:\.\d+)?|X|/|-)){2,}(?=\s|$)",
        # things like 1-1/4X3-1/2
        r"\d+-\d+/\d+\s*X\s*\d+-\d+/\d+",
        # things like 1-1/2
        r"\b\d+-\d+/\d+",
        # MODIFIED 2026-09-28 -- order 28, rank 3. The dash was removed
        # from the class. It merged rather than deleted -- 6203-2RS
        # became 62032RS and CROUSE-HINDS became CROUSEHINDS -- and the
        # merged form is rarer, so it was likelier to fall below
        # min_token_count and vanish entirely.
        r"[\./()<>#&+\"!@$%^\*]",
        # instances of #X#
        r"\d+X\d+",
        # numbers preceded by space and followed by space or end of text
        r"(?<=\s)\d+(?=\s|$)",
        # REMOVED 2026-09-28 -- order 31, rank 3.
        # Tooth counts (24T bandsaw) and motor frame sizes (184T).
        # Low cardinality and category-bearing -- closer to a class
        # than a measurement.
        #   r"(?<=\s)\d+T(?=\s|$)",
        # ADDED 2026-09-28 -- cleanup for the fragments left behind by
        # removing order 4.
        #
        # That rule used to consume a whole dimension run in one go.
        # Without it the narrower rules chew the run from the middle
        # out and leave tails: "1/2X1X1/4IN" -> "XIN", "1-1/4X3-1/2"
        # -> "-".
        #
        # This deletes a token made ONLY of separators and unit
        # letters, and only when it contains at least one separator.
        # The lookahead is what protects the units themselves: "IN",
        # "FT" and "LB" have no separator and survive, which is the
        # whole point of having removed orders 20-22. Anything with a
        # digit or another letter -- 6203-2RS, CROUSE-HINDS, TIALN,
        # FT-LB, MIL-SPEC -- is left alone because the group cannot
        # cover the whole token.
        rf"(?<=\s)(?=\S*[X\-/.])(?:[X\-/.]|{spec_units_pattern})+(?=\s|$)",
        # stray instances of "X"
        r"(?<=\s)X(?=\s|$)",
        # stray instances of "SZ", or "SZ" followed by numbers
        rf"SZ(?=\s|$)|SZ\d\S*|\d+SZ\S*|SZ[./\-:;]\S*|SZ(?:{sizes_pattern})\b",
    ]

    replacement_2 = {
        # replace multiple consecutive spaces with a single space
        r"\s{2,}": " ",
    }

    return {
        "deletion_1": deletion_1,
        "replacement_1": replacement_1,
        "deletion_2": deletion_2,
        "replacement_2": replacement_2,
    }


def apply_patterns(series: pd.Series, patterns: Dict[str, object]) -> pd.Series:
    """The two-phase pass on a bare Series.

    Training and scoring both come through here, which is the point: the
    vectorizer's vocabulary was fitted on the output of these patterns, so
    a second implementation for scoring would be a second chance to drift.
    """
    out = series.astype(str)
    for pattern in patterns["deletion_1"]:
        out = out.str.replace(pattern, "", regex=True)
    for pattern, replacement in patterns["replacement_1"].items():
        out = out.str.replace(pattern, replacement, regex=True)
    for pattern in patterns["deletion_2"]:
        out = out.str.replace(pattern, "", regex=True)
    for pattern, replacement in patterns["replacement_2"].items():
        out = out.str.replace(pattern, replacement, regex=True)
    return out


def build_expansion_pattern(expansions: Dict[str, str]) -> Optional[re.Pattern]:
    """One alternation matching any abbreviation, longest first.

    Longest first for the same reason sort_by_length exists: without it a
    short key can match inside a longer one and the longer entry never
    fires.
    """
    if not expansions:
        return None
    keys = sorted(expansions, key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b")


def expand_abbreviations(series: pd.Series, expansions: Dict[str, str]) -> pd.Series:
    """Replace whole-word abbreviations with their expansions.

    One pass, not one pass per entry. Replacing in sequence would let an
    expansion be re-matched by a later abbreviation -- expand SS to
    STAINLESS STEEL, then have a separate ST entry rewrite part of it. A
    single alternation consumes each match once and moves on, so the
    output depends only on the file, not on the order of its rows.

    Runs last in the procedure, on text that has already had its
    punctuation removed, so \\b sits between plain alphanumerics.
    """
    pattern = build_expansion_pattern(expansions)
    if pattern is None:
        return series
    return series.astype(str).str.replace(
        pattern, lambda m: expansions[m.group(0)], regex=True
    )


def normalize_misspellings(series: pd.Series, normalizations: Dict[str, str]) -> pd.Series:
    """Fold misspellings and spelling variants onto one form.

    Same single-alternation trick as expand_abbreviations, and for the
    same reason: replacing entry by entry would let one correction be
    re-matched by a later one, making the result depend on row order in
    the CSV rather than on its contents.

    Runs *before* the regex pass, which is the opposite of abbreviation
    expansion. A misspelling has to be fixed while the word is still
    whole, because everything downstream -- the regex lists, the
    vocabulary filter, the vectorizer -- matches on real words.
    """
    pattern = build_expansion_pattern(normalizations)
    if pattern is None:
        return series
    return series.astype(str).str.replace(
        pattern, lambda m: normalizations[m.group(0)], regex=True
    )


def filter_known_tokens(
    series: pd.Series,
    known_tokens: set,
    min_token_length: int = 2,
) -> pd.Series:
    """Keep only tokens the catalogue recognises.

    'Recognised' is the union of three things, and all three are needed:
    the English word list, the designated good tokens, and (optionally)
    the abbreviation keys. Drop any one of them and the filter starts
    deleting signal -- English alone throws away COROMILL and NPT, and
    without the good tokens there is nothing to put them back.

    Lossy by design, so it is off unless asked for. Tokens shorter than
    min_token_length go regardless: a one- or two-character survivor of
    the regex pass is almost always a fragment.
    """
    if not known_tokens:
        return series

    def _filter(text: str) -> str:
        kept = [
            tok for tok in str(text).split()
            if len(tok) >= min_token_length and tok.upper() in known_tokens
        ]
        return " ".join(kept)

    return series.astype(str).map(_filter)


def build_known_tokens(
    vocabulary: Optional[set] = None,
    good_tokens: Optional[set] = None,
    expansions: Optional[Dict[str, str]] = None,
    keep_abbrev_keys: bool = True,
) -> set:
    """The uppercase set that filter_known_tokens matches against."""
    known: set = set()
    if vocabulary:
        known |= {w.upper() for w in vocabulary}
    if good_tokens is None:
        good_tokens = load_good_tokens()
    known |= {t.upper() for t in good_tokens}
    if keep_abbrev_keys:
        if expansions is None:
            expansions = load_expansions()
        known |= {k.upper() for k in expansions}
        # The expansions themselves are legitimate output of the cleaner,
        # so their words have to survive a filter applied afterwards.
        for value in expansions.values():
            known |= {part.upper() for part in str(value).split()}
    return known


def attach_vendor_token(
    df: pd.DataFrame,
    text_col: str,
    vendor_col: str,
    prefix: str = "VND_",
) -> pd.DataFrame:
    """Prepend a vendor marker to each description.

    Vendor identity is real evidence -- a supplier who only sells
    abrasives narrows the category enormously -- but it is evidence the
    description does not carry. Folding it in as a token lets the same
    vectorizer and the same model use it with no special casing.

    Pair this with count or binary vectorization. Under TF-IDF a marker
    that appears on every one of a large vendor's rows gets a low IDF
    weight, which suppresses exactly the vendors there is most data for.
    """
    out = df.copy()
    if vendor_col not in out.columns:
        raise KeyError(
            f"attach_vendor requires column {vendor_col!r}; found "
            f"{list(out.columns)}"
        )
    vendor = (
        out[vendor_col].fillna("").astype(str).str.strip()
        .str.replace(r"\s+", "", regex=True)
    )
    marker = prefix + vendor
    marker = marker.where(vendor != "", "")
    out[text_col] = (marker + " " + out[text_col].fillna("").astype(str)).str.strip()
    return out


def regex_cleaning_proc(
    df: pd.DataFrame,
    column: str,
    lists: Optional[Dict[str, List[str]]] = None,
    patterns: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    """The two-phase pass: compounds, then reference terms and punctuation."""
    if patterns is None:
        if lists is None:
            lists = {k: upper(v) for k, v in load_cleaning_lists().items()}
        patterns = build_patterns(lists)

    out = df.copy()
    out[column] = apply_patterns(out[column], patterns)
    return out


# ---------------------------------------------------------------------------
# The recipe as one object
# ---------------------------------------------------------------------------

@dataclass
class DescriptionCleaner:
    """The whole cleaning recipe, in one picklable object.

    Exists so that training and scoring cannot use different recipes. The
    vectorizer's vocabulary is fitted on whatever comes out of here, so a
    scoring path that cleans even slightly differently sees mostly
    out-of-vocabulary words -- and fails silently, as a raised OOV rate
    rather than an error.

    Save it with the model (artifacts.save_bundle takes cleaning_cfg=) and
    load it back at scoring time. That includes the English vocabulary, so
    scoring needs no NLTK download and cannot pick up a different word
    list than training used.
    """

    lists: Dict[str, List[str]]
    vocabulary: set = field(default_factory=set)
    strip_leading_code: bool = True
    uppercase: bool = False
    patterns: Optional[Dict[str, object]] = None
    # Abbreviation expansion, applied after everything else. The mapping
    # travels with the cleaner so scoring expands exactly as training did,
    # even if the CSV on disk changes afterwards.
    expand_abbrevs: bool = False
    expansions: Dict[str, str] = field(default_factory=dict)
    # Misspelling normalisation, applied before the regex pass.
    normalize_misspellings: bool = False
    normalizations: Dict[str, str] = field(default_factory=dict)
    # Vocabulary filter, applied last. Lossy, so off unless asked for.
    filter_to_known_tokens: bool = False
    known_tokens: set = field(default_factory=set)
    min_token_length: int = 2

    def __post_init__(self):
        if self.patterns is None:
            self.patterns = build_patterns(self.lists)

    def clean_series(self, series: pd.Series) -> pd.Series:
        """Clean text without dropping anything.

        Row-preserving on purpose: scoring needs one output per input, and
        a description reduced to nothing is a decision for the readability
        gate rather than something to silently discard here.

        The order of the six steps is load-bearing. Normalisation has to
        run before the regex pass, while words are still whole;
        expansion has to run after it, once punctuation is gone; and the
        vocabulary filter has to run after both, or it deletes the
        misspellings before they are corrected and the abbreviations
        before they are expanded.
        """
        out = series.fillna("").astype(str)
        if self.uppercase:
            out = out.str.upper()
        if self.strip_leading_code:
            out = out.map(self._strip_leading)
        if self.normalize_misspellings and self.normalizations:
            out = normalize_misspellings(out, self.normalizations)
        out = apply_patterns(out, self.patterns)
        out = out.str.strip()
        if self.expand_abbrevs and self.expansions:
            out = expand_abbreviations(out, self.expansions)
        if self.filter_to_known_tokens and self.known_tokens:
            out = filter_known_tokens(out, self.known_tokens, self.min_token_length)
        return out.str.strip()

    def _strip_leading(self, text: str) -> str:
        if first_word_is_real(text, self.vocabulary):
            return text
        parts = text.split(maxsplit=1)
        return parts[1] if len(parts) > 1 else ""

    def __call__(self, series: pd.Series) -> pd.Series:
        return self.clean_series(series)

    def describe(self) -> str:
        return (
            f"DescriptionCleaner(lists={sorted(self.lists)}, "
            f"vocabulary={len(self.vocabulary):,} words, "
            f"strip_leading_code={self.strip_leading_code}, "
            f"uppercase={self.uppercase}, "
            f"expand_abbrevs={self.expand_abbrevs}"
            + (f" ({len(self.expansions):,} pairs)" if self.expand_abbrevs else "")
            + f", normalize_misspellings={self.normalize_misspellings}"
            + (f" ({len(self.normalizations):,} pairs)"
               if self.normalize_misspellings else "")
            + f", filter_to_known_tokens={self.filter_to_known_tokens}"
            + (f" ({len(self.known_tokens):,} tokens)"
               if self.filter_to_known_tokens else "")
            + ")"
        )


def build_cleaner(
    lists: Optional[Dict[str, List[str]]] = None,
    vocabulary: Optional[set] = None,
    strip_leading_code: bool = True,
    uppercase: bool = False,
    expand_abbrevs: bool = False,
    expansions: Optional[Dict[str, str]] = None,
    normalize_misspellings: bool = False,
    normalizations: Optional[Dict[str, str]] = None,
    filter_to_known_tokens: bool = False,
    good_tokens: Optional[set] = None,
    keep_abbrev_keys: bool = True,
    min_token_length: int = 2,
) -> DescriptionCleaner:
    """A DescriptionCleaner with the packaged lists and NLTK vocabulary.

    Every reference list is resolved here and stored on the object, so
    the cleaner that gets pickled alongside a model is self-contained.
    Scoring then needs no NLTK download and cannot pick up a different
    word list, good-token file or abbreviation CSV than training used --
    which is the failure this design exists to prevent, because it shows
    up as a raised out-of-vocabulary rate rather than as an error.
    """
    if lists is None:
        lists = {k: upper(v) for k, v in load_cleaning_lists().items()}
    if vocabulary is None and (strip_leading_code or filter_to_known_tokens):
        vocabulary = load_english_vocabulary()
    if expansions is None:
        needs_expansions = expand_abbrevs or (filter_to_known_tokens and keep_abbrev_keys)
        expansions = load_expansions(required=expand_abbrevs) if needs_expansions else {}
    if normalizations is None:
        normalizations = (
            load_normalizations(required=True) if normalize_misspellings else {}
        )

    known_tokens: set = set()
    if filter_to_known_tokens:
        known_tokens = build_known_tokens(
            vocabulary=vocabulary,
            good_tokens=good_tokens,
            expansions=expansions,
            keep_abbrev_keys=keep_abbrev_keys,
        )

    return DescriptionCleaner(
        lists=lists,
        vocabulary=vocabulary or set(),
        strip_leading_code=strip_leading_code,
        uppercase=uppercase,
        expand_abbrevs=expand_abbrevs,
        expansions=expansions,
        normalize_misspellings=normalize_misspellings,
        normalizations=normalizations,
        filter_to_known_tokens=filter_to_known_tokens,
        known_tokens=known_tokens,
        min_token_length=min_token_length,
    )


def cleaner_from_config(cfg, **overrides) -> DescriptionCleaner:
    """Build a cleaner from a config.CleaningConfig.

    Vendor attachment is deliberately *not* part of the cleaner: the
    marker has to be prepended after the regex pass and the vocabulary
    filter, both of which would otherwise shred it. See
    clean_descriptions(), which sequences the two correctly.
    """
    kwargs = dict(
        strip_leading_code=cfg.strip_leading_code,
        uppercase=cfg.uppercase,
        expand_abbrevs=cfg.expand_abbrevs,
        normalize_misspellings=cfg.normalize_misspellings,
        filter_to_known_tokens=cfg.filter_to_known_tokens,
        keep_abbrev_keys=cfg.keep_abbrev_keys,
        min_token_length=cfg.min_token_length,
    )
    kwargs.update(overrides)
    return build_cleaner(**kwargs)


# ---------------------------------------------------------------------------
# The whole procedure
# ---------------------------------------------------------------------------

def attach_marker_tokens(
    df: pd.DataFrame,
    text_col: str,
    markers: Dict[str, Dict[str, object]],
    verbose: bool = True,
) -> pd.DataFrame:
    """Prepend one or more categorical codes to each description.

    The generalisation of `attach_vendor_token`. `markers` maps a
    source column to how it should be emitted:

        {"ICSP Prodcat": {"prefix": "CAT_"},
         "UNSPSC":       {"prefix": "UNSPSC", "levels": (2, 4, 6, 8)},
         "Prodline":     {"prefix": "PLINE_"}}

    `levels` is for hierarchical codes. UNSPSC is eight digits nesting
    segment / family / class / commodity, so emitting only the full
    code would give the model one all-or-nothing token that is useless
    the moment it meets an unseen commodity. Emitting the prefixes as
    well -- `UNSPSC2_46`, `UNSPSC4_4618`, `UNSPSC6_461815`,
    `UNSPSC8_46181504` -- lets it fall back to the segment when the
    commodity is new, which is exactly the `unseen in training` case
    that carries 74% of the remaining error.

    Each level gets its own prefix so a four-digit code cannot collide
    with the first four digits of an eight-digit one.

    Values are stripped and de-spaced, and a blank produces no token at
    all rather than a bare prefix -- a bare `CAT_` on every row with a
    missing code would be a feature meaning "this field is empty",
    which is occasionally useful and more often just noise.
    """
    out = df.copy()
    pieces = []

    for col, spec in (markers or {}).items():
        if col not in out.columns:
            raise KeyError(
                f"marker column {col!r} is not in the frame; found "
                f"{list(out.columns)}"
            )
        prefix = str(spec.get("prefix", ""))
        levels = spec.get("levels")

        value = (
            out[col].fillna("").astype(str).str.strip()
            .str.replace(r"\s+", "", regex=True)
        )
        if levels:
            for n in levels:
                trimmed = value.str.slice(0, int(n))
                # Only where the code is actually that long; a 4-digit
                # value must not masquerade as an 8-digit prefix.
                token = (f"{prefix}{n}_" + trimmed).where(
                    value.str.len() >= int(n), "")
                pieces.append(token)
        else:
            pieces.append((prefix + value).where(value != "", ""))

        if verbose:
            present = (value != "").mean()
            print(f"[clean] marker {col!r}: present on {present:.1%} of rows, "
                  f"{value[value != ''].nunique():,} distinct")

    if not pieces:
        return out

    joined = pieces[0]
    for part in pieces[1:]:
        joined = (joined + " " + part).str.strip()
    out[text_col] = (
        (joined + " " + out[text_col].fillna("").astype(str))
        .str.replace(r"\s+", " ", regex=True).str.strip()
    )
    return out


def clean_descriptions_with(
    df: pd.DataFrame,
    cleaner: DescriptionCleaner,
    column: str = "FullDesc",
    orig_col: Optional[str] = DEFAULT_ORIG_COL,
    vendor_col: Optional[str] = None,
    vendor_prefix: str = "VND_",
    markers: Optional[Dict[str, Dict[str, object]]] = None,
    drop_blank: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run a prepared cleaner over a frame, then attach any markers.

    The one function both training and scoring go through. Scoring calls
    it with drop_blank=False, because a description reduced to nothing
    is a row that still needs an answer (or an explicit refusal) rather
    than a row that quietly disappears.
    """
    out = df.copy()
    n_before = len(out)

    if orig_col:
        out[orig_col] = out[column]

    out[column] = cleaner.clean_series(out[column])

    # After cleaning, never before: the regex pass strips digits and the
    # vocabulary filter does not know the marker, so either one would
    # destroy it.
    if vendor_col:
        out = attach_vendor_token(out, column, vendor_col, vendor_prefix)
    if markers:
        out = attach_marker_tokens(out, column, markers, verbose=verbose)

    if drop_blank:
        out = out[out[column].astype(str).str.strip() != ""].reset_index(drop=True)
        if verbose and len(out) < n_before:
            print(f"[clean] dropped {n_before - len(out):,} rows left blank by cleaning")
    return out


def clean_descriptions(
    df: pd.DataFrame,
    column: str = "FullDesc",
    lists: Optional[Dict[str, List[str]]] = None,
    vocabulary: Optional[set] = None,
    orig_col: Optional[str] = DEFAULT_ORIG_COL,
    strip_leading_code: bool = True,
    uppercase: bool = False,
    expand_abbrevs: bool = False,
    expansions: Optional[Dict[str, str]] = None,
    normalize_misspellings_flag: bool = False,
    normalizations: Optional[Dict[str, str]] = None,
    filter_to_known_tokens: bool = False,
    good_tokens: Optional[set] = None,
    keep_abbrev_keys: bool = True,
    min_token_length: int = 2,
    vendor_col: Optional[str] = None,
    vendor_prefix: str = "VND_",
    verbose: bool = True,
) -> pd.DataFrame:
    """Preserve the original, drop blanks, strip leading codes, run the regex.

    Returns a new frame. Row counts are reported at each step because the
    steps drop rows, and a total that shifts without explanation is the
    thing most likely to go unnoticed here.

    uppercase is off to match the original procedure, which assumes the
    ERP descriptions arrive uppercase. Switch it on if that stops being
    true -- the patterns and reference lists are uppercase, so lowercase
    input would slip past almost all of them.
    """
    out = df.copy()
    steps = [("loaded", len(out))]

    if lists is None:
        lists = {k: upper(v) for k, v in load_cleaning_lists().items()}

    if orig_col:
        out[orig_col] = out[column]

    if uppercase:
        out[column] = out[column].astype(str).str.upper()

    out = out.dropna(subset=[column])
    out = out[out[column].astype(str).str.strip() != ""]
    steps.append(("after dropping null/blank descriptions", len(out)))

    if strip_leading_code:
        if vocabulary is None:
            vocabulary = load_english_vocabulary()
        out = add_first_word_real(out, column, vocabulary)
        n_stripped = int((out[FIRST_WORD_COL] == 0).sum())
        out = strip_first_term(out, column)
        steps.append((f"after stripping {n_stripped:,} leading codes", len(out)))

    # Before the regex pass, while the words are still whole.
    if normalize_misspellings_flag:
        if normalizations is None:
            normalizations = load_normalizations(required=True)
        out[column] = normalize_misspellings(out[column], normalizations)
        steps.append((f"after normalising {len(normalizations):,} spelling(s)", len(out)))

    out = regex_cleaning_proc(out, column, lists)
    out[column] = out[column].astype(str).str.strip()

    out = out[out[column] != ""]
    steps.append(("after the regex pass, blanks removed", len(out)))

    # Substitutes rather than deletes, so it cannot empty a row and the
    # count below cannot fall.
    if expand_abbrevs:
        if expansions is None:
            expansions = load_expansions(required=True)
        out[column] = expand_abbreviations(out[column], expansions)
        steps.append((f"after expanding {len(expansions):,} abbreviation(s)", len(out)))

    # The one step that can empty a row, so it reports its own losses.
    if filter_to_known_tokens:
        if vocabulary is None:
            vocabulary = load_english_vocabulary()
        if expansions is None and keep_abbrev_keys:
            expansions = load_expansions()
        known = build_known_tokens(
            vocabulary=vocabulary,
            good_tokens=good_tokens,
            expansions=expansions,
            keep_abbrev_keys=keep_abbrev_keys,
        )
        out[column] = filter_known_tokens(out[column], known, min_token_length)
        out = out[out[column].astype(str).str.strip() != ""]
        steps.append(
            (f"after filtering to {len(known):,} known token(s), blanks removed",
             len(out))
        )

    # Last of all: the marker has to survive everything above it.
    if vendor_col:
        out = attach_vendor_token(out, column, vendor_col, vendor_prefix)
        steps.append(("after attaching the vendor marker", len(out)))

    out = out.reset_index(drop=True)
    if verbose:
        for label, n in steps:
            print(f"[clean] {n:>9,}  {label}")
        dropped = steps[0][1] - steps[-1][1]
        print(f"[clean] {dropped:>9,}  rows dropped in total "
              f"({dropped / steps[0][1]:.2%})" if steps[0][1] else "")
    out.attrs["steps"] = steps
    return out
