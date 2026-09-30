"""Regex-based cleaning of free-text product descriptions.

This step is optional. If you already have a cleaned extract, point
DataConfig.path at it and leave PipelineConfig.clean_text = False.

The transforms are deliberately small and composable so that a cleaning
recipe is data, not code: see CleaningConfig below.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import pandas as pd

from .lists import load_cleaning_lists, sort_by_length, upper


# ---------------------------------------------------------------------------
# Primitive transforms
# ---------------------------------------------------------------------------

def regex_deletion(series: pd.Series, patterns: Sequence[str]) -> pd.Series:
    """Delete every match of every pattern."""
    for pattern in patterns:
        series = series.str.replace(pattern, "", regex=True)
    return series


def regex_replacement(series: pd.Series, pattern_replacements: Dict[str, str]) -> pd.Series:
    """Apply {pattern: replacement} substitutions in dict order."""
    for pattern, replacement in pattern_replacements.items():
        series = series.str.replace(pattern, replacement, regex=True)
    return series


def word_boundary_pattern(terms: Sequence[str]) -> str:
    """Build a single alternation matching any term as a whole word.

    Terms are sorted longest-first and regex-escaped, so 'EXTRA LARGE'
    wins over 'LARGE' and '2XL' is not read as a quantifier.
    """
    if not terms:
        return r"(?!x)x"  # never matches
    alternatives = "|".join(re.escape(t) for t in sort_by_length(list(terms)))
    return rf"\b(?:{alternatives})\b"


def collapse_whitespace(series: pd.Series) -> pd.Series:
    return series.str.replace(r"\s+", " ", regex=True).str.strip()


# ---------------------------------------------------------------------------
# Recipe
# ---------------------------------------------------------------------------

@dataclass
class CleaningConfig:
    """Which cleaning steps to run, in order.

    remove_lists names keys from the cleaning-list directory whose terms
    should be stripped out of the description entirely (colours and sizes
    are usually noise for category prediction; units often are not, since
    'GAL' is a strong signal for a container category — hence the default).
    """

    list_dir: Optional[str] = None
    uppercase: bool = True
    remove_lists: List[str] = field(
        default_factory=lambda: ["colors", "color_abbrevs", "sizes"]
    )
    # Applied before list removal.
    delete_patterns: List[str] = field(
        default_factory=lambda: [
            r"\bhttps?://\S+\b",  # urls
            r"[^\w\s/\-\.]",      # punctuation except / - .
        ]
    )
    # Applied after list removal, e.g. normalising plurals to singulars.
    replace_patterns: Dict[str, str] = field(
        default_factory=lambda: {
            r"(?<=\d)\s*(?:X|BY)\s*(?=\d)": "X",  # 4 X 6 -> 4X6
            r"\b(\d+)\s*(IN|FT|MM|CM)\b": r"\1\2",  # 4 IN -> 4IN
        }
    )
    drop_digits: bool = False
    min_token_length: int = 1


def clean_series(series: pd.Series, cfg: Optional[CleaningConfig] = None) -> pd.Series:
    """Run the full cleaning recipe over one text column."""
    cfg = cfg or CleaningConfig()
    lists = load_cleaning_lists(cfg.list_dir)

    out = series.fillna("").astype(str)
    if cfg.uppercase:
        out = out.str.upper()

    out = regex_deletion(out, cfg.delete_patterns)

    for key in cfg.remove_lists:
        if key not in lists:
            raise KeyError(
                f"Cleaning list {key!r} not found. Available: {sorted(lists)}"
            )
        terms = upper(lists[key]) if cfg.uppercase else lists[key]
        out = out.str.replace(word_boundary_pattern(terms), " ", regex=True)

    out = regex_replacement(out, cfg.replace_patterns)

    if cfg.drop_digits:
        out = out.str.replace(r"\b\d+\b", " ", regex=True)

    out = collapse_whitespace(out)

    if cfg.min_token_length > 1:
        n = cfg.min_token_length
        out = out.str.replace(rf"\b\w{{1,{n - 1}}}\b", " ", regex=True)
        out = collapse_whitespace(out)

    return out


def clean_dataframe(
    df: pd.DataFrame,
    text_col: str,
    cfg: Optional[CleaningConfig] = None,
    out_col: Optional[str] = None,
) -> pd.DataFrame:
    """Return a copy of df with the cleaned text in `out_col` (default: in place)."""
    df = df.copy()
    df[out_col or text_col] = clean_series(df[text_col], cfg)
    return df
