"""Loading, filtering, and splitting the golden set.

The same import path for all three model families, which is the whole
reason this module is separate from any of them.
"""

from __future__ import annotations

import pathlib
from typing import List, Optional, Sequence, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from . import paths
from .config import DataConfig, FilterConfig, SplitConfig
from .cleaning.tokens import drop_rare_classes, drop_rare_paths


# The pull that produces raw_data.txt. Kept here rather than only in the
# notebook so the schema the rest of the package assumes is written down
# in one place and can be diffed when the source tables change.
HIERARCHY_SQL = """
with 
    base as (
        select s.id as `ID`, 
               s.erp_cono as `Cono`, 
               s.erp_product_number as `Vallen ID`, 
               replace(i.descrip, ';', ' ') as `FullDesc`,
               s.category_l4_id, 
               s.global_vendor_number as `Global Vendno`, 
               i.prodcat as `ICSP Prodcat`, 
               i.zzunspsc as `UNSPSC`,  
               concat(s.erp_cono, w.prodline) as `Prodline`
        from prd_bronze.stibo_ftp.sku s 
        inner join prd_bronze.openedge.icsp i 
        on s.erp_product_number = i.prod 
        left join prd_bronze.openedge.icsw w 
        on s.erp_product_number = w.prod 
           and w.st_row_current=1 
           and w.cono=1 
           and w.whse = 'zdat'
        where s.st_row_current=1 
            and s.category_l4_id is not null 
            and i.st_row_current=1 
            and i.cono in (1,55) 
            and i.descrip is not null
    ), 

    hierarchy as (
        SELECT
            l1.id   AS `Level 1 ID`,
            l1.name AS `Level 1 Name`,
            l2.id   AS `Level 2 ID`,
            l2.name AS `Level 2 Name`,
            l3.id   AS `Level 3 ID`,
            l3.name AS `Level 3 Name`,
            l4.id   AS `Level 4 ID`,
            l4.name AS `Level 4 Name`
        FROM prd_bronze.stibo_ftp.hierarchy l1
        LEFT JOIN prd_bronze.stibo_ftp.hierarchy l2
            ON l2.parent_id = l1.id
        AND l2.level = 2
        AND l2.st_row_current = 1
        LEFT JOIN prd_bronze.stibo_ftp.hierarchy l3
            ON l3.parent_id = l2.id
        AND l3.level = 3
        AND l3.st_row_current = 1
        LEFT JOIN prd_bronze.stibo_ftp.hierarchy l4
            ON l4.parent_id = l3.id
        AND l4.level = 4
        AND l4.st_row_current = 1
        WHERE l1.level = 1
        AND l1.st_row_current = 1
    )

select *
from base b
inner join hierarchy h
on b.category_l4_id = h.`Level 4 ID` 
where h.`Level 4 ID` is not null
"""


# Strings that mean "no value" after a frame has been stringified.
#
# run_pull() calls .astype("str") on purpose -- category IDs like
# '00123' lose their leading zeros the moment anything guesses a
# numeric dtype. The cost is that SQL NULL becomes the literal 'None'
# and NaN becomes 'nan', and every downstream check for emptiness
# then passes: 'None' != '' is True, so a null vendor produced a
# VND_None token that looked like a real supplier.
NULL_STRINGS = frozenset({
    "", "none", "nan", "nat", "null", "<na>", "na", "n/a", "#n/a",
})


def blank_out_nulls(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Turn stringified nulls back into empty strings.

    Matched case-insensitively against NULL_STRINGS after stripping.
    A genuine product code of 'NONE' or 'NA' would be caught too --
    vanishingly unlikely, and the count is reported so it is visible
    rather than silent.
    """
    out = df.copy()
    cols = list(columns) if columns is not None else list(out.columns)
    changed = {}
    for col in cols:
        if col not in out.columns:
            continue
        values = out[col].astype(str)
        mask = values.str.strip().str.lower().isin(NULL_STRINGS)
        n = int(mask.sum())
        if n:
            out.loc[mask, col] = ""
            changed[col] = n
    if verbose and changed:
        print("[data] blanked stringified nulls:")
        for col, n in sorted(changed.items(), key=lambda kv: -kv[1]):
            print(f"[data]   {col}: {n:,} rows ({n / max(len(out), 1):.1%})")
    return out


def run_pull(spark, sql: str = HIERARCHY_SQL) -> pd.DataFrame:
    """Execute the pull and bring it back as an all-string pandas frame.

    All-string on purpose: category IDs like '00123' lose their leading
    zeros the moment anything guesses a numeric dtype, and the resulting
    key mismatch is silent.
    """
    if spark is None:
        raise ValueError("run_pull needs a spark session; pass spark=spark.")
    raw = spark.sql(sql).toPandas().astype("str")
    # .astype("str") stringifies NULL to 'None'. Undo that here, once,
    # rather than leaving every consumer to recognise it.
    return blank_out_nulls(raw)


def save_table(df: pd.DataFrame, filename: str, delimiter: str = "|") -> str:
    """Write a frame to the data directory, creating it if needed."""
    target = paths.data_dir() / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, sep=delimiter, index=False)
    print(f"[data] wrote {len(df):,} rows to {target}")
    return str(target)


def load_data(cfg: DataConfig, spark=None, require_labels: bool = True) -> pd.DataFrame:
    """Read a product file as all-string columns.

    require_labels=False skips the check for the hierarchy columns,
    which is what scoring needs: unlabelled records are the whole point
    there, and demanding the columns the model exists to create would
    refuse exactly the input it exists to handle.
    """
    if cfg.file_format == "csv":
        df = pd.read_csv(cfg.path, delimiter=cfg.delimiter, dtype="str")
    elif cfg.file_format == "excel":
        df = pd.read_excel(cfg.path, dtype="str")
    elif cfg.file_format in ("delta", "spark_table"):
        if spark is None:
            raise ValueError(
                f"file_format={cfg.file_format!r} requires a spark session; "
                "pass spark=spark from the notebook."
            )
        sdf = (
            spark.read.format("delta").load(cfg.path)
            if cfg.file_format == "delta"
            else spark.table(cfg.path)
        )
        df = sdf.toPandas().astype("str")
    else:
        raise ValueError(f"Unsupported file_format: {cfg.file_format!r}")

    validate_columns(df, cfg, require_labels=require_labels)
    return df


def validate_columns(
    df: pd.DataFrame, cfg: DataConfig, require_labels: bool = True
) -> None:
    """Fail early and loudly on a schema mismatch."""
    expected = [cfg.text_col]
    if require_labels:
        expected += list(cfg.level_columns)
    if cfg.id_col:
        expected.append(cfg.id_col)
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise KeyError(
            f"Columns missing from the input data: {missing}. "
            f"Found: {list(df.columns)}"
        )


def drop_null_rows(
    df: pd.DataFrame,
    text_col: str,
    level_columns: Sequence[str],
    drop_null_text: bool = True,
    drop_null_labels: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Drop rows that cannot be used for training or scoring."""
    n_before = len(df)
    if drop_null_text:
        df = df[df[text_col].notna() & (df[text_col].astype(str).str.strip() != "")]
    if drop_null_labels:
        df = df.dropna(subset=list(level_columns))
        for col in level_columns:
            df = df[~df[col].astype(str).str.strip().isin(("", "nan", "None"))]
    if verbose and len(df) < n_before:
        print(f"[data] dropped {n_before - len(df):,} rows with null text/labels")
    return df.reset_index(drop=True)


def build_hierarchy_lookup(
    df: pd.DataFrame,
    level_columns: Sequence[str],
    name_columns: Sequence[str],
) -> pd.DataFrame:
    """The ID -> Name table, one row per distinct path.

    Training happens on IDs because names are not unique across the
    tree. This is what puts the names back on a prediction, and it is
    saved with the model so that a later rename upstream cannot
    retroactively change what an old run appears to have said.
    """
    cols = [c for c in list(level_columns) + list(name_columns) if c in df.columns]
    return df[cols].drop_duplicates().reset_index(drop=True)


def split_data(
    df: pd.DataFrame,
    level_columns: Sequence[str],
    cfg: SplitConfig,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/validation split on the deepest active level.

    Stratifying on the deepest level implicitly stratifies the
    shallower ones, since the hierarchy is a tree.
    """
    level_columns = list(level_columns)
    stratify = df[level_columns[-1]] if cfg.stratify else None
    if stratify is not None:
        singles = stratify.value_counts()
        singles = singles[singles < 2]
        if len(singles):
            raise ValueError(
                f"{len(singles)} classes at '{level_columns[-1]}' have a single "
                "member, so a stratified split is impossible. Raise "
                "FilterConfig.min_class_count (>= 2) or set stratify=False."
            )
    train_df, val_df = train_test_split(
        df,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=stratify,
    )
    if verbose:
        print(f"[data] train={len(train_df):,} rows, validation={len(val_df):,} rows")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def load_id_list(
    path: str,
    column_name: Optional[str] = None,
    verbose: bool = True,
) -> set:
    """Read a list of IDs from a one- or many-column delimited file.

    Strings, always. An ID like '00123' loses its leading zeros the
    moment anything guesses a numeric dtype, and the resulting
    mismatch is silent -- the row is simply not found and nothing is
    filtered.

    Handles more shapes than the name suggests, because every one of
    them has turned up:

      * one value per line, with or without a header;
      * a real CSV with several columns -- the ID column is found by
        matching `column_name` (or a bare 'id') in the header, and
        falls back to the first column with a warning;
      * comma, tab or pipe delimited, detected from the first line;
      * quoted fields, so a description containing a comma does not
        shift the columns;
      * a BOM, blank lines, stray quotes and duplicates.

    An earlier version read whole lines. Given a three-column export
    it produced IDs like '00123,WIDGET,4120', matched nothing, and
    filtered nothing -- which is why `drop_ids` shouts when no listed
    ID matches.
    """
    import csv

    raw = pathlib.Path(path).read_text(encoding="utf-8-sig").splitlines()
    lines = [line for line in raw if line.strip()]
    if not lines:
        if verbose:
            print(f"[blacklist] {path} is empty")
        return set()

    delimiter = max(",\t|", key=lambda d: lines[0].count(d))
    if lines[0].count(delimiter) == 0:
        delimiter = ","                      # single column; harmless

    rows = [r for r in csv.reader(lines, delimiter=delimiter) if any(f.strip() for f in r)]
    n_cols = max(len(r) for r in rows)

    # Which column holds the ID?
    wanted = (column_name or "id").strip().lower()
    header, index = None, 0
    first = [f.strip().strip('"').strip("'").lower() for f in rows[0]]
    looks_like_header = any(
        f in {wanted, "id", "ids", "item", "item id", "sku", "vallen id"}
        for f in first
    )
    if looks_like_header:
        header = rows[0]
        match = [i for i, f in enumerate(first) if f == wanted]
        if not match:
            match = [i for i, f in enumerate(first)
                     if f in {"id", "ids", "item", "item id", "sku", "vallen id"}]
        index = match[0]
        rows = rows[1:]

    values = [r[index].strip().strip('"').strip("'") for r in rows if len(r) > index]
    values = [v for v in values if v]
    ids = set(values)

    if verbose:
        print(f"[blacklist] {len(ids):,} distinct IDs from {path}")
        if n_cols > 1:
            where = (f"column {index} ({header[index].strip()!r})" if header
                     else f"column {index} (no header found -- using the first)")
            print(f"[blacklist]   {n_cols} columns, delimiter {delimiter!r}; "
                  f"took {where}")
            if not header:
                print("[blacklist]   >> No header row recognised. If the IDs "
                      "are not in the first column, add a header naming it.")
        elif header:
            print(f"[blacklist]   dropped header line {header[0]!r}")
        if len(values) != len(ids):
            print(f"[blacklist]   {len(values) - len(ids):,} duplicate lines collapsed")
    return ids


def drop_ids(
    df: pd.DataFrame,
    ids: set,
    id_col: str = "ID",
    label: str = "rows",
    verbose: bool = True,
) -> pd.DataFrame:
    """Remove rows whose `id_col` is in `ids`.

    Reports how many IDs in the list matched nothing. That number is
    the one worth reading: a blacklist that matches zero rows is
    usually a dtype or whitespace problem rather than a list of IDs
    that happen to be absent, and it fails silently otherwise.
    """
    if id_col not in df.columns:
        raise KeyError(
            f"id_col={id_col!r} is not in the frame. Columns: "
            f"{list(df.columns)}"
        )
    if not ids:
        if verbose:
            print("[blacklist] empty list; nothing removed")
        return df

    key = df[id_col].astype(str).str.strip()
    mask = key.isin(ids)
    matched = set(key[mask].unique())
    unmatched = len(ids) - len(matched)

    out = df.loc[~mask].reset_index(drop=True)
    if verbose:
        print(f"[blacklist] removed {int(mask.sum()):,} of {len(df):,} {label} "
              f"({mask.mean():.2%})")
        print(f"[blacklist]   {len(matched):,} of {len(ids):,} listed IDs matched")
        if unmatched:
            print(f"[blacklist]   {unmatched:,} listed IDs matched nothing")
        if not matched:
            print("[blacklist]   >> NOTHING matched. Check the ID column and "
                  "that both sides are strings -- leading zeros are the "
                  "usual cause.")
    return out


def prepare(
    df: pd.DataFrame,
    data_cfg: DataConfig,
    filter_cfg: FilterConfig,
    split_cfg: SplitConfig,
    level_columns: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Null-drop, rare-class and rare-path filters, then split.

    In that order, and the order matters: filtering after the split
    would leave classes in validation that training has never seen, and
    filtering rare *paths* before rare *classes* wastes work, since the
    class filter removes most of the offending rows anyway.
    """
    levels = list(level_columns or data_cfg.level_columns)
    df = drop_null_rows(
        df,
        data_cfg.text_col,
        levels,
        data_cfg.drop_null_text,
        data_cfg.drop_null_labels,
    )
    df = drop_rare_classes(df, levels[-1], filter_cfg.min_class_count)
    df = drop_rare_paths(df, levels, filter_cfg.min_path_count)
    return split_data(df, levels, split_cfg)


def class_distribution(
    df: pd.DataFrame, level_columns: Sequence[str]
) -> pd.DataFrame:
    """How many distinct classes and how thin the tail is, per level."""
    rows = []
    for col in level_columns:
        counts = df[col].value_counts()
        rows.append({
            "level": col,
            "n_classes": int(len(counts)),
            "min_rows": int(counts.min()) if len(counts) else 0,
            "median_rows": float(counts.median()) if len(counts) else 0.0,
            "max_rows": int(counts.max()) if len(counts) else 0,
            "classes_under_10": int((counts < 10).sum()),
        })
    return pd.DataFrame(rows)
