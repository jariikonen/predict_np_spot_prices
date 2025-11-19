import os
import sys
from typing import List, Tuple, TypeAlias, Literal
import pandas as pd
from predict_np_spot_prices.common import (
    DATA_DIR,
    DATA_DIR_SPECIAL,
    DATA_DIR_PREPROCESSED,
    DF_FILE_EXTENSION,
    DataCategory,
    Freq,
    change_area_codes,
    get_filename_start,
    get_filename_start_with_areas,
    get_tuple_name_pairs,
    read_df,
    rename_tuple_columns,
    write_df,
)


def validate_datetime_index(
    df,
    freq: Freq = None,
    start: pd.Timestamp = None,
    end: pd.Timestamp = None,
):
    ident = df.attrs.get('name', f'id={id(df)}')

    # check that index is a DatetimeIndex and UTC
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            f'Expected DatetimeIndex, got {type(df.index)} for DataFrame "{ident}".'
        )
    if df.index.tz is None or str(df.index.tz) != 'UTC':
        raise ValueError(
            f'Expected the index to have UTZ tz, got {type(df.index)} for DataFrame "{ident}".'
        )

    # check that the resolution is correct
    inferred_freq = pd.infer_freq(df.index)
    if freq.value is not None and inferred_freq != freq.value:
        raise ValueError(
            f'Expected freq {freq}, got {inferred_freq} for DataFrame "{ident}".'
        )

    # check df has the correct time period
    if start is not None and df.index.min() != pd.Timestamp(
        start, tz=df.index.tz
    ):
        raise ValueError(
            f'Wrong start time: {df.index.min()} != {start} for DataFrame "{ident}".'
        )

    if end is not None and df.index.max() != pd.Timestamp(end, tz=df.index.tz):
        raise ValueError(
            f'Wrong end time: {df.index.max()} != {end} for DataFrame "{ident}".'
        )

    # check there are no missing timestamps
    if freq is not None:
        expected = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq=freq.value,
            tz=df.index.tz,
        )
        missing = expected.difference(df.index)
        if len(missing):
            raise ValueError(
                f'Missing timestamps: {len(missing)} for DataFrame "{ident}".'
            )

    return True


def drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].sum() == 0:
            df = df.drop(columns=[col])
    return df


MergeMethod: TypeAlias = Literal[
    'first',
    'last',
    'average',
]


def merge_columns(
    df: pd.DataFrame, columns: Tuple[str, str], method: MergeMethod = 'first'
) -> pd.Series:
    """
    Merge multiple columns into one continuous series.

    Args:
        df (pd.DataFrame): DataFrame containing the columns to merge.
        columns (List[str]): List of columns to merge.
        method (str, default 'first'): Method for resolving overlaps.
            Methods: 'first' - take the first column's value, 'last' - take
            the last column's value, 'average' - take the mean of overlapping
            values.

    Returns:
        pd.Series : merged continuous series
    """
    if len(columns) < 2:
        return df[columns[0]].copy()

    # Detect overlaps: more than one non-NaN value per row
    overlap_mask = df[[columns[0], columns[1]]].notna().sum(axis=1) > 1
    if overlap_mask.any():
        ident = df.attrs.get('name', f'id={id(df)}')
        print(
            f'Warning: Overlaps detected in dataframe {ident} at the following indices:'
        )
        print(df.index[overlap_mask])

    if method == 'first':
        merged = df[columns[0]].copy()
        merged = merged.combine_first(df[columns[1]])
    elif method == 'last':
        merged = df[columns[1]].copy()
        merged = merged.combine_first(df[columns[0]])
    elif method == 'average':
        merged = df[[columns[0], columns[1]]].mean(axis=1, skipna=True)
    else:
        raise ValueError('method must be one of "first", "last", or "average"')

    return merged


def merge_tuple_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses a dataframe with multi-level columns flattened to single-level
    columns with tuple-like strings as names, by merging them to their simply
    named counterparts.

    This happens by creating a new merged column where the values from the
    simply named column and its tuple-like counterpart are merged using
    pd.DataFrame.combine_first function. After merging, the original columns
    are dropped. Any remaining tuple named columns are also dropped.
    """
    df_new = rename_tuple_columns(df)
    pairs, tuple_columns = get_tuple_name_pairs(df_new)
    for p in pairs:
        tuple_col, simple_col = p
        if 'Aggregate' in tuple_col:
            merged = merge_columns(df_new, p, method='first')
            df_new['merged'] = merged
            df_new = df_new.drop(columns=[tuple_col, simple_col])
            df_new = df_new.rename(columns={'merged': simple_col})
            tuple_columns.remove(tuple_col)
    df_new = df_new.drop(columns=tuple_columns)
    return df_new


def flatten_multi_level_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses a NO and NO_4 generation dataframe with multi-level columns
    by converting the multi-level columns to single-level columns and removing
    the consumption columns.
    """
    df_new = df.copy()
    df_new.columns = df_new.columns.to_flat_index()
    df_new.columns = [f'{c[0]} ({c[1]})' for c in df_new.columns]
    consumption_columns = [c for c in df_new.columns if 'Consumption' in c]
    df_new.drop(columns=consumption_columns)
    df_new = df_new.rename(
        columns={
            c: c.replace(' (Actual Aggregated)', '')
            for c in list(df_new.columns)
            if 'Aggregated' in c
        }
    )
    return df_new


def preprocess_generation_df(
    df: pd.DataFrame,
    handle_tuple_columns: bool = True,
) -> pd.DataFrame:
    df_hourly = df.resample('h').mean()

    if handle_tuple_columns:
        if isinstance(df_hourly.columns, pd.core.indexes.multi.MultiIndex):
            df_hourly = flatten_multi_level_columns(df_hourly)
        else:
            df_hourly = merge_tuple_columns(df_hourly)

    df_hourly = df_hourly.fillna(0)
    df_hourly = drop_empty_columns(df_hourly)

    return df_hourly


def preprocess_generation(
    dir_from: str,
    dir_to: str,
    handle_tuple_columns: bool = True,
    prefix: str = 'pp',
) -> pd.DataFrame:
    filename_start = get_filename_start(DataCategory.GENERATION.value)
    files = [
        f
        for f in os.listdir(dir_from)
        if f.startswith(filename_start) and f.endswith(DF_FILE_EXTENSION)
    ]
    if len(files) == 0:
        raise ValueError(
            f'No matching {DF_FILE_EXTENSION} files in directory {dir_from}.'
        )

    for f in files:
        print(f'found file: {f}')
        file_path = os.path.join(dir_from, f)
        df = read_df(file_path)
        df = preprocess_generation_df(df, handle_tuple_columns)
        validate_datetime_index(df, Freq.HOUR)
        save_df(df, dir_to, f, prefix)


def preprocess_generation_special(
    dir_from: str,
    dir_to: str,
):
    filename_start = (
        get_filename_start_with_areas(DataCategory.GENERATION.value, 'NO') + '_'
    )
    files = [
        f
        for f in os.listdir(dir_from)
        if f.startswith(filename_start) and f.endswith(DF_FILE_EXTENSION)
    ]
    if len(files) == 0:
        raise ValueError(
            f'No matching {DF_FILE_EXTENSION} files in directory {dir_from}.'
        )

    for f in files:
        df = read_df(os.path.join(dir_from, f))
        df_pp = preprocess_generation_df(df, handle_tuple_columns=False)
        validate_datetime_index(df_pp, Freq.HOUR)
        save_df(df_pp, dir_to, f, 'pp_special', 'NO_TUPLE')


def preprocess(
    special: bool = False,
    dir_from: str | None = DATA_DIR,
    dir_to: str | None = DATA_DIR_PREPROCESSED,
):
    if special:
        preprocess_special()
        sys.exit(0)

    preprocess_generation(dir_from, dir_to, True)


def preprocess_special(
    dir_from: str | None = DATA_DIR,
    dir_to: str | None = DATA_DIR_SPECIAL,
):
    preprocess_generation_special(dir_from, dir_to)


def save_df(
    df: pd.DataFrame,
    dir_path: str,
    filename: str,
    prefix: str | None = None,
    new_areas: str | List[str] | None = None,
):
    """
    Saves the dataframe to dir_name/filename as a parquet file.

    Adds also the filename as a "name" attribute to the dataframe before saving
    it to the file.

    Args:
        df (pd.DataFrame): Dataframe to be saved.
        dir_path (str): Path to the directory where the file is saved.
        filename (str): Name of the file.
        prefix (str | None): A prefix added to the filename when used as a
            "name" attribute.
        new_areas (str | List(str) | None): New area code(s) used in the
            filename if provided.
    """
    os.makedirs(dir_path, exist_ok=True)

    if new_areas is not None:
        filename = change_area_codes(filename, new_areas)

    base, _ = os.path.splitext(filename)
    prefix_part = f'{prefix}_' if prefix is not None else ''
    df.attrs['name'] = f'{prefix_part}{base}'

    file_path = os.path.join(dir_path, filename)
    write_df(df, file_path)
    print(f'preprocessed dataframe written to {file_path}')
