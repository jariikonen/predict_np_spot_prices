import os
from typing import List
import pandas as pd
from predict_np_spot_prices.common import (
    DATA_DIR,
    DATA_DIR_EDA,
    DataItem,
    Freq,
    change_area_codes,
    get_filename_start,
    get_tuple_name_pairs,
    rename_tuple_columns,
    set_utc,
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


def preprocess_tuple_columns(df: pd.DataFrame) -> pd.DataFrame:
    df_new = rename_tuple_columns(df)
    pairs, tuple_columns = get_tuple_name_pairs(df_new)
    for p in pairs:
        tuple_col, simple_col = p
        cutoff = set_utc(pd.Timestamp('2025-01-01'))
        if 'Aggregate' in tuple_col:
            merged = df_new[simple_col].copy()
            merged.loc[df_new.index >= cutoff] = df_new.loc[
                df_new.index >= cutoff, tuple_col
            ]
            df_new['merged'] = merged
            df_new = df_new.drop(columns=[tuple_col, simple_col])
            df_new = df_new.rename(columns={'merged': simple_col})
            tuple_columns.remove(tuple_col)
    df_new = df_new.drop(columns=tuple_columns)
    return df_new


def preprocess_multi_level_columns(df: pd.DataFrame) -> pd.DataFrame:
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


def preprocess_generation(
    df: pd.DataFrame, handle_tuple_columns: bool = True
) -> pd.DataFrame:
    df_hourly = df.resample('h').mean()
    df_hourly = df_hourly.fillna(0)
    df_hourly = drop_empty_columns(df_hourly)

    if handle_tuple_columns:
        if isinstance(df_hourly.columns, pd.core.indexes.multi.MultiIndex):
            df_hourly = preprocess_multi_level_columns(df_hourly)
        else:
            df_hourly = preprocess_tuple_columns(df_hourly)

    return df_hourly


def preprocess_generation_eda(
    df: pd.DataFrame, handle_tuple_columns: bool = True
) -> pd.DataFrame:
    df_pp = preprocess_generation(df, handle_tuple_columns)
    df_pp['sum'] = df_pp.sum(axis=1)

    for col in df_pp.columns:
        if col != 'sum':
            df_pp[col] = df_pp[col] / df_pp['sum'] * 100

    return df_pp


def preprocess(eda: bool = False):
    if eda:
        return preprocess_eda()
    else:
        raise NotImplementedError(
            'Normal preprocessing is not yet implemented.'
        )


def save_df(
    df: pd.DataFrame,
    dir_name: str,
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
        dir_name (str): Name of the directory where the file is saved.
        filename (str): Name of the file.
        prefix (str | None): A prefix added to the filename when used as a
            "name" attribute.
        new_areas (str | List(str) | None): New area code(s) used in the
            filename if provided.
    """
    if new_areas is not None:
        filename = change_area_codes(filename, new_areas)

    base, _ = os.path.splitext(filename)
    df.attrs['name'] = f'{prefix}_{base}'

    file_path = os.path.join(dir_name, filename)
    df.to_parquet(file_path)
    print(f'preprocessed dataframe written to {file_path}')


def preprocess_eda():
    filename_start = get_filename_start(DataItem.GENERATION.value)
    from_dir = DATA_DIR
    files = [
        f
        for f in os.listdir(from_dir)
        if f.startswith(filename_start) and f.endswith('.pqt')
    ]
    if len(files) == 0:
        raise ValueError(f'No matching .pqt files in directory "{from_dir}".')

    to_dir = DATA_DIR_EDA
    for f in files:
        print(f'found file: {f}')
        df = pd.read_parquet(os.path.join(DATA_DIR, f))
        if 'NO' in f and not isinstance(
            df.columns, pd.core.indexes.multi.MultiIndex
        ):
            df = preprocess_generation_eda(df, False)
            validate_datetime_index(df, Freq.HOUR)
            save_df(df, to_dir, f, 'pp_eda', 'NO_TUPLE')
            df = pd.read_parquet(os.path.join(DATA_DIR, f))
        df = preprocess_generation_eda(df)
        validate_datetime_index(df, Freq.HOUR)
        save_df(df, to_dir, f, 'pp_eda')
