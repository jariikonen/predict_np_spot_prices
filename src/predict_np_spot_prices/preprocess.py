import os
from pathlib import Path
import sys
from typing import Callable, List, Tuple, TypeAlias, Literal
import pandas as pd
from predict_np_spot_prices.common import (
    DATA_DIR_ENTSOE,
    DATA_DIR_FMI,
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
    if (
        freq is not None
        and freq.value is not None
        and inferred_freq != freq.value
    ):
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


def merge_tuple_generation_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses a generation data dataframe with multi-level columns flattened
    to single-level columns with tuple-like strings as names, by merging them
    to their simply named counterparts.

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


def flatten_multi_level_generation_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses a generation data dataframe with multi-level columns by
    converting the multi-level columns to single-level columns.
    """
    df_new = df.copy()
    df_new.columns = df_new.columns.to_flat_index()
    df_new.columns = [f'{c[0]} ({c[1]})' for c in df_new.columns]
    df_new = df_new.rename(
        columns={
            c: c.replace(' (Actual Aggregated)', '')
            for c in list(df_new.columns)
            if 'Aggregated' in c
        }
    )
    return df_new


def get_rows_with_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.isnull().any(axis=1)]


def get_rows_with_str_values(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.map(lambda x: isinstance(x, str)).any(axis=1)]


def preprocess_prices_df(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df_hourly = df.resample('h').mean()
    missing = get_rows_with_missing_values(df_hourly)
    df_hourly = df_hourly.fillna(0)
    if len(missing) > 0:
        raise Exception('Missing values')
    df_hourly = drop_empty_columns(df_hourly)

    # prices from ENTSO-E are "day-ahead", so they must be aligned to the
    # actual consumption day
    df_hourly.index = df_hourly.index + pd.Timedelta(days=1)

    return df_hourly


def preprocess_load_df(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df_hourly = df.resample('h').mean()
    df_hourly = df_hourly.fillna(0)
    missing = get_rows_with_missing_values(df_hourly)
    if len(missing) > 0:
        raise Exception(f'Missing values ({len(missing)})')
    df_hourly = drop_empty_columns(df_hourly)

    return df_hourly


def preprocess_generation_df(
    df: pd.DataFrame,
    handle_tuple_columns: bool = True,
) -> pd.DataFrame:
    df_hourly = df.resample('h').mean()

    if handle_tuple_columns:
        if isinstance(df_hourly.columns, pd.core.indexes.multi.MultiIndex):
            df_hourly = flatten_multi_level_generation_columns(df_hourly)
        else:
            df_hourly = merge_tuple_generation_columns(df_hourly)

    # drop consumption columns
    consumption_columns = [c for c in df_hourly.columns if 'Consumption' in c]
    df_hourly.drop(columns=consumption_columns)

    df_hourly = df_hourly.fillna(0)
    missing = get_rows_with_missing_values(df_hourly)
    if len(missing) > 0:
        raise Exception(f'Missing values ({len(missing)})')

    df_hourly = drop_empty_columns(df_hourly)

    return df_hourly


def preprocess_flows_df(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df_hourly = df.resample('h').mean()
    df_hourly = df_hourly.fillna(0)
    missing = get_rows_with_missing_values(df_hourly)
    if len(missing) > 0:
        raise Exception(f'Missing values {len(missing)}')
    df_hourly = drop_empty_columns(df_hourly)

    return df_hourly


def preprocess_exchanges_df(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df_hourly = df.resample('h').mean()
    df_hourly = df_hourly.fillna(0)
    missing = get_rows_with_missing_values(df_hourly)
    if len(missing) > 0:
        raise Exception(f'Missing values ({len(missing)})')
    df_hourly = drop_empty_columns(df_hourly)

    return df_hourly


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


def weather_create_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    df_pp = df.copy()
    df_pp['datetime'] = pd.to_datetime(
        df_pp['Vuosi'].astype(str)
        + '-'
        + df_pp['Kuukausi'].astype(str)
        + '-'
        + df_pp['Päivä'].astype(str)
        + ' '
        + df_pp['Aika [UTC]'],
        utc=True,
    )
    df_pp = df_pp.drop(columns=['Vuosi', 'Kuukausi', 'Päivä', 'Aika [UTC]'])
    df_pp = df_pp.rename(columns={'datetime': 'time'})
    df_pp = df_pp.set_index('time')
    df_pp = df_pp.asfreq('h')
    return df_pp


def print_missing_dates(df: pd.DataFrame):
    full_range = pd.date_range(
        start=df.index.min(), end=df.index.max(), freq='h'
    )
    missing = full_range.difference(df.index)
    print(missing)


def weather_common_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df_pp = weather_create_datetime_index(df)
    return df_pp


def preprocess_temperature(df: pd.DataFrame) -> pd.DataFrame:
    df_pp = weather_common_preprocessing(df)
    station = df_pp['Havaintoasema'].dropna().iloc[0]
    df_pp = df_pp.rename(columns={'Lämpötilan keskiarvo [°C]': station})
    df_pp = df_pp.drop(columns=['Havaintoasema'])
    df_pp[station] = df_pp[station].interpolate(method='time')
    missing = get_rows_with_missing_values(df_pp)
    if len(missing) > 0:
        raise Exception(
            f'Filling missing values failed [{", ".join(missing)}].'
        )
    return df_pp


def preprocess_wind(df: pd.DataFrame) -> pd.DataFrame:
    df_pp = weather_common_preprocessing(df)
    station = df_pp['Havaintoasema'].dropna().iloc[0]
    avg_speed = f'{station}_avg_speed'
    df_pp = df_pp.rename(
        columns={
            'Keskituulen nopeus [m/s]': avg_speed,
        }
    )
    df_pp = df_pp.drop(
        columns=['Havaintoasema', 'Tuulen suunnan keskiarvo [°]']
    )

    df_pp = df_pp.apply(pd.to_numeric, errors='coerce')

    pd.set_option('future.no_silent_downcasting', True)
    df_pp[avg_speed] = df_pp[avg_speed].ffill().bfill()
    missing = get_rows_with_missing_values(df_pp)
    if len(missing) > 0:
        raise Exception(
            f'Filling missing values failed [{", ".join(missing)}].'
        )
    return df_pp


def preprocess(
    special: bool = False,
    dir_from: str | None = None,
    dir_to: str | None = DATA_DIR_PREPROCESSED,
):
    dir_from_entsoe = dir_from if dir_from is not None else DATA_DIR_ENTSOE
    dir_from_fmi = dir_from if dir_from is not None else DATA_DIR_FMI

    if special:
        preprocess_special(dir_from_entsoe, DATA_DIR_SPECIAL)
        sys.exit(0)

    preprocess_entsoe(dir_from_entsoe, dir_to)
    preprocess_fmi(dir_from_fmi, dir_to)


def preprocess_entsoe(
    dir_from: str,
    dir_to: str,
):
    for dcategory in ['prices', 'load', 'generation', 'exchanges', 'flows']:
        filename_start = get_filename_start(dcategory)
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

            if dcategory == DataCategory.PRICES.value:
                df = preprocess_prices_df(df)
            elif dcategory == DataCategory.LOAD.value:
                df = preprocess_load_df(df)
            elif dcategory == DataCategory.GENERATION.value:
                df = preprocess_generation_df(df, handle_tuple_columns=True)
            elif dcategory == DataCategory.EXCHANGES.value:
                df = preprocess_exchanges_df(df)
            elif dcategory == DataCategory.FLOWS.value:
                df = preprocess_flows_df(df)

            validate_datetime_index(df, Freq.HOUR)
            save_df(df, dir_to, f)


def preprocess_fmi(dir_from: str, dir_to: str):
    def read_and_preprocess(
        file_path: str,
        filename: str,
        preprocessing_func: Callable[[pd.DataFrame], pd.DataFrame],
    ):
        print(f'found file: {filename}')
        df = pd.read_csv(file_path)
        df = preprocessing_func(df)
        validate_datetime_index(df, Freq.HOUR)
        return df

    temperature_dir = os.path.join(dir_from, 'temperature')
    temperature_files = [
        f for f in os.listdir(temperature_dir) if f.endswith('.csv')
    ]
    temperature_dfs = []
    for f in temperature_files:
        file_path = os.path.join(temperature_dir, f)
        temperature_dfs.append(
            read_and_preprocess(file_path, f, preprocess_temperature)
        )
    df_pp = pd.concat(temperature_dfs, axis=1)
    df_pp['temperature_mean'] = df_pp.mean(axis=1)
    df_pp = df_pp[['temperature_mean']]
    range_start = pd.Timestamp(df_pp.iloc[0].name).strftime('%Y-%m-%d')
    range_end = pd.Timestamp(df_pp.iloc[-1].name).strftime('%Y-%m-%d')
    filename = f'temperature_mean_{range_start}-{range_end}'
    save_df(df_pp, dir_to, filename)

    wind_dir = os.path.join(dir_from, 'wind')
    wind_files = [f for f in os.listdir(wind_dir) if f.endswith('.csv')]
    wind_dfs = []
    for f in wind_files:
        file_path = os.path.join(wind_dir, f)
        wind_dfs.append(read_and_preprocess(file_path, f, preprocess_wind))
    df_pp = pd.concat(wind_dfs, axis=1)
    df_pp['avg_speed'] = df_pp.mean(axis=1)
    df_pp = df_pp[['avg_speed']]
    range_start = pd.Timestamp(df_pp.iloc[0].name).strftime('%Y-%m-%d')
    range_end = pd.Timestamp(df_pp.iloc[-1].name).strftime('%Y-%m-%d')
    filename = f'wind_mean_{range_start}-{range_end}'
    save_df(df_pp, dir_to, filename)


def preprocess_special(dir_from: str, dir_to: str):
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

    filename = Path(filename)
    if not filename.suffix == DF_FILE_EXTENSION:
        filename = filename.with_suffix(DF_FILE_EXTENSION)

    file_path = os.path.join(dir_path, filename)
    write_df(df, file_path)
    print(f'preprocessed dataframe written to {file_path}')
