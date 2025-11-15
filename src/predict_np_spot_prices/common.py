import ast
from enum import Enum
import os
import re
from typing import List, Literal, Tuple, TypeAlias

import pandas as pd


DATA_DIR = 'data'
DATA_DIR_EDA = os.path.join(DATA_DIR, 'eda')


AreaValue: TypeAlias = Literal['FI', 'SE', 'SE_1', 'SE_3', 'NO', 'NO_4', 'EE']


class Area(Enum):
    FI = 'FI'
    SE = 'SE'
    SE_1 = 'SE_1'
    SE_3 = 'SE_3'
    NO = 'NO'
    NO_4 = 'NO_4'
    NO_TUPLE = 'NO_TUPLE'
    EE = 'EE'

    @classmethod
    def list_values(cls) -> list[str]:
        return [member.value for member in cls]

    @classmethod
    def list_countries(cls) -> list[str]:
        return ['FI', 'SE', 'NO', 'EE']

    @classmethod
    def list_bidding_zones(cls) -> list[str]:
        return ['FI', 'SE_1', 'SE_3', 'NO_4', 'EE']

    @classmethod
    def list_fetchable(cls) -> list[str]:
        return ['FI', 'SE', 'SE_1', 'SE_3', 'NO', 'NO_4', 'EE']

    @classmethod
    def list_exchanges(cls) -> list[tuple[str]]:
        return [
            ('FI', 'SE_1'),
            ('SE_1', 'FI'),
            ('FI', 'SE_3'),
            ('SE_3', 'FI'),
            ('FI', 'NO_4'),
            ('NO_4', 'FI'),
            ('FI', 'EE'),
            ('EE', 'FI'),
        ]

    @classmethod
    def is_area_code(cls, area: AreaValue) -> bool:
        return str.upper(area) in cls


def get_area_code(area: AreaValue):
    if Area.is_area_code(area):
        return str.upper(area)
    else:
        raise ValueError(f'"{area}" is not an area code.')


def get_country(area: AreaValue):
    code = get_area_code(area)[:2]
    if code == 'FI':
        return 'Finland'
    elif code == 'SE':
        return 'Sweden'
    elif code == 'NO':
        return 'Norway'
    elif code == 'EE':
        return 'Estonia'
    else:
        raise ValueError(f'No country for {area}.')


class DataItem(Enum):
    GENERATION = 'generation'
    LOAD = 'load'
    PRICES = 'prices'
    EXCHANGES = 'exchanges'
    WATER_RESERVOIRS = 'water_reservoirs'

    @classmethod
    def list_names(cls) -> list[str]:
        return [member.name for member in cls]

    @classmethod
    def list_values(cls) -> list[str]:
        return [member.value for member in cls]

    @classmethod
    def is_data_item(cls, data_item: str) -> bool:
        return str.lower(data_item) in cls


DataItemValue: TypeAlias = Literal[
    'generation', 'load', 'pricces', 'exchanges', 'water_reservoirs'
]


def get_filename_start(data_item: DataItemValue) -> str:
    if data_item == DataItem.PRICES.value:
        return 'prices'
    if data_item == DataItem.LOAD.value:
        return 'load'
    elif data_item == DataItem.GENERATION.value:
        return 'generation_per_prod_type'
    elif data_item == DataItem.WATER_RESERVOIRS.value:
        return 'water_reservoirs_and_hydro_storage'
    elif data_item == DataItem.EXCHANGES.value:
        return 'scheduled_exchanges'
    else:
        raise ValueError(f'No filename start for {data_item}')


def get_filename_start_with_areas(
    data_item: DataItemValue,
    area_from: AreaValue,
    area_to: AreaValue | None = None,
) -> str:
    """
    Creates the filename start with the area codes followed by an underscore
    and number 2, which is the first character of date range (to separate,
    e.g., SE from SE_1).
    """
    if data_item == DataItem.EXCHANGES.value:
        if area_to is None:
            raise ValueError('DataItem.EXCHANGES needs two area codes.')
        return f'{get_filename_start(data_item)}_{area_from}-{area_to}_2'
    else:
        return f'{get_filename_start(data_item)}_{area_from}_2'


def set_utc(ts: pd.Timestamp) -> pd.Timestamp:
    if ts.tzinfo is None:
        new_ts = ts.tz_localize(tz='UTC')
    else:
        new_ts = ts.tz_convert(tz='UTC')
    return new_ts


def get_timestamp(date_str: str) -> pd.Timestamp:
    ts = pd.Timestamp(date_str)
    ts = set_utc(ts)
    return ts


class Freq(Enum):
    HOUR = 'h'


def write_file(file_path: str, data: bytes, silent: bool = False):
    """
    Writes the bytes to a local file.
    """
    if len(data) == 0:
        print('No data to write.')
        return
    with open(file_path, 'wb') as f:
        f.write(data)
    if not silent:
        print(f'Data written to {file_path}')


def get_unique_path(path):
    """
    Generate a unique file path by appending (n) if the file already exists.
    Example:
      "file.txt" -> "file(1).txt" -> "file(2).txt"
    """
    base, ext = os.path.splitext(path)
    counter = 1
    new_path = path

    while os.path.exists(new_path):
        new_path = f'{base}({counter}){ext}'
        counter += 1

    return new_path


def write_file_unique_name(path: str, data: bytes, silent: bool = False) -> str:
    """
    Writes data to the path creating a unique name variant if the file already
    exists.

    Returns:
        string: The path to the file.
    """
    file_path = get_unique_path(path)
    write_file(file_path, data, silent)
    return file_path


def date_to_str(date: pd.Timestamp) -> str:
    return date.strftime('%Y-%m-%d')


def create_df_filename(
    name_start: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    area_from: AreaValue | None = None,
    area_to: AreaValue | None = None,
):
    if area_from is not None and area_to is not None:
        return f'{name_start}_{area_from}-{area_to}_{date_to_str(start)}--{date_to_str(end)}.pqt'
    elif area_to is None:
        return f'{name_start}_{area_from}_{date_to_str(start)}--{date_to_str(end)}.pqt'
    else:
        raise ValueError(
            'Argument area or arguments area_from and area_to must not be None.'
        )


def get_areas_from_filename(filename: str) -> List[str]:
    pattern = r"""
        ^[a-z_]+_                 # filename start
        (?P<codes>[A-Z0-9_]+      # first area code
        (?:-[A-Z0-9_]+)?)         # optional second area code
        _\d{4}-\d{2}-\d{2}        # first date
        --\d{4}-\d{2}-\d{2}       # second date
        \..+$                     # suffix
    """
    match = re.match(pattern, filename, re.VERBOSE)
    if match:
        codes = match.group('codes')
        return codes.split('-')
    return []


def get_date_range_from_filename(
    filename: str,
) -> None | Tuple[pd.Timestamp, pd.Timestamp]:
    pattern = r'\d{4}-\d{2}-\d{2}--\d{4}-\d{2}-\d{2}'
    match = re.search(pattern, filename)
    if not match:
        return None
    start_str, end_str = match.group(0).split('--')
    return pd.Timestamp(start_str), pd.Timestamp(end_str)


def change_area_codes(filename: str, new_codes: str | list[str]) -> str:
    """
    Change the area code(s) in a filename with structure:
    <start>_<area_code>_<'%Y-%m-%d'>--<'%Y-%m-%d'>
    or
    <start>_<area_code>-<area_code>_<'%Y-%m-%d'>--<'%Y-%m-%d'>

    Args:
        filename (str): Original filename.
        new_codes (str | list[str]): New area code(s). Use a list for multiple.

    Returns:
        str: Updated filename.
    """
    # convert new_codes to a string (handles single or multiple)
    if isinstance(new_codes, list):
        new_area = '-'.join(new_codes)
    else:
        new_area = new_codes

    # regex to find the area code section between first underscore and the date
    new_filename = re.sub(
        r'_(?:[A-Za-z]+(?:-[A-Za-z]+)*)_(\d{4}-\d{2}-\d{2}--\d{4}-\d{2}-\d{2})',
        lambda m: f'_{new_area}_{m.group(1)}',
        filename,
    )

    return new_filename


def rename_tuple_columns(df: pd.DataFrame) -> pd.DataFrame:
    df_new = df.rename(
        columns={
            c: f'{ast.literal_eval(c)[0]} ({ast.literal_eval(c)[1]})'
            for c in list(df.columns)
            if c.startswith('(')
        }
    )
    return df_new


def get_tuple_name_pairs(
    df: pd.DataFrame,
) -> Tuple[List[Tuple[str, str]], List[str]]:
    tuple_cols = [c for c in df.columns if '(' in c]
    simple_cols = [c for c in df.columns if '(' not in c]

    def split_col_name(c):
        m = re.match(r'^(.*?)\s*\((.*?)\)$', c)
        return m.groups() if m else (c, None)

    pairs = []
    for t in tuple_cols:
        type, _ = split_col_name(t)
        for s in simple_cols:
            if type == s:
                pairs.append((t, s))

    return pairs, tuple_cols


def get_df_files(
    dir_path: str,
    filename_start: str,
) -> List[str]:
    files = [
        f
        for f in os.listdir(dir_path)
        if f.startswith(filename_start) and f.endswith('.pqt')
    ]
    return files


def get_dfs(
    dir_path: str,
    data_item: DataItemValue,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    area_from: AreaValue | None = None,
    area_to: AreaValue | None = None,
):
    filename_start = get_filename_start(data_item)
    if area_from is not None and area_to is not None:
        filename_start += (
            f'_{get_area_code(area_from)}-{get_area_code(area_to)}'
        )
    elif area_from is not None:
        filename_start += f'_{get_area_code(area_from)}'
        if area_to is not None:
            filename_start += f'-{get_area_code(area_to)}'
    if start is not None and end is not None:
        filename_start += f'{date_to_str(start)}--{date_to_str(end)}'

    files = get_df_files(dir_path, filename_start)
    if len(files) == 0:
        raise ValueError(f'No matching .pqt files in directory "{dir_path}".')

    dfs: List[pd.DataFrame] = []
    for f in files:
        file_path = os.path.join(dir_path, f)
        df = pd.read_parquet(file_path)

        # add name attr to the df if there isn't one
        name = df.attrs.get('name', None)
        if name is None:
            base, _ = os.path.splitext(file_path)
            df.attrs['name'] = f'{base}'

        dfs.append(df)

    return dfs


def check_multi_index(df: pd.DataFrame):
    return {
        'columns': isinstance(df.columns, pd.MultiIndex),
        'index': isinstance(df.index, pd.MultiIndex),
    }


def print_df_data(df: pd.DataFrame):
    print(f'\nhead:\n{df.head()}')
    print(f'\ntail:\n{df.tail()}')
    print(f'\ndtypes:\n{df.dtypes}')
    print(f'\nrow[0]:\n{df.iloc[0]}')
    print(f'\nrow[0].sum():\n{df.iloc[0].sum()}')
    print(f'\nrow[{df.shape[0]}]:\n{df.iloc[-1]}')
    print(f'\nrow[{df.shape[0]}].sum():\n{df.iloc[-1].sum()}')
    print('\nColumn means:')
    for col in df.columns:
        print(f'{col}: {df[col].mean()}')
    multi_index = check_multi_index(df)
    print(f'\nMultilevel columns: {multi_index["columns"]}')
    print(f'\nMultilevel index: {multi_index["index"]}\n\n')


def get_next_hour(ts: pd.Timestamp) -> pd.Timestamp:
    return (ts + pd.Timedelta(nanoseconds=1)).ceil('h')
