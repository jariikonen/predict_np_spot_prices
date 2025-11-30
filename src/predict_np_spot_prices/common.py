import ast
from enum import Enum
import os
from pathlib import Path
import re
from typing import Dict, List, Literal, Tuple, TypeAlias
import pandas as pd
import json


DATA_DIR = 'data'
DATA_DIR_ENTSOE = os.path.join(DATA_DIR, 'entsoe')
DATA_DIR_FMI = os.path.join(DATA_DIR, 'fmi')
DATA_DIR_SPECIAL = os.path.join(DATA_DIR, 'special')
DATA_DIR_ARCHIVED = os.path.join(DATA_DIR_SPECIAL, 'archived')
DATA_DIR_PREPROCESSED = os.path.join(DATA_DIR, 'preprocessed')

DF_FILE_EXTENSION = '.pqt'


AreaValue: TypeAlias = Literal[
    'FI',
    'SE',
    'SE_1',
    'SE_2',
    'SE_3',
    'SE_4',
    'NO',
    'NO_1',
    'NO_2',
    'NO_3',
    'NO_4',
    'NO_5',
    'EE',
    'RU',
]
AreaValueCompact: TypeAlias = Literal[
    'FI',
    'SE',
    'SE1',
    'SE2',
    'SE3',
    'NO',
    'NO1',
    'NO2',
    'NO3',
    'NO4',
    'NO5',
    'EE',
    'RU',
]


class Area(Enum):
    FI = 'FI'
    SE = 'SE'
    SE_1 = 'SE_1'
    SE_2 = 'SE_2'
    SE_3 = 'SE_3'
    SE_4 = 'SE_4'
    NO = 'NO'
    NO_1 = 'NO_1'
    NO_2 = 'NO_2'
    NO_3 = 'NO_3'
    NO_4 = 'NO_4'
    NO_5 = 'NO_5'
    NO_TUPLE = 'NO_TUPLE'
    EE = 'EE'
    RU = 'RU'

    @classmethod
    def list_values(cls) -> list[str]:
        return [member.value for member in cls]

    @classmethod
    def list_countries(cls) -> list[str]:
        return ['FI', 'SE', 'NO', 'EE', 'RU']

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
            ('FI', 'EE'),
            ('EE', 'FI'),
            ('FI', 'NO_4'),
            ('NO_4', 'FI'),
            ('FI', 'RU'),
            ('RU', 'FI'),
        ]

    @classmethod
    def is_area_code(cls, area: AreaValue) -> bool:
        return str.upper(area) in cls


def get_area_code(area: AreaValue):
    if Area.is_area_code(area):
        return str.upper(area)
    else:
        raise ValueError(f'"{area}" is not an area code.')


def get_compact_form(area: AreaValue):
    if area is None:
        return None
    if area == Area.SE_1.value:
        return 'SE1'
    elif area == Area.SE_2.value:
        return 'SE2'
    elif area == Area.SE_3.value:
        return 'SE3'
    elif area == Area.SE_4.value:
        return 'SE4'
    elif area == Area.NO_1.value:
        return 'NO1'
    elif area == Area.NO_2.value:
        return 'NO2'
    elif area == Area.NO_3.value:
        return 'NO3'
    elif area == Area.NO_4.value:
        return 'NO4'
    elif area == Area.NO_5.value:
        return 'NO5'
    elif Area.is_area_code(area):
        return area
    else:
        raise ValueError(f'"{area}" is not an area code.')


def get_area_value_from_compact_form(area: AreaValueCompact):
    if area == 'EE' or area == 'FI' or area == 'SE' or area == 'NO':
        return area
    elif area == 'SE1':
        return 'SE_1'
    elif area == 'SE2':
        return 'SE_2'
    elif area == 'SE3':
        return 'SE_3'
    elif area == 'SE4':
        return 'SE_4'
    elif area == 'NO1':
        return 'NO_1'
    elif area == 'NO2':
        return 'NO_2'
    elif area == 'NO3':
        return 'NO_3'
    elif area == 'NO4':
        return 'NO_4'
    elif area == 'NO5':
        return 'NO_5'
    else:
        raise ValueError(f'Unknown compact form area value {area}.')


def get_long_name(area: AreaValue):
    if not Area.is_area_code(area):
        raise ValueError(f'Not an area code {area}')

    code = get_area_code(area)
    country_code = code[:2]

    if country_code == 'FI':
        country = 'Finland'
    elif country_code == 'SE':
        country = 'Sweden'
    elif country_code == 'NO':
        country = 'Norway'
    elif country_code == 'EE':
        country = 'Estonia'

    match = re.search(r'_(\d+)$', code)
    if match:
        return f'{country} {match.group(1)}'
    else:
        return country


class DataCategory(Enum):
    GENERATION = 'generation'
    LOAD = 'load'
    PRICES = 'prices'
    EXCHANGES = 'exchanges'
    FLOWS = 'flows'
    WATER_RESERVOIRS = 'water_reservoirs'

    @classmethod
    def list_names(cls) -> list[str]:
        return [member.name for member in cls]

    @classmethod
    def list_values(cls) -> list[str]:
        return [member.value for member in cls]

    @classmethod
    def is_data_category(cls, data_category: str) -> bool:
        return str.lower(data_category) in cls


DataCategoryValue: TypeAlias = Literal[
    'generation',
    'load',
    'prices',
    'exchanges',
    'flows',
    'water_reservoirs',
]


def get_filename_start(data_category: DataCategoryValue) -> str:
    if data_category == DataCategory.PRICES.value:
        return 'prices'
    if data_category == DataCategory.LOAD.value:
        return 'load'
    elif data_category == DataCategory.GENERATION.value:
        return 'generation_per_prod_type'
    elif data_category == DataCategory.WATER_RESERVOIRS.value:
        return 'water_reservoirs_and_hydro_storage'
    elif data_category == DataCategory.EXCHANGES.value:
        return 'scheduled_exchanges'
    elif data_category == DataCategory.FLOWS.value:
        return 'physical_flows'
    else:
        raise ValueError(f'No filename start for {data_category}')


def get_filename_start_with_areas(
    data_category: DataCategoryValue,
    area_from: AreaValue,
    area_to: AreaValue | None = None,
    add_separator: bool = False,
) -> str:
    if area_from is None and area_to is None:
        return get_filename_start(data_category)
    area_f = get_compact_form(area_from)
    area_t = get_compact_form(area_to)

    filename_start = ''
    if area_to is not None:
        filename_start = (
            f'{get_filename_start(data_category)}_{area_f}-{area_t}'
        )
    else:
        filename_start = f'{get_filename_start(data_category)}_{area_f}'
    if add_separator:
        if data_category == DataCategory.EXCHANGES.value and area_to is None:
            filename_start += '-'
        else:
            filename_start += '_'

    return filename_start


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
    if len(os.path.dirname(file_path)) > 0:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
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


def write_json_file(path: str, data: List[str], silent: bool = False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    if not silent:
        print(f'Data written to {path}')


def date_to_str(date: pd.Timestamp) -> str:
    return date.strftime('%Y-%m-%d')


def create_df_filename(
    name_start: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    area_from: AreaValue | None = None,
    area_to: AreaValue | None = None,
):
    area_f = get_compact_form(area_from) if area_from is not None else None
    area_t = get_compact_form(area_to) if area_to is not None else None
    if area_from is not None and area_to is not None:
        return f'{name_start}_{area_f}-{area_t}_{date_to_str(start)}--{date_to_str(end)}{DF_FILE_EXTENSION}'
    elif area_to is None:
        return f'{name_start}_{area_f}_{date_to_str(start)}--{date_to_str(end)}{DF_FILE_EXTENSION}'
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
        codes = match.group('codes').split('-')
        return [get_area_value_from_compact_form(c) for c in codes]
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


def get_filename_start_from_filename(filename: str) -> str:
    pattern = re.compile(
        r"""
        ^
        (?P<start>[a-z_]+)                      # start: lowercase + underscores
        _
        (?P<code>[A-Z0-9_]+(?:-[A-Z0-9_]+)?)    # code or code-code
        _
        \d{4}-\d{2}-\d{2}--\d{4}-\d{2}-\d{2}    # date range
        \.[^.]+                                 # suffix
        $
        """,
        re.VERBOSE,
    )

    m = pattern.match(filename)
    if not m:
        return None
    return m.group('start'), m.group('code')


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
        if f.startswith(filename_start) and f.endswith(DF_FILE_EXTENSION)
    ]
    return files


def get_dfs(
    dir_path: str,
    data_category: DataCategoryValue,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    area_from: AreaValue | None = None,
    area_to: AreaValue | None = None,
):
    filename_start = get_filename_start(data_category)
    if area_from is not None and area_to is not None:
        filename_start += (
            f'_{get_compact_form(area_from)}-{get_compact_form(area_to)}'
        )
    elif area_from is not None:
        filename_start += f'_{get_compact_form(area_from)}'
        if area_to is not None:
            filename_start += f'-{get_compact_form(area_to)}'
    if start is not None and end is not None:
        filename_start += f'{date_to_str(start)}--{date_to_str(end)}'

    # add underscore to separate, e.g., SE from SE1
    filename_start += '_'

    files = get_df_files(dir_path, filename_start)
    if len(files) == 0:
        raise ValueError(
            f'No matching {DF_FILE_EXTENSION} files in directory "{dir_path}" ({filename_start}).'
        )

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


def cast_df_array_to_dict(dfs: List[pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    df_dict = {}
    for df in dfs:
        name = df.attrs.get('name', None)
        area = get_areas_from_filename(name + DF_FILE_EXTENSION)[0]
        df_dict[area] = df
    return df_dict


def get_df_area(df: pd.DataFrame) -> List[AreaValue]:
    """
    Returns area codes based on the df filename stored as attrs.name.
    """
    name = df.attrs.get('name', None)
    return get_areas_from_filename(name + DF_FILE_EXTENSION)


def check_multi_index(df: pd.DataFrame):
    return {
        'columns': isinstance(df.columns, pd.MultiIndex),
        'index': isinstance(df.index, pd.MultiIndex),
    }


def print_df_data(df: pd.DataFrame, head: int = 5, tail: int = 5):
    print(f'\ndescription:\n{df.describe()}')
    print(f'\nhead:\n{df.head(head)}')
    print(f'\ntail:\n{df.tail(tail)}')
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


def prompt_yes_no(prompt='Continue? (y/n): '):
    while True:
        ans = input(prompt).strip().lower()
        if ans in ('y', 'yes'):
            return True
        if ans in ('n', 'no'):
            return False
        print('Please enter yes or no (y/n).')


def write_df(df: pd.DataFrame, file_path: str):
    path = Path(file_path)
    if not path.suffix == DF_FILE_EXTENSION:
        path = path.with_suffix(DF_FILE_EXTENSION)
    df.to_parquet(path)


def read_df(file_path: str) -> pd.DataFrame:
    df = pd.read_parquet(file_path)

    # add name attr to the df if there isn't one
    name = df.attrs.get('name', None)
    if name is None:
        base, _ = os.path.splitext(file_path)
        df.attrs['name'] = f'{base}'

    return df
