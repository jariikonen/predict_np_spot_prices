import glob
import os
import sys
from typing import List, TypeAlias
from collections.abc import Callable
from predict_np_spot_prices.common import (
    DATA_DIR,
    DATA_DIR_EDA,
    Area,
    AreaValue,
    DataItem,
    DataItemValue,
    create_df_filename,
    get_areas_from_filename,
    get_df_files,
    get_dfs,
    get_filename_start,
    get_filename_start_with_areas,
    get_next_hour,
    print_df_data,
    set_utc,
)
import streamlit as st
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError
import pandas as pd


FetchFunction: TypeAlias = (
    Callable[
        [EntsoePandasClient, pd.Timestamp, pd.Timestamp, AreaValue],
        pd.DataFrame,
    ]
    | Callable[
        [EntsoePandasClient, pd.Timestamp, pd.Timestamp, AreaValue, AreaValue],
        pd.DataFrame,
    ]
    | None
)


def get_today():
    today = pd.Timestamp.now()
    today = today.replace(hour=0, minute=0, second=0, microsecond=0)
    return today


def get_five_years_back_from_start_of_year():
    today = get_today()
    start_of_current_year = pd.Timestamp(year=today.year, month=1, day=1)
    five_years_back_from_start_of_year = start_of_current_year - pd.DateOffset(
        years=5
    )
    return pd.Timestamp(five_years_back_from_start_of_year, 'UTC')


def fetch_data(
    data_item: DataItemValue | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    area_from: AreaValue | None = None,
    area_to: AreaValue | None = None,
    dir_path: str = DATA_DIR,
    return_df: bool = False,
    no_update: bool = False,
) -> pd.DataFrame:
    def fetch_and_save_df(area):
        if isinstance(area, tuple):
            print(announcement_template.format(area[0], area[1]))
            df = func(client, start, end, area[0], area[1])
        else:
            print(announcement_template.format(area))
            df = func(client, start, end, area)

        if return_df:
            return

        if isinstance(area, tuple):
            if df is not None:
                filepath = save_df_file(
                    df,
                    dir_path,
                    filename_start,
                    start,
                    end,
                    area[0],
                    area[1],
                    set_name=True,
                )
                print(f'data stored to file {filepath}')
            else:
                print(no_data_template.format(area[0], area[1]))
        else:
            if df is not None:
                filepath = save_df_file(
                    df,
                    dir_path,
                    filename_start,
                    start,
                    end,
                    area,
                    set_name=True,
                )
                print(f'data stored to file {filepath}')
            else:
                print(no_data_template.format(area))

    def update_if_exists(area):
        if isinstance(area, tuple):
            filename_start_with_area = get_filename_start_with_areas(
                data_item, area[0], area[1]
            )
            pattern = os.path.join(
                dir_path, f'{filename_start_with_area}*{".pqt"}'
            )
            files = glob.glob(pattern)
            if len(files) > 0:
                for f in files:
                    print(f'updating file {f}')
                    update_data(data_item, area[0], area[1], dir_path)
            else:
                fetch_and_save_df(area)
        else:
            filename_start_with_area = get_filename_start_with_areas(
                data_item, area
            )
            pattern = os.path.join(
                dir_path, f'{filename_start_with_area}*{".pqt"}'
            )
            files = glob.glob(pattern)
            if len(files) > 0:
                for f in files:
                    print(f'updating file {f}')
                    update_data(data_item, area, dir_path=dir_path)
            else:
                fetch_and_save_df(area)

    if data_item is None:
        for d_item in DataItem.list_values():
            fetch_data(d_item, start, end, area_from, area_to)
        sys.exit()

    api_key = st.secrets['entso-e']['api_key']
    client = EntsoePandasClient(api_key=api_key)

    if end is None:
        end = get_today()
    if start is None:
        start = get_five_years_back_from_start_of_year()

    func: FetchFunction = None
    filename_start = ''
    if data_item == DataItem.PRICES.value:
        func = fetch_prices
        area_list = Area.list_bidding_zones()
        filename_start = 'prices'
        announcement_template = 'fetching prices for area {}...'
        no_data_template = 'no price data for area {}'
    elif data_item == DataItem.LOAD.value:
        func = fetch_load
        area_list = Area.list_bidding_zones()
        filename_start = 'load'
        announcement_template = 'fetching load for area {}...'
        no_data_template = 'no load data for area {}'
    elif data_item == DataItem.GENERATION.value:
        func = fetch_generation_per_prod_type
        area_list = Area.list_fetchable()
        filename_start = 'generation_per_prod_type'
        announcement_template = (
            'fetching actual generation per production type for area {}...'
        )
        no_data_template = 'no generation data for area {}'
    elif data_item == DataItem.WATER_RESERVOIRS.value:
        func = fetch_water_reservoirs_and_hydro_storage
        area_list = Area.list_fetchable()
        area_list.remove('EE')
        filename_start = 'water_reservoirs_and_hydro_storage'
        announcement_template = 'fetching the aggregate filling rate of water reservoirs and hydro storage for area {}...'
        no_data_template = 'no fill rate data for area {}'
    elif data_item == DataItem.EXCHANGES.value:
        if area_from is not None and area_to is None:
            raise ValueError(
                'The "area_to" argument must be set for fetching scheduled exchanges.'
            )
        func = fetch_scheduled_exchanges
        area_list = Area.list_exchanges()
        filename_start = 'scheduled_exchanges'
        announcement_template = (
            'fetching scheduled commercial exchanges from area {} to area {}...'
        )
        no_data_template = 'no exchange data for area {}'
    else:
        raise ValueError(f'No query for data item "{data_item}".')

    # set timestamps as UTC (localize or convert depending on df)
    start = set_utc(start)
    end = set_utc(end)

    print(f'Using time range {start} to {end}:')

    if area_from is None:
        for area in area_list:
            if no_update:
                fetch_and_save_df(area)
            else:
                update_if_exists(area)
    else:
        if no_update:
            fetch_and_save_df((area_from, area_to))
        else:
            update_if_exists((area_from, area_to))


def save_df_file(
    df: pd.DataFrame,
    dir_path: str,
    filename_start: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    area_from: AreaValue | None = None,
    area_to: AreaValue | None = None,
    set_name: bool | str = False,
) -> str:
    os.makedirs(dir_path, exist_ok=True)
    filename = create_df_filename(
        filename_start, start, end, area_from, area_to
    )
    filepath = os.path.join(dir_path, filename)
    if set_name:
        if isinstance(set_name, str):
            df.attrs['name'] = set_name
        else:
            df.attrs['name'] = filepath
    df.to_parquet(filepath)
    return filepath


def fetch_prices(
    client: EntsoePandasClient,
    start: pd.Timestamp,
    end: pd.Timestamp,
    area: AreaValue,
    _=None,
):
    try:
        ts = client.query_day_ahead_prices(area, start=start, end=end)
        df = ts.to_frame(name='price')
        df.index.name = 'time'
        df.index = df.index.tz_convert('UTC')
        return df
    except NoMatchingDataError:
        print(f'No matching price data for are {area}.')


def fetch_load(
    client: EntsoePandasClient,
    start: pd.Timestamp,
    end: pd.Timestamp,
    area: AreaValue,
    _=None,
):
    try:
        df = client.query_load_and_forecast(area, start=start, end=end)
        df.index.name = 'time'
        df.index = df.index.tz_convert('UTC')
        return df
    except NoMatchingDataError:
        print(f'No matching load data for are {area}.')


def fetch_generation_per_prod_type(
    client: EntsoePandasClient,
    start: pd.Timestamp,
    end: pd.Timestamp,
    area: AreaValue,
    _=None,
):
    try:
        df = client.query_generation(area, start=start, end=end)
        df.index.name = 'time'
        df.index = df.index.tz_convert('UTC')
        return df
    except NoMatchingDataError:
        print(f'No matching generation data for are {area}.')


def fetch_water_reservoirs_and_hydro_storage(
    client: EntsoePandasClient,
    start: pd.Timestamp,
    end: pd.Timestamp,
    area: AreaValue,
    _=None,
):
    try:
        ts = client.query_aggregate_water_reservoirs_and_hydro_storage(
            area, start=start, end=end
        )
        df = ts.to_frame(name='amount')
        df.index.name = 'time'
        df.index = df.index.tz_convert('UTC')
        return df
    except NoMatchingDataError:
        print(f'No matching fill rate data for are {area}.')


def fetch_scheduled_exchanges(
    client: EntsoePandasClient,
    start: pd.Timestamp,
    end: pd.Timestamp,
    area_from: AreaValue,
    area_to: AreaValue,
):
    try:
        ts = client.query_scheduled_exchanges(
            area_from, area_to, start=start, end=end
        )
        df = ts.to_frame(name='amount')
        df.index.name = 'time'
        df.index = df.index.tz_convert('UTC')
        return df
    except NoMatchingDataError:
        print(
            f'No matching data for exchanges between {area_from} and {area_to}.'
        )


def update_data(
    data_item: DataItemValue | None = None,
    area_from: AreaValue | None = None,
    area_to: AreaValue | None = None,
    dir_path: str = DATA_DIR,
):
    if data_item is None:
        for d_item in DataItem.list_values():
            update_data(d_item, area_from, area_to)
        sys.exit()

    filename_start = get_filename_start_with_areas(
        data_item, area_from, area_to
    )
    files = get_df_files(dir_path, filename_start)
    filename_start = get_filename_start(data_item)

    for f in files:
        file_path = os.path.join(dir_path, f)
        df = pd.read_parquet(file_path)

        areas = get_areas_from_filename(f)

        df_last_date = df.iloc[-1].name
        df_start = df.iloc[0].name
        start = get_next_hour(df_last_date)
        end = set_utc(get_today())

        print(f"file's date range is {df_start} to {df_last_date}")
        if start == end:
            print(f'{f} already up to date')
            continue

        update_df = fetch_data(
            data_item,
            start,
            end,
            area_from,
            area_to,
            return_df=True,
            no_update=True,
        )
        new_df = pd.concat([df, update_df])
        area_from = areas[0] if len(areas) > 0 else area_from
        area_to = areas[1] if len(areas) > 1 else area_to
        file_path = save_df_file(
            new_df,
            dir_path,
            filename_start,
            df_start,
            end,
            area_from,
            area_to,
            set_name=True,
        )
        print(f'data stored to file {file_path}')
        print(f'removing old file {f}')
        if file_path != f:
            os.remove(os.path.join(dir_path, f))
        else:
            raise ValueError(
                'Updated file and the old file have the same name ({f}).'
            )


def show_dfs(
    data_item: DataItemValue | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    area_from: AreaValue | None = None,
    area_to: AreaValue | None = None,
    eda: bool = False,
):
    dir_path = DATA_DIR_EDA if eda else DATA_DIR

    dfs: List[pd.DataFrame] = []
    try:
        if data_item is not None:
            dfs = get_dfs(dir_path, data_item, area_from, area_to)
        else:
            for ditem in DataItem.list_values():
                df_batch = get_dfs(dir_path, ditem, area_from, area_to)
                dfs.extend(df_batch)
    except ValueError:
        pass

    for df in dfs:
        show_df(df, start, end)


def show_df(
    df: pd.DataFrame,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
):
    ident = df.attrs.get('name', f'id={id(df)}')
    print(f'Showing dataframe "{ident}":')
    df_to_use = df
    if start is not None or end is not None:
        print(f'dates: {start} -> {end}')
        df_to_use = df_to_use.loc[start:end]

    print_df_data(df_to_use)
