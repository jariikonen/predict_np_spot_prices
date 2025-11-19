import glob
import os
import sys
from typing import List, TypeAlias
from collections.abc import Callable
from predict_np_spot_prices.common import (
    DATA_DIR,
    DF_FILE_EXTENSION,
    Area,
    AreaValue,
    DataCategory,
    DataCategoryValue,
    create_df_filename,
    get_areas_from_filename,
    get_compact_form,
    get_date_range_from_filename,
    get_dfs,
    get_filename_start,
    get_filename_start_from_filename,
    get_filename_start_with_areas,
    get_next_hour,
    print_df_data,
    prompt_yes_no,
    read_df,
    set_utc,
    write_df,
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
    data_category: DataCategoryValue | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    area_from: AreaValue | None = None,
    area_to: AreaValue | None = None,
    dir_path: str = DATA_DIR,
    return_df: bool = False,
    no_update: bool = False,
    keep: bool | None = None,
) -> pd.DataFrame:
    def fetch_and_save_df(area):
        df = pd.DataFrame()
        if isinstance(area, tuple):
            print(
                announcement_template.format(
                    area1=area[0],
                    area2=area[1],
                    start=start,
                    end=end,
                )
            )
            df = func(client, start, end, area[0], area[1])
        else:
            print(
                announcement_template.format(area1=area, start=start, end=end)
            )
            df = func(client, start, end, area)

        if return_df:
            return df

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
                print(f'data stored to file {filepath}\n')
            else:
                print(no_data_template.format(area1=area[0], area2=area[1]))
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
                print(f'data stored to file {filepath}\n')
            else:
                print(no_data_template.format(area1=area))

    def update_if_exists(area):
        if isinstance(area, tuple):
            filename_start_with_area = get_filename_start_with_areas(
                data_category, area[0], area[1]
            )
            # add underscore to separate, e.g., SE from SE1
            filename_start_with_area_sep = filename_start_with_area + '_'
            pattern = os.path.join(
                dir_path, f'{filename_start_with_area_sep}*{DF_FILE_EXTENSION}'
            )
            files = glob.glob(pattern)
            if len(files) > 0:
                for f in files:
                    print(f'updating file {f}')
                    filename = os.path.basename(f)
                    update_df_file(
                        filename, dir_path, data_category, start, end, keep=keep
                    )
            else:
                print(
                    f'no existing file for {filename_start_with_area}, fetching new data:'
                )
                fetch_and_save_df(area)
        else:
            filename_start_with_area = get_filename_start_with_areas(
                data_category, area
            )
            pattern = os.path.join(
                dir_path, f'{filename_start_with_area}*{DF_FILE_EXTENSION}'
            )
            files = glob.glob(pattern)
            if len(files) > 0:
                for f in files:
                    print(f'updating file {f}')
                    filename = os.path.basename(f)
                    update_df_file(
                        filename, dir_path, data_category, start, end, keep=keep
                    )
            else:
                fetch_and_save_df(area)

    if data_category is None:
        for d_category in DataCategory.list_values():
            fetch_data(
                d_category, start, end, area_from, area_to, dir_path, keep=keep
            )
        sys.exit()

    api_key = st.secrets['entso-e']['api_key']
    client = EntsoePandasClient(api_key=api_key)

    if end is None:
        end = get_today()
    if start is None:
        start = get_five_years_back_from_start_of_year()

    func: FetchFunction = None
    filename_start = get_filename_start(data_category)
    if data_category == DataCategory.PRICES.value:
        if area_to is not None:
            raise ValueError(
                'The "area_to" argument cannot be set for fetching prices.'
            )
        func = fetch_prices
        area_list = Area.list_bidding_zones()
        announcement_template = (
            'fetching prices for area {area1} between {start} and {end} ...'
        )
        no_data_template = 'no price data for area {area1}'
    elif data_category == DataCategory.LOAD.value:
        if area_to is not None:
            raise ValueError(
                'The "area_to" argument cannot be set for fetching load.'
            )
        func = fetch_load
        area_list = Area.list_bidding_zones()
        announcement_template = (
            'fetching load for area {area1} between {start} and {end} ...'
        )
        no_data_template = 'no load data for area {area1}'
    elif data_category == DataCategory.GENERATION.value:
        if area_to is not None:
            raise ValueError(
                'The "area_to" argument cannot be set for fetching generation.'
            )
        func = fetch_generation_per_prod_type
        area_list = Area.list_fetchable()
        announcement_template = 'fetching actual generation per production type for area {area1} between {start} and {end} ...'
        no_data_template = 'no generation data for area {area1}'
    elif data_category == DataCategory.WATER_RESERVOIRS.value:
        if area_to is not None:
            raise ValueError(
                'The "area_to" argument cannot be set for fetching water reservoirs.'
            )
        func = fetch_water_reservoirs_and_hydro_storage
        area_list = Area.list_fetchable()
        area_list.remove('EE')
        announcement_template = 'fetching the aggregate filling rate of water reservoirs and hydro storage for area {area1} between {start} and {end} ...'
        no_data_template = 'no fill rate data for area {area1}'
    elif data_category == DataCategory.EXCHANGES.value:
        if area_from is not None and area_to is None:
            raise ValueError(
                'The "area_to" argument must be set for fetching scheduled exchanges.'
            )
        func = fetch_scheduled_exchanges
        area_list = Area.list_exchanges()
        announcement_template = 'fetching scheduled commercial exchanges from area {area1} to area {area2} between {start} and {end} ...'
        no_data_template = 'no exchange data for area {area1} to area {area2}'
    elif data_category == DataCategory.FLOWS.value:
        func = fetch_physical_flows
        if area_to is not None:
            raise ValueError(
                'The "area_to" argument cannot be set for fetching physical flows.'
            )
        area_list = ['FI']
        announcement_template = 'fetching crossborder physical flows for area {area1} between {start} and {end} ...'
        no_data_template = 'no flow data for area {area1}'
    else:
        raise ValueError(f'No query for data category "{data_category}".')

    # set timestamps as UTC (localize or convert depending on df)
    start = set_utc(start)
    end = set_utc(end)

    if not return_df:
        print(f'Fetching {data_category}, using time range {start} to {end}:\n')

    if area_from is None:
        for area in area_list:
            if no_update:
                df = fetch_and_save_df(area)
            else:
                update_if_exists(area)
    else:
        if no_update:
            df = fetch_and_save_df((area_from, area_to))
        else:
            update_if_exists((area_from, area_to))

    if return_df:
        return df


def save_df_file(
    df: pd.DataFrame,
    dir_path: str,
    filename_start: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    area_from: AreaValue | None = None,
    area_to: AreaValue | None = None,
    set_name: bool | str = True,
) -> str:
    """
    Saves the dataframe to a file with a unique name based on the data category
    (DataCategory), time range and areas.

    Args:
        df (pd.DataFrame): Dataframe to be saved.
        dir_path (str): Path to the directory where the file is saved.
        filename_start (str): The starting part of the filename based on the
            data category.
        start (pd.Timestamp): Start timestamp of the data.
        end (pd.Timestamp): End timestamp of the data.
        area_from (AreaValue | None): Area code for "from" area if applicable.
        area_to (AreaValue | None): Area code for "to" area if applicable.
        set_name (bool | str): If True, sets the dataframe's "name" attribute
            to the filepath. If a string is provided, uses that as the name.

    Returns:
        str: The filepath where the dataframe is saved.
    """
    df_processed = df.copy()
    os.makedirs(dir_path, exist_ok=True)
    filename = create_df_filename(
        filename_start, start, end, area_from, area_to
    )
    filepath = os.path.join(dir_path, filename)
    if set_name:
        if isinstance(set_name, str):
            df.attrs['name'] = set_name
        else:
            df.attrs['name'] = filename
    write_df(df_processed, filepath)
    return filepath


def update_df_file(
    filename: str,
    dir_path: str,
    data_category: DataCategoryValue,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    keep: bool | None = None,
):
    def save_df_and_remove_old(new_df: pd.DataFrame):
        filename_start = get_filename_start_from_filename(filename)[0]

        # Water reservoirs data may have gaps, which may lead to no new data
        # for the requested update range.
        if (
            data_category == DataCategory.WATER_RESERVOIRS.value
            and len(new_df) == 0
        ):
            new_df = df.copy()

        new_file_path = save_df_file(
            new_df,
            dir_path,
            filename_start,
            start,
            end,
            area_from,
            area_to,
            set_name=True,
        )
        print(f'data stored to file {new_file_path}\n')

        keep_old = keep if keep is not None else False
        new_df_start = new_df.iloc[0].name
        new_df_last_date = new_df.iloc[-1].name
        if keep_old is None and (
            new_df_start != df_start or new_df_last_date != df_last_date
        ):
            prompt = (
                'The start date differs from previous, should the old data be kept? (y/n): '
                if new_df_start != df_start
                else 'The end date is smaller than the previous, should the old data be kept? (y/n): '
            )
            keep_old = prompt_yes_no(prompt)
        if not keep_old:
            if file_path != new_file_path:
                print(f'removing old file {file_path}')
                os.remove(file_path)

    if start is not None and end is not None and start > end:
        raise ValueError('Start date cannot be after end date.')
    if start is None and end is not None and start == end:
        print('Start date equals end date, nothing to update.\n')
        return

    today = set_utc(get_today())
    if end is not None and end > today:
        raise ValueError('End date cannot be in the future.')
    if end is None:
        end = today

    file_path = os.path.join(dir_path, filename)
    df = read_df(file_path)
    if len(df) == 0:
        print(f'file {file_path} is empty, nothing to update.\n')
        return

    df_last_date = df.iloc[-1].name
    df_start = df.iloc[0].name
    print(f"file's date range is {df_start} to {df_last_date}")

    next_hour = get_next_hour(df_last_date)
    update_start = start
    if start is None or (start >= df_start and start <= df_last_date):
        update_start = next_hour
    if end <= df_last_date:
        update_start = start

    if update_start == end:
        print(f'{file_path} already up to date\n')
        return

    areas = get_areas_from_filename(filename)
    area_from = areas[0] if len(areas) > 0 else None
    area_to = areas[1] if len(areas) > 1 else None

    # There may be gaps in water reservoir fill rate data so that the dates
    # between contain the same value as the last date before the gap.
    # Therefore, df_last_date may not be the same as the range end in the filename.
    _, range_end = get_date_range_from_filename(filename)
    range_end = set_utc(range_end)
    if (
        data_category == DataCategory.WATER_RESERVOIRS.value
        and update_start >= df_last_date
        and update_start <= range_end
    ):
        update_start = range_end

    if (
        update_start == df_start
        and end == next_hour
        or end == df_last_date
        or end == range_end
    ):
        print(
            'requested update range matches existing data range => nothing to update\n'
        )
        return
    if update_start >= df_start and end <= next_hour:
        print(
            'requested update range is inside existing data range => only slicing data'
        )
        new_df = df.loc[start:end]
        save_df_and_remove_old(new_df)
        return

    update_df = fetch_data(
        data_category,
        update_start,
        end,
        area_from,
        area_to,
        return_df=True,
        no_update=True,
    )

    if start is None:
        start = df_start

    # Water reservoirs data may have gaps, which may lead to no new data
    # for the requested update range.
    new_df = pd.DataFrame()
    if update_df is not None:
        # If the start is inside the old df's date range, concatenate
        # update to old data. Otherwise just use the update data.
        new_df = update_df.copy()
        if start >= df_start and update_start <= next_hour:
            new_df = pd.concat([df.loc[start:], update_df])

    save_df_and_remove_old(new_df)


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
        print(f'fetched price data: {df.columns}, {df.shape}')
        return df
    except NoMatchingDataError:
        print(f'No matching price data for area {area}.')


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
        print(f'fetched load data: {df.columns}, {df.shape}')
        return df
    except NoMatchingDataError:
        print(f'No matching load data for area {area}.')


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
        print(f'fetched generation data: {df.columns}, {df.shape}')
        return df
    except NoMatchingDataError:
        print(f'No matching generation data for area {area}.')


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
        print(f'fetched water reservoir data: {df.columns}, {df.shape}')
        return df
    except NoMatchingDataError:
        print(f'No matching fill rate data for area {area}.')


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
        print(f'fetched exchanges data: {df.columns}, {df.shape}')
        return df
    except NoMatchingDataError:
        print(
            f'No matching data for exchanges between {area_from} and {area_to}.'
        )


def fetch_physical_flows(
    client: EntsoePandasClient,
    start: pd.Timestamp,
    end: pd.Timestamp,
    area_from: AreaValue,
    _=None,
):
    try:
        df_export = client.query_physical_crossborder_allborders(
            area_from, start=start, end=end, export=True, per_hour=True
        )
        df_export.index.name = 'time'
        df_export.index = df_export.index.tz_convert('UTC')
        print(f'fetched export data: {df_export.columns}, {df_export.shape}')

        df_import = client.query_physical_crossborder_allborders(
            area_from, start=start, end=end, export=False, per_hour=True
        )
        df_import.index.name = 'time'
        df_import.index = df_import.index.tz_convert('UTC')
        print(f'fetched import data: {df_import.columns}, {df_import.shape}')

        return df_export.join(df_import, lsuffix='_export', rsuffix='_import')

    except NoMatchingDataError:
        print(f'No matching data for physical flows for {area_from}.')


def update_data(
    data_category: DataCategoryValue | None = None,
    area_from: AreaValue | None = None,
    area_to: AreaValue | None = None,
    dir_path: str = DATA_DIR,
    keep: bool | None = None,
):
    def create_glob_pattern():
        filename_start = get_filename_start(data_category)
        area_from_pattern = (
            get_compact_form(area_from) if area_from is not None else '*'
        )
        area_to_pattern = (
            f'-{get_compact_form(area_to)}_' if area_to is not None else ''
        )
        if area_from is not None and area_to is None:
            area_from_pattern += (
                '-' if data_category == DataCategory.EXCHANGES.value else '_'
            )
        pattern = os.path.join(
            dir_path,
            f'{filename_start}_{area_from_pattern}{area_to_pattern}*{DF_FILE_EXTENSION}',
        )
        return pattern

    if data_category is None:
        for d_category in DataCategory.list_values():
            update_data(d_category, area_from, area_to, dir_path, keep=keep)
        sys.exit()

    pattern = create_glob_pattern()
    files = glob.glob(pattern)

    for f in files:
        print(f'Updating file {f}:')
        filename = os.path.basename(f)
        update_df_file(filename, dir_path, data_category, keep=keep)


def show_dfs(
    data_category: DataCategoryValue | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    area_from: AreaValue | None = None,
    area_to: AreaValue | None = None,
    dir_path: str = DATA_DIR,
):
    dfs: List[pd.DataFrame] = []
    try:
        if data_category is not None:
            dfs = get_dfs(
                dir_path, data_category, area_from=area_from, area_to=area_to
            )
        else:
            for dcategory in DataCategory.list_values():
                df_batch = get_dfs(
                    dir_path, dcategory, area_from=area_from, area_to=area_to
                )
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
        df_to_use = df_to_use.loc[start:end]

    print_df_data(df_to_use)
