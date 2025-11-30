from typing import Dict, List, Tuple
import pandas as pd
from predict_np_spot_prices.common import (
    DATA_DIR_ARCHIVED,
    DATA_DIR_PREPROCESSED,
    AreaValue,
    DataCategoryValue,
    get_df_area,
    get_dfs,
    get_tuple_name_pairs,
    rename_tuple_columns,
)


def get_archived_dfs(
    data_item: DataCategoryValue,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    area_from: AreaValue | None = None,
    area_to: AreaValue | None = None,
) -> List[pd.DataFrame]:
    return get_dfs(DATA_DIR_ARCHIVED, data_item, start, end, area_from, area_to)


def get_all_prices():
    df_prices = None
    price_dfs = get_dfs(DATA_DIR_PREPROCESSED, 'prices')
    for df in price_dfs:
        area = get_df_area(df)[0]
        if df_prices is None:
            df_prices = df
            df_prices = df_prices.rename(columns={'price': f'price_{area}'})
        else:
            df_prices[f'price_{area}'] = df['price']
    return df_prices


def process_prices_to_area(df: pd.DataFrame, area: AreaValue):
    price_column = f'price_{area}'
    if price_column in df.columns:
        df_prices_area = df[[price_column]]
        df_prices_area = df_prices_area.rename(columns={price_column: 'price'})
    else:
        date_index = pd.date_range(
            start='2025-01-01', end='2025-01-10', freq='D'
        )
        df_prices_area = pd.DataFrame(index=date_index, columns=['time'])
    return df_prices_area


def process_prices_to_quarterly_means(df: pd.DataFrame) -> pd.DataFrame:
    df_quarterly = df.resample('QE')['price'].mean()
    df_quarterly = df_quarterly.reset_index()
    df_quarterly['year'] = df_quarterly['time'].dt.year
    df_quarterly['quarter'] = 'Q' + df_quarterly['time'].dt.quarter.astype(str)
    df_quarterly = df_quarterly[['year', 'quarter', 'price']]
    return df_quarterly


def process_prices_to_daily_means(df: pd.DataFrame) -> pd.DataFrame:
    df_daily = df.resample('D').mean()
    return df_daily


def get_country_total_flows():
    """
    Returns total import and export values for Sweden and Norwaey since they
    are divided into more than one bidding zone and therefore the totals have
    to be calculated from the bidding zone values.
    """
    external = {
        'SE_1': ['FI', 'NO_4'],
        'SE_2': ['NO_3', 'NO_4'],
        'SE_3': ['DK_1', 'FI', 'NO_1'],
        'SE_4': ['DE_LU', 'DK_2', 'LT', 'PL'],
        'NO_1': ['SE_3'],
        'NO_2': ['DE_LU', 'DK_1', 'NL', 'GB'],
        'NO_3': ['SE_2'],
        'NO_4': ['SE_1', 'SE_2', 'FI'],
        'NO_5': [],
    }
    sweden = ['SE_1', 'SE_2', 'SE_3', 'SE_4']
    norway = ['NO_1', 'NO_2', 'NO_3', 'NO_4', 'NO_5']

    df_totals = None
    for bz in sweden + norway:
        df = get_dfs(DATA_DIR_PREPROCESSED, 'flows', area_from=bz)[0]
        if df_totals is None:
            df_totals = df.copy()
            df_totals = df_totals.drop(columns=df.columns.to_list())
        in_cols = [f'{ext_bz}_in' for ext_bz in external[bz]]
        out_cols = [f'{ext_bz}_out' for ext_bz in external[bz]]
        df_totals[f'{bz}_in'] = df[in_cols].sum(axis=1)
        df_totals[f'{bz}_out'] = df[out_cols].sum(axis=1)
    df_totals = df_totals.fillna(0)

    se_in_cols = [f'{c}_in' for c in sweden]
    se_out_cols = [f'{c}_out' for c in sweden]
    df_totals['SE_in'] = df_totals[se_in_cols].sum(axis=1)
    df_totals['SE_out'] = df_totals[se_out_cols].sum(axis=1)
    no_in_cols = [f'{c}_in' for c in norway]
    no_out_cols = [f'{c}_out' for c in norway]
    df_totals['NO_in'] = df_totals[se_in_cols].sum(axis=1)
    df_totals['NO_out'] = df_totals[se_out_cols].sum(axis=1)

    cols_to_drop = se_in_cols + se_out_cols + no_in_cols + no_out_cols
    df_totals = df_totals.drop(columns=cols_to_drop)

    return df_totals


def get_load_generation() -> pd.DataFrame:
    df_lg = None

    load_dfs = get_dfs(DATA_DIR_PREPROCESSED, 'load')
    for df in load_dfs:
        area = get_df_area(df)[0]
        if df_lg is None:
            df_lg = df
            df_lg = df_lg.drop(columns=['Forecasted Load'])
            df_lg = df_lg.rename(columns={'Actual Load': f'load_{area}'})
        else:
            df_lg[f'load_{area}'] = df['Actual Load']

    generation_dfs = get_dfs(DATA_DIR_PREPROCESSED, 'generation')
    for df in generation_dfs:
        df['sum'] = df.sum(axis=1)
        area = get_df_area(df)[0]
        df_lg[f'generation_{area}'] = df['sum']

    return df_lg


def process_lg_to_area(df_lg: pd.DataFrame, area: AreaValue) -> pd.DataFrame:
    load_column = f'load_{area}'
    generation_column = f'generation_{area}'
    df_lga = df_lg[[load_column, generation_column]]
    df_lga = df_lga.rename(
        columns={load_column: 'load', generation_column: 'generation'}
    )
    return df_lga


def process_to_gle(
    df_gl: pd.DataFrame, df_ne: pd.DataFrame, area: AreaValue
) -> pd.DataFrame:
    load_column = f'load_{area}'
    generation_column = f'generation_{area}'
    df_glda = df_gl[[load_column, generation_column]]
    df_glda = df_glda.rename(
        columns={load_column: 'load', generation_column: 'generation'}
    )

    exchange_column = f'{area}_net'
    if exchange_column in df_ne.columns:
        df_ne = df_ne[[exchange_column]]
        df_ne = df_ne.rename(columns={exchange_column: 'net_exchange'})
        df_glda = pd.concat([df_glda, df_ne], axis=1)
    return df_glda


def process_to_gle_daily_means(
    df_gl: pd.DataFrame, df_ne: pd.DataFrame, area: AreaValue
) -> pd.DataFrame:
    df_glda = process_to_gle(df_gl, df_ne, area)
    return df_glda.resample('D').mean()


def get_all_exchanges() -> pd.DataFrame:
    df_exch = None
    exchange_dfs = get_dfs(DATA_DIR_PREPROCESSED, 'exchanges')
    for df in exchange_dfs:
        area_from, area_to = get_df_area(df)
        areas = ('-').join([area_from, area_to])
        if df_exch is None:
            df_exch = df.copy()
            df_exch = df_exch.rename(columns={'amount': areas})
        else:
            df_exch[areas] = df['amount']
        if area_from == 'FI' or area_to == 'FI':
            if 'FI_net' not in df_exch.columns:
                df_exch['FI_net'] = (
                    df['amount'] if area_to == 'FI' else -df['amount']
                )
            else:
                df_exch['FI_net'] += (
                    df['amount'] if area_to == 'FI' else -df['amount']
                )

    return df_exch


def get_netted_exchanges() -> pd.DataFrame:
    df_exch = None
    exchange_dfs = get_dfs(DATA_DIR_PREPROCESSED, 'exchanges')
    for df in exchange_dfs:
        area_from, area_to = get_df_area(df)
        if df_exch is None:
            df_exch = df.copy()
        if area_from == 'FI' or area_to == 'FI':
            if 'FI_net' not in df_exch.columns:
                df_exch['FI_net'] = (
                    df['amount'] if area_to == 'FI' else -df['amount']
                )
            else:
                df_exch['FI_net'] += (
                    df['amount'] if area_to == 'FI' else -df['amount']
                )

    return df_exch


def process_to_generation_mix(df: pd.DataFrame) -> pd.DataFrame:
    df_mix = df.copy()
    df_mix['sum'] = df_mix.sum(axis=1)

    for col in df_mix.columns:
        if col != 'sum':
            df_mix[col] = df_mix[col] / df_mix['sum'] * 100

    df_mix = df_mix.drop(columns=['sum'])

    return df_mix


def process_to_yearly_average(df: pd.DataFrame) -> pd.DataFrame:
    df_yearly = df.resample('YE').mean()
    df_yearly['year'] = df_yearly.index.year
    return df_yearly


def get_generation_mix(area: AreaValue):
    df = get_dfs(DATA_DIR_PREPROCESSED, 'generation', area_from=area)[0]
    return process_to_generation_mix(df), df


def get_norway_generation_with_tuple_cols():
    """
    Simplifies the tuple formatted column names in the norwegian generation
    data where the tuple columns have been preserved, and forms an array of
    pairs of related column names.

    The returned dataframe is shortened to contain only the day means from the
    beginning of 2024 to the end of the 2025 data. The "Consumption" columns
    are also dropped.

    Returns:
        A tuple of the  and an array of pairs of related column names.
    """
    df = get_archived_dfs('generation', area_from='NO_TUPLE')[0]
    df = rename_tuple_columns(df)

    df = df.resample('D').mean()
    df = df['2024-01-01':]
    df = df.drop(columns=[col for col in df.columns if 'Consumption' in col])

    pairs = get_tuple_name_pairs(df)[0]

    return df, pairs


def process_to_import_export_prices(
    df_exch: pd.DataFrame,
    df_prices: pd.DataFrame,
    df_lg: pd.DataFrame,
):
    df_e = df_exch.rename(
        columns={
            'FI-EE': 'Export to EE',
            'EE-FI': 'Import from EE',
            'FI-SE_1': 'Export to SE1',
            'SE_1-FI': 'Import from SE1',
            'FI-SE_3': 'Export to SE3',
            'SE_3-FI': 'Import from SE3',
        }
    )
    df_e = df_e.drop(columns=['FI_net'])

    df_p = df_prices.rename(
        columns={
            'price_EE': 'Price Difference EE-FI',
            'price_SE_1': 'Price Difference SE1-FI',
            'price_SE_3': 'Price Difference SE3-FI',
        }
    )
    df_p['Price Difference EE-FI'] = (
        df_p['Price Difference EE-FI'] - df_p['price_FI']
    )
    df_p['Price Difference SE1-FI'] = (
        df_p['Price Difference SE1-FI'] - df_p['price_FI']
    )
    df_p['Price Difference SE3-FI'] = (
        df_p['Price Difference SE3-FI'] - df_p['price_FI']
    )
    df_p = df_p.drop(columns=['price_NO_4', 'price_FI'])

    df_lgd = df_lg[['load_FI', 'generation_FI']].copy()
    df_lgd['Supply-Demand Imbalance'] = (
        df_lgd['generation_FI'] - df_lgd['load_FI']
    )
    df_lgd = df_lgd.drop(columns=['load_FI', 'generation_FI'])

    return pd.concat([df_e, df_lgd], axis=1), df_p


def find_cheapest_and_most_expensive_periods(
    df: pd.DataFrame, year: str, n: int = 5
) -> Dict[str, List[pd.Timestamp]]:
    df_year = df.loc[str(year)].copy()
    periods = {
        'cheapest days': [],
        'most expensive days': [],
        'cheapest weeks': [],
        'most expensive weeks': [],
        'cheapest months': [],
        'most expensive months': [],
        'cheapest quarters': [],
        'most expensive quarters': [],
    }
    df_period = df_year.resample('D').mean()
    periods['cheapest days'] = df_period.nsmallest(n, 'price').index.to_list()
    periods['most expensive days'] = df_period.nlargest(
        n, 'price'
    ).index.to_list()

    df_period = df_year.resample('W-MON').mean()
    periods['cheapest weeks'] = df_period.nsmallest(n, 'price').index.to_list()
    periods['most expensive weeks'] = df_period.nlargest(
        n, 'price'
    ).index.to_list()

    df_period = df_year.resample('ME').mean()
    periods['cheapest months'] = df_period.nsmallest(n, 'price').index.to_list()
    periods['most expensive months'] = df_period.nlargest(
        n, 'price'
    ).index.to_list()

    df_period = df_year.resample('QE').mean()
    periods['cheapest quarters'] = df_period.nsmallest(
        n, 'price'
    ).index.to_list()
    periods['most expensive quarters'] = df_period.nlargest(
        n, 'price'
    ).index.to_list()

    options = process_min_max_periods_to_options(periods)

    return periods, options


def ordinal(n):
    """Convert integer n to its ordinal string."""
    if 10 <= n % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f'{n}{suffix}'


def process_min_max_periods_to_options(
    periods: Dict[str, List[pd.Timestamp]],
) -> List[str]:
    cheapest_days = [
        f'{ordinal(i + 1)} cheapest day: {day.strftime("%Y/%m/%d")}'
        for i, day in enumerate(periods['cheapest days'])
    ]
    most_expensive_days = [
        f'{ordinal(i + 1)} most expensive day: {day.strftime("%Y/%m/%d")}'
        for i, day in enumerate(periods['most expensive days'])
    ]

    cheapest_weeks = [
        f'{ordinal(i + 1)} cheapest week: {day.strftime("%Y/%m/%d")} - {(day + pd.Timedelta(weeks=1)).strftime("%Y/%m/%d")}'
        for i, day in enumerate(periods['cheapest weeks'])
    ]
    most_expensive_weeks = [
        f'{ordinal(i + 1)} most expensive week: {day.strftime("%Y/%m/%d")} - {(day + pd.Timedelta(weeks=1)).strftime("%Y/%m/%d")}'
        for i, day in enumerate(periods['most expensive weeks'])
    ]

    cheapest_months = [
        f'{ordinal(i + 1)} cheapest month: {day.strftime("%Y/%m/%d")} - {(day + pd.offsets.MonthEnd(0)).strftime("%Y/%m/%d")}'
        for i, day in enumerate(periods['cheapest months'])
    ]
    most_expensive_months = [
        f'{ordinal(i + 1)} most expensive month: {day.strftime("%Y/%m/%d")} - {(day + pd.offsets.MonthEnd(0)).strftime("%Y/%m/%d")}'
        for i, day in enumerate(periods['most expensive months'])
    ]

    cheapest_quarters = [
        f'{ordinal(i + 1)} cheapest quarter: {day.strftime("%Y/%m/%d")} - {(day + pd.offsets.QuarterEnd(0)).strftime("%Y/%m/%d")}'
        for i, day in enumerate(periods['cheapest quarters'])
    ]
    most_expensive_quarters = [
        f'{ordinal(i + 1)} most expensive quarter: {day.strftime("%Y/%m/%d")} - {(day + pd.offsets.QuarterEnd(0)).strftime("%Y/%m/%d")}'
        for i, day in enumerate(periods['most expensive quarters'])
    ]

    options = (
        cheapest_days
        + most_expensive_days
        + cheapest_weeks
        + most_expensive_weeks
        + cheapest_months
        + most_expensive_months
        + cheapest_quarters
        + most_expensive_quarters
    )
    return options


def get_period_dates_from_option(
    option_text, default
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    if option_text is None:
        return default, default
    range_str = option_text.split(': ')[1]
    date_strs = range_str.split(' - ')
    if len(date_strs) == 0:
        return default, default
    if len(date_strs) > 1:
        return pd.Timestamp(date_strs[0].strip()).to_pydatetime(), pd.Timestamp(
            date_strs[1].strip()
        ).to_pydatetime()
    else:
        return pd.Timestamp(date_strs[0].strip()).to_pydatetime(), pd.Timestamp(
            date_strs[0].strip()
        ).to_pydatetime()
