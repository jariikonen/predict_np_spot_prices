import os
from typing import List
import pandas as pd
from predict_np_spot_prices.common import (
    DATA_DIR_EDA,
    AreaValue,
    DataItemValue,
    get_dfs,
    get_tuple_name_pairs,
    rename_tuple_columns,
)


DATA_DIR_ARCHIVED = os.path.join(DATA_DIR_EDA, 'archived')


def get_archived_dfs(
    data_item: DataItemValue,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    area_from: AreaValue | None = None,
    area_to: AreaValue | None = None,
) -> List[pd.DataFrame]:
    return get_dfs(DATA_DIR_ARCHIVED, data_item, start, end, area_from, area_to)


def get_generation_df(area: AreaValue):
    return get_dfs(DATA_DIR_EDA, 'generation', area_from=area)[0]


def process_to_generation_mix_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=['sum'])

    yearly_means = df.resample('YE').mean()
    yearly_means['year'] = yearly_means.index.year

    return yearly_means


def get_generation_mix_df(area: AreaValue):
    df = get_generation_df(area)
    return process_to_generation_mix_df(df)


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
