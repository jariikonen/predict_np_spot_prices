import datetime
from typing import List, Tuple
import pandas as pd
import streamlit as st
import altair as alt

from predict_np_spot_prices.eda import (
    find_cheapest_and_most_expensive_periods,
    get_period_dates_from_option,
    process_lg_to_area,
    process_prices_to_area,
    process_prices_to_quarterly_means,
    process_to_gle,
    process_to_gle_daily_means,
    process_to_import_export_prices,
    process_to_yearly_average,
)
from predict_np_spot_prices.common import (
    DATA_DIR_PREPROCESSED,
    Area,
    AreaValue,
    get_area_value_from_compact_form,
    get_compact_form,
    get_dfs,
    get_long_name,
)


# a locked color scale for production types
colors = alt.Scale(
    domain=[
        'Biomass',
        'Fossil Coal-derived gas',
        'Fossil Gas',
        'Fossil Hard coal',
        'Fossil Oil',
        'Fossil Oil shale',
        'Fossil Peat',
        'Hydro Pumped Storage',
        'Hydro Run-of-river and poundage',
        'Hydro Water Reservoir',
        'Marine',
        'Nuclear',
        'Other',
        'Other renewable',
        'Solar',
        'Waste',
        'Wind Onshore',
    ],
    range=[
        '#AEAE20',  # biomass - olive green
        '#686868',  # coal-derived gas - dark gray
        '#7F7F7F',  # gas - gray
        '#333333',  # coal - dark gray
        '#000000',  # oil - black
        '#241C07',  # oil shale - dark brown
        '#5A4616',  # peat - brown
        '#134390',  # hydro pumped - dark blue
        '#1C5CCA',  # hydro river - blue
        '#377FFD',  # hydro reservoir - light blue
        '#177478',  # marine - dark turquoise
        '#FF62EA',  # nuclear - pink
        '#F9A861',  # other - orange
        '#FF7F0E',  # other renewable - bright orange
        '#EBEB21',  # solar - light blue
        '#452D11',  # waste - dark brown
        '#CDDFF7',  # wind onshore - light blue
    ],
)


def area_selectbox(
    key: str,
    label: str = 'Select area',
    width: int | str = 'stretch',
    areas: List[AreaValue] = Area.list_fetchable(),
):
    options = [
        f'{get_compact_form(code)} - {get_long_name(code)}' for code in areas
    ]
    return st.selectbox(label=label, options=options, key=key, width=width)


def split_area_option_text(
    option_text: str, compact: str = False
) -> Tuple[str, str]:
    """
    Returns area value and long name from a selectbox option string.
    """
    area_value, area_name = option_text.split(' - ')
    if compact is False:
        area_value = get_area_value_from_compact_form(area_value)
    return area_value, area_name


def production_structure_chart_yearly(
    area_option: str, df: pd.DataFrame | None = None
):
    area_value, area_name = split_area_option_text(area_option)
    if df is None:
        df = get_dfs(DATA_DIR_PREPROCESSED, 'generation', area_from=area_value)[
            0
        ]

    df = process_to_yearly_average(df)

    melted = df.melt('year', var_name='type', value_name='generation')
    chart = (
        alt.Chart(melted)
        .mark_bar()
        .encode(
            x='year:O',
            y=alt.Y(
                'generation:Q',
                stack='normalize',
                axis=alt.Axis(title='Fraction of total generation (%)'),
            ),
            color=alt.Color('type:N', scale=colors),
        )
        .properties(title=f'Generation per production type for {area_name}')
    )

    st.altair_chart(chart, width='stretch', height=400)


def yearly_average_price(df_price: pd.DataFrame, area_option: str):
    _, area_name = split_area_option_text(area_option)
    df = process_to_yearly_average(df_price)

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x='year:O',
            y=alt.Y(
                'price:Q',
                axis=alt.Axis(title='EUR/MWh'),
            ),
        )
        .properties(title=f'Yearly average price for {area_name}')
    )

    st.altair_chart(chart, width='stretch', height=400)


def basic_data_description(df: pd.DataFrame, width: int = 150):
    st.markdown('**Data description**')
    st.dataframe(df.describe(), width=width)


def price_quarterly_means_chart(df: pd.DataFrame):
    plot_df = process_prices_to_quarterly_means(df)
    chart = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X('year:O', axis=alt.Axis(title='Year')),
            xOffset='quarter:N',
            y=alt.Y(
                'price:Q',
                axis=alt.Axis(title='Eur/MWh'),
                scale=alt.Scale(domain=[0, 275]),
            ),
            color=alt.Color('quarter:N', legend=alt.Legend(title='Quarter')),
        )
        .properties(title='Quarterly average prices (Eur/MWh)')
    )
    st.altair_chart(chart, width='stretch')


def price_daily_means_chart(df: pd.DataFrame, year: int, area: str):
    plot_df = df.loc[str(year)]
    plot_df = plot_df.reset_index()
    year_start = plot_df.iloc[0]['time']
    year_end = year_start + pd.offsets.YearEnd(0)
    chart = (
        alt.Chart(plot_df)
        .mark_line()
        .encode(
            x=alt.X(
                'time:T',
                scale=alt.Scale(
                    domain=[
                        year_start,
                        year_end,
                    ]
                ),
                axis=alt.Axis(title='Time', format='%b', tickCount='month'),
            ),
            y=alt.Y(
                'price:Q',
                axis=alt.Axis(title='Eur/MWh'),
                scale=alt.Scale(domain=[-200, 900]),
            ),
        )
        .properties(title=f'Daily average prices (Eur/MWh) in {area} in {year}')
    )
    st.altair_chart(chart, width='stretch')


def load_generation_exchanges_chart(
    df_gl: pd.DataFrame, df_ne: pd.DataFrame, year: str, area_option_text: str
):
    area_value, area_name = split_area_option_text(area_option_text)
    area_df = process_to_gle_daily_means(df_gl, df_ne, area_value)

    net_exchanges_exists = 'net_exchange' in area_df.columns
    title = (
        f'Daily average load, generation and netted commercial exchanges (MW) in {area_name} in {year}'
        if net_exchanges_exists
        else f'Daily average load and generation (MW) in {area_name} in {year}'
    )

    plot_df = area_df.loc[str(year)]
    plot_df = plot_df.rename(
        columns={
            'load': 'Load',
            'generation': 'Generation',
            'net_exchange': 'Net Exchange',
        }
    )
    plot_df = plot_df.reset_index()
    year_start = plot_df.iloc[0]['time']
    year_end = year_start + pd.offsets.YearEnd(0)
    plot_df = plot_df.melt('time', var_name='series', value_name='value')

    scale = (
        alt.Scale(
            domain=['Load', 'Generation', 'Net Exchange'],
            range=['coral', 'steelblue', 'orange'],
        )
        if net_exchanges_exists
        else alt.Scale(
            domain=['Load', 'Generation'],
            range=['coral', 'steelblue'],
        )
    )
    load = (
        alt.Chart(plot_df)
        .mark_line()
        .encode(
            x=alt.X(
                'time:T',
                scale=alt.Scale(
                    domain=[
                        year_start,
                        year_end,
                    ]
                ),
                axis=alt.Axis(title='Time', format='%b', tickCount='month'),
            ),
            y=alt.Y(
                'value:Q',
                axis=alt.Axis(title='MW'),
            ),
            color=alt.Color(
                'series:N',
                title='Domain',
                scale=scale,
            ),
        )
    )
    generation = (
        alt.Chart(plot_df)
        .mark_line()
        .encode(x='time', y='value', color=alt.Color('series:N', scale=scale))
    )
    net_exchange = (
        alt.Chart(plot_df)
        .mark_line()
        .encode(x='time', y='value', color=alt.Color('series:N', scale=scale))
    )
    chart = (load + generation + net_exchange).properties(title=title)
    st.altair_chart(chart, width='stretch')


def load_generation_chart(
    df_gl: pd.DataFrame,
    df_ne: pd.DataFrame,
    df_prices: pd.DataFrame,
    area_option: str,
    period: Tuple[datetime.datetime, datetime.datetime]
    | Tuple[datetime.date, datetime.date],
):
    area_value, area_name = split_area_option_text(area_option)

    mw_df = process_to_gle(df_gl, df_ne, area_value)
    mw_df = pick_and_resample(mw_df, period)
    eur_df = process_prices_to_area(df_prices, area_value)
    eur_df = pick_and_resample(eur_df, period)

    period_presentation = get_presentation(period)
    net_exchanges_exists = 'net_exchange' in mw_df.columns
    title = (
        f'Load, generation and netted commercial exchanges (MW) vs. prices (EUR/MWh) in {area_name} in {period_presentation}'
        if net_exchanges_exists
        else f'Load and generation (MW) vs. prices(EUR/MWh) in {area_name} in {period_presentation}'
    )

    mw_df = mw_df.melt('time', var_name='series', value_name='value')
    eur_df = eur_df.melt('time', var_name='series', value_name='value')
    mw_df['unit'] = 'MW'
    eur_df['unit'] = 'EUR/MWh'
    plot_df = pd.concat([mw_df, eur_df], axis=0)

    scale = alt.Scale(
        domain=['load', 'generation', 'net_exchange', 'price'],
        range=['coral', 'steelblue', 'orange', 'black'],
    )

    period_length = get_period_length(period)
    x_axis = (
        alt.Axis(format='%b', title='Time', tickCount='month')
        if period_length.days > 30
        else alt.Axis(format='%b %-d %H:%M', title='Time', tickCount='day')
    )

    base = alt.Chart(plot_df).encode(x=alt.X('time:T', axis=x_axis))

    mw_lines = (
        base.transform_filter(alt.datum.unit == 'MW')
        .mark_line()
        .encode(
            x=alt.X('time:T', axis=x_axis),
            y=alt.Y('value:Q', axis=alt.Axis(title='MW')),
            color=alt.Color('series:N', title='Domain', scale=scale),
        )
    )

    eur_lines = (
        base.transform_filter(alt.datum.unit == 'EUR/MWh')
        .mark_line()
        .encode(
            x=alt.X('time:T'),
            y=alt.Y('value:Q', axis=alt.Axis(title='EUR/MWh')),
            color=alt.Color('series:N', scale=scale),
        )
    )
    chart = (
        (mw_lines + eur_lines)
        .resolve_scale(y='independent')
        .properties(title=title)
    )
    st.altair_chart(chart, width='stretch')


def pick_a_period(
    df: pd.DataFrame,
    period: Tuple[datetime.datetime, datetime.datetime]
    | Tuple[datetime.date, datetime.date],
) -> pd.DataFrame:
    new_df = df.loc[str(period[0]) : str(period[1])].copy()
    return new_df.reset_index()


def get_period_length(
    period: Tuple[datetime.datetime, datetime.datetime]
    | Tuple[datetime.date, datetime.date],
):
    return (
        period[1] - period[0]
        if period[0] != period[1]
        else datetime.timedelta(days=1)
    )


def pick_and_resample(
    df: pd.DataFrame,
    period: Tuple[datetime.datetime, datetime.datetime]
    | Tuple[datetime.date, datetime.date],
):
    period_length = get_period_length(period)
    if period_length.days > 31:
        new_df = df.resample('D').mean()
        new_df = pick_a_period(new_df, period)
    else:
        new_df = pick_a_period(df, period)
    return new_df


def get_presentation(
    period: Tuple[datetime.datetime, datetime.datetime]
    | Tuple[datetime.date, datetime.date],
) -> str:
    period_length = get_period_length(period)
    if period_length.days == 365 or period_length.days == 366:
        return f'{period[0].year}'
    elif (
        period[0].year == period[1].year
        and period[0].month == period[1].month
        and period[0].day == period[1].day
    ):
        return f'{period[0].strftime("%Y/%m/%d")}'
    else:
        return f'{period[0].strftime("%Y/%m/%d")} to {period[1].strftime("%Y/%m/%d")}'


def import_export_prices_chart(
    df_exch: pd.DataFrame,
    df_prices: pd.DataFrame,
    df_lg: pd.DataFrame,
    area_option: str,
    period: Tuple[datetime.datetime, datetime.datetime]
    | Tuple[datetime.date, datetime.date],
):
    area_compact, area_name = split_area_option_text(area_option, compact=True)

    mw_df, eur_df = process_to_import_export_prices(df_exch, df_prices, df_lg)
    mw_df = pick_and_resample(mw_df, period)
    eur_df = pick_and_resample(eur_df, period)

    period_presentation = get_presentation(period)

    mw_df = mw_df.melt('time', var_name='series', value_name='value')
    eur_df = eur_df.melt('time', var_name='series', value_name='value')
    mw_df['unit'] = 'MW'
    eur_df['unit'] = 'EUR/MWh'
    plot_df = pd.concat([mw_df, eur_df], axis=0)

    scale = alt.Scale(
        domain=[
            f'Export to {area_compact}',
            f'Import from {area_compact}',
            f'Price Difference {area_compact}-FI',
            'Supply-Demand Imbalance',
        ],
        range=[
            '#3B7A45',  # export
            '#ECBF2D',  # import
            '#4960BB',  # price difference
            '#4D0A44',  # supply-demand imbalance
        ],
    )

    period_length = get_period_length(period)
    x_axis = (
        alt.Axis(format='%b', title='Time', tickCount='month')
        if period_length.days > 30
        else alt.Axis(format='%b %-d %H:%M', title='Time', tickCount='day')
    )

    base = alt.Chart(plot_df).encode(x=alt.X('time:T', axis=x_axis))

    mw_lines = (
        base.transform_filter(alt.datum.unit == 'MW')
        .mark_line()
        .encode(
            x=alt.X('time:T', axis=x_axis),
            y=alt.Y('value:Q', axis=alt.Axis(title='MW')),
            color=alt.Color('series:N', title='Domain', scale=scale),
        )
    )

    eur_lines = (
        base.transform_filter(alt.datum.unit == 'EUR/MWh')
        .mark_line()
        .encode(
            x=alt.X('time:T'),
            y=alt.Y('value:Q', axis=alt.Axis(title='EUR/MWh')),
            color=alt.Color('series:N', scale=scale),
        )
    )
    chart = (
        (mw_lines + eur_lines)
        .resolve_scale(y='independent')
        .properties(
            title=f'Finnish imports from and exports to {area_name} vs. price difference and supply-demand imbalance in {period_presentation}'
        )
    )
    st.altair_chart(chart, width='stretch')


def finnish_production_structure_chart(
    df_price: pd.DataFrame,
    period: Tuple[datetime.datetime, datetime.datetime]
    | Tuple[datetime.date, datetime.date],
):
    df_mix = get_dfs(DATA_DIR_PREPROCESSED, 'generation', area_from='FI')[0]
    df_mix = pick_and_resample(df_mix, period)

    period_presentation = get_presentation(period)

    df_mix = df_mix.melt('time', var_name='series', value_name='value')

    period_length = get_period_length(period)
    x_axis = (
        alt.Axis(format='%b', title='Time', tickCount='month')
        if period_length.days > 30
        else alt.Axis(format='%b %-d %H:%M', title='Time', tickCount='day')
    )

    chart = (
        alt.Chart(df_mix)
        .transform_window(
            total_value='sum(value)', frame=[None, None], groupby=['time']
        )
        .transform_calculate(
            # calculate normalized value (fraction) for the tooltip
            percent_value=alt.datum.value / alt.datum.total_value
        )
        .mark_area()
        .encode(
            x=alt.X('time:T', axis=x_axis),
            y=alt.Y(
                'value:Q',
                stack='normalize',
                axis=alt.Axis(title='Fraction of total generation (%)'),
            ),
            color=alt.Color('series:N', title='Domain', scale=colors),
            tooltip=[
                alt.Tooltip('series:N'),
                alt.Tooltip('value:Q', title='MW'),
                alt.Tooltip(
                    'percent_value:Q',
                    title='Fraction',
                    format='.1%',
                ),
            ],
        )
        .properties(
            title=f'Finnish production structure in {period_presentation}',
        )
    )
    st.altair_chart(chart, width='stretch', height=400)


def wind_prices_chart(
    df_wind: pd.DataFrame,
    df_prices: pd.DataFrame,
    period: Tuple[datetime.datetime, datetime.datetime]
    | Tuple[datetime.date, datetime.date],
):
    wind_df = pick_and_resample(df_wind, period)
    eur_df = process_prices_to_area(df_prices, 'FI')
    eur_df = pick_and_resample(eur_df, period)

    period_presentation = get_presentation(period)

    wind_df = wind_df.melt('time', var_name='series', value_name='value')
    eur_df = eur_df.melt('time', var_name='series', value_name='value')
    wind_df['unit'] = 'm/s'
    eur_df['unit'] = 'EUR/MWh'
    plot_df = pd.concat([wind_df, eur_df], axis=0)

    scale = alt.Scale(
        domain=['avg_speed', 'price'],
        range=['steelblue', 'black'],
    )

    period_length = get_period_length(period)
    x_axis = (
        alt.Axis(format='%b', title='Time', tickCount='month')
        if period_length.days > 30
        else alt.Axis(format='%b %-d %H:%M', title='Time', tickCount='day')
    )

    base = alt.Chart(plot_df).encode(x=alt.X('time:T', axis=x_axis))

    wind_lines = (
        base.transform_filter(alt.datum.unit == 'm/s')
        .mark_line()
        .encode(
            x=alt.X('time:T', axis=x_axis),
            y=alt.Y('value:Q', axis=alt.Axis(title='m/s')),
            color=alt.Color('series:N', title='Domain', scale=scale),
        )
    )

    eur_lines = (
        base.transform_filter(alt.datum.unit == 'EUR/MWh')
        .mark_line()
        .encode(
            x=alt.X('time:T'),
            y=alt.Y('value:Q', axis=alt.Axis(title='EUR/MWh')),
            color=alt.Color('series:N', scale=scale),
        )
    )
    chart = (
        (wind_lines + eur_lines)
        .resolve_scale(y='independent')
        .properties(
            title=f'Average wind speed (m/s) vs. price during {period_presentation}'
        )
    )
    st.altair_chart(chart, width='stretch')


def wind_generation_chart(
    df_wind: pd.DataFrame,
    df_lg: pd.DataFrame,
    period: Tuple[datetime.datetime, datetime.datetime]
    | Tuple[datetime.date, datetime.date],
):
    wind_df = pick_and_resample(df_wind, period)
    gen_df = process_lg_to_area(df_lg, 'FI')
    gen_df = pick_and_resample(gen_df, period)
    gen_df = gen_df.drop(columns=['load'])
    period_presentation = get_presentation(period)

    wind_df = wind_df.melt('time', var_name='series', value_name='value')
    gen_df = gen_df.melt('time', var_name='series', value_name='value')
    wind_df['unit'] = 'm/s'
    gen_df['unit'] = 'MW'
    plot_df = pd.concat([wind_df, gen_df], axis=0)

    scale = alt.Scale(
        domain=['avg_speed', 'generation'],
        range=['steelblue', '#271575'],
    )

    period_length = get_period_length(period)
    x_axis = (
        alt.Axis(format='%b', title='Time', tickCount='month')
        if period_length.days > 30
        else alt.Axis(format='%b %-d %H:%M', title='Time', tickCount='day')
    )

    base = alt.Chart(plot_df).encode(x=alt.X('time:T', axis=x_axis))

    wind_lines = (
        base.transform_filter(alt.datum.unit == 'm/s')
        .mark_line()
        .encode(
            x=alt.X('time:T', axis=x_axis),
            y=alt.Y('value:Q', axis=alt.Axis(title='m/s')),
            color=alt.Color('series:N', title='Domain', scale=scale),
        )
    )

    mw_lines = (
        base.transform_filter(alt.datum.unit == 'MW')
        .mark_line()
        .encode(
            x=alt.X('time:T'),
            y=alt.Y('value:Q', axis=alt.Axis(title='MW')),
            color=alt.Color('series:N', scale=scale),
        )
    )
    chart = (
        (wind_lines + mw_lines)
        .resolve_scale(y='independent')
        .properties(
            title=f'Average wind speed (m/s) vs. generation {period_presentation}'
        )
    )
    st.altair_chart(chart, width='stretch')


def temperature_prices_chart(
    df_temperature: pd.DataFrame,
    df_prices: pd.DataFrame,
    period: Tuple[datetime.datetime, datetime.datetime]
    | Tuple[datetime.date, datetime.date],
):
    temp_df = pick_and_resample(df_temperature, period)
    eur_df = process_prices_to_area(df_prices, 'FI')
    eur_df = pick_and_resample(eur_df, period)

    period_presentation = get_presentation(period)

    temp_df = temp_df.melt('time', var_name='series', value_name='value')
    eur_df = eur_df.melt('time', var_name='series', value_name='value')
    temp_df['unit'] = '°C'
    eur_df['unit'] = 'EUR/MWh'
    plot_df = pd.concat([temp_df, eur_df], axis=0)

    scale = alt.Scale(
        domain=['temperature_mean', 'price'],
        range=['coral', 'black'],
    )

    period_length = get_period_length(period)
    x_axis = (
        alt.Axis(format='%b', title='Time', tickCount='month')
        if period_length.days > 30
        else alt.Axis(format='%b %-d %H:%M', title='Time', tickCount='day')
    )

    base = alt.Chart(plot_df).encode(x=alt.X('time:T', axis=x_axis))

    temp_lines = (
        base.transform_filter(alt.datum.unit == '°C')
        .mark_line()
        .encode(
            x=alt.X('time:T', axis=x_axis),
            y=alt.Y('value:Q', axis=alt.Axis(title='°C')),
            color=alt.Color('series:N', title='Domain', scale=scale),
        )
    )

    eur_lines = (
        base.transform_filter(alt.datum.unit == 'EUR/MWh')
        .mark_line()
        .encode(
            x=alt.X('time:T'),
            y=alt.Y('value:Q', axis=alt.Axis(title='EUR/MWh')),
            color=alt.Color('series:N', scale=scale),
        )
    )
    chart = (
        (temp_lines + eur_lines)
        .resolve_scale(y='independent')
        .properties(
            title=f'Mean temperature (°C) vs. price during {period_presentation}'
        )
    )
    st.altair_chart(chart, width='stretch')


def temperature_load_chart(
    df_temperature: pd.DataFrame,
    df_lg: pd.DataFrame,
    period: Tuple[datetime.datetime, datetime.datetime]
    | Tuple[datetime.date, datetime.date],
):
    temp_df = pick_and_resample(df_temperature, period)
    gen_df = process_lg_to_area(df_lg, 'FI')
    gen_df = pick_and_resample(gen_df, period)
    gen_df = gen_df.drop(columns=['generation'])

    period_presentation = get_presentation(period)

    temp_df = temp_df.melt('time', var_name='series', value_name='value')
    gen_df = gen_df.melt('time', var_name='series', value_name='value')
    temp_df['unit'] = '°C'
    gen_df['unit'] = 'MW'
    plot_df = pd.concat([temp_df, gen_df], axis=0)

    scale = alt.Scale(
        domain=['temperature_mean', 'load'],
        range=['coral', '#271575'],
    )

    period_length = get_period_length(period)
    x_axis = (
        alt.Axis(format='%b', title='Time', tickCount='month')
        if period_length.days > 30
        else alt.Axis(format='%b %-d %H:%M', title='Time', tickCount='day')
    )

    base = alt.Chart(plot_df).encode(x=alt.X('time:T', axis=x_axis))

    temp_lines = (
        base.transform_filter(alt.datum.unit == '°C')
        .mark_line()
        .encode(
            x=alt.X('time:T', axis=x_axis),
            y=alt.Y('value:Q', axis=alt.Axis(title='°C')),
            color=alt.Color('series:N', title='Domain', scale=scale),
        )
    )

    mw_lines = (
        base.transform_filter(alt.datum.unit == 'MW')
        .mark_line()
        .encode(
            x=alt.X('time:T'),
            y=alt.Y('value:Q', axis=alt.Axis(title='MW')),
            color=alt.Color('series:N', scale=scale),
        )
    )
    chart = (
        (temp_lines + mw_lines)
        .resolve_scale(y='independent')
        .properties(
            title=f'Mean temperature (°C) vs. load during {period_presentation}'
        )
    )
    st.altair_chart(chart, width='stretch')


def get_year_start_end(df: pd.DataFrame, year: int):
    df_year = df.loc[str(year)]
    year_start = df_year.iloc[0].name
    year_end = df_year.iloc[-1].name
    return year_start.to_pydatetime().date(), year_end.to_pydatetime().date()


def get_date_range(df: pd.DataFrame):
    start = pd.Timestamp(df.iloc[0].name).to_pydatetime().date()
    end = pd.Timestamp(df.iloc[-1].name).to_pydatetime().date()
    return start, end


def period_selector(
    key: str,
    df_prices_all: pd.DataFrame,
    include_area: bool = False,
    area_options: List[AreaValue] = Area.list_fetchable(),
) -> (
    Tuple[int, Tuple[datetime.datetime, datetime.datetime]]
    | Tuple[str, int, Tuple[datetime.datetime, datetime.datetime]]
):
    df_prices_FI = process_prices_to_area(df_prices_all, 'FI')
    range_start, range_end = get_date_range(df_prices_FI)
    with st.container(horizontal=True):
        if include_area:
            area = area_selectbox(
                f'{key}_area',
                label='Select export/import area',
                width=200,
                areas=area_options,
            )

        selectbox_year = f'{key}_year'
        selectbox_period = f'{key}_select_period'
        date_picker = f'{key}_date_picker'

        # -- selectbox for year -- start->
        years = df_prices_all.index.year.unique().to_list()

        def on_year_change():
            selected_year = st.session_state[selectbox_year]
            st.session_state[selectbox_period] = None
            st.session_state[date_picker] = get_year_start_end(
                df_prices_FI, selected_year
            )

        if 'selected_year' not in st.session_state:
            st.session_state.selected_year = years[0]

        year = st.selectbox(
            'Select year',
            options=years,
            key=selectbox_year,
            width=150,
            on_change=on_year_change,
        )
        # <- end -- selectbox for year --

        # selectbox for number
        n = st.selectbox(
            'Select number', key=f'{key}_number', options=[5, 10, 25], width=100
        )

        # -- selectbox for preselected periods -- start->
        if year is None:
            year = str(range_start.year)
        _, select_period_options = find_cheapest_and_most_expensive_periods(
            df_prices_FI, year, n
        )

        def on_select_period_change():
            selected_label = st.session_state[selectbox_period]
            st.session_state['select_period_triggered_date_picker_change'] = (
                True
            )
            if selected_label is None:
                st.session_state[date_picker] = get_year_start_end(
                    df_prices_FI, year
                )
                return

            index = select_period_options.index(selected_label)
            period_selected = select_period_options[index]
            st.session_state['select_period_index'] = index
            st.session_state[date_picker] = get_period_dates_from_option(
                period_selected, default=(range_start, range_end)
            )

        if 'selected_period_index' not in st.session_state:
            st.session_state['selected_period_index'] = None

        period_selected = st.selectbox(
            'Select preset',
            options=select_period_options,
            key=selectbox_period,
            width=400,
            index=st.session_state.selected_period_index,
            on_change=on_select_period_change,
        )
        # <- end -- selectbox for preselected periods --

        period_selected_dates = (
            get_year_start_end(df_prices_FI, year)
            if period_selected == select_period_options[0]
            or period_selected is None
            else get_period_dates_from_option(
                period_selected, default=(range_start, range_end)
            )
        )

        # -- date picker for manual date selection -- start ->
        def on_date_change():
            if (
                'select_period_triggered_date_picker_change' in st.session_state
                and st.session_state[
                    'select_period_triggered_date_picker_change'
                ]
                is True
            ):
                st.session_state[
                    'select_period_triggered_date_picker_change'
                ] = False
            else:
                st.session_state[selectbox_period] = None

        if date_picker not in st.session_state:
            st.session_state[date_picker] = period_selected_dates

        period_manual = st.date_input(
            'Select period manually',
            key=date_picker,
            min_value=range_start,
            max_value=range_end,
            on_change=on_date_change,
        )
        # period_manual + timedelta
        # <- end -- date picker for manual date selection --

        period_to_use = period_selected_dates
        if len(period_manual) > 1 and (
            period_manual[0] != period_selected_dates[0]
            or period_manual[1] != period_selected_dates[1]
        ):
            period_to_use = period_manual

        # Because the dates returned by date picker do not have time of day,
        # they are both given time 00:00. This leads the period to become one
        # day too short in timedelta. Therefore, we add time of day (23:59 as
        # the end) to them. Periods in the selector box SHOULD BE ADJUSTED to
        # this (THIS IS STILL TODO).
        p = period_to_use
        period_to_use = (
            datetime.datetime(p[0].year, p[0].month, p[0].day, 0, 0),
            datetime.datetime(p[1].year, p[1].month, p[1].day, 23, 59),
        )

        if include_area:
            return area, year, period_to_use
        return year, period_to_use
