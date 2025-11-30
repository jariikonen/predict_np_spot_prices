import os
import pandas as pd
from predict_np_spot_prices.eda import (
    get_all_exchanges,
    get_all_prices,
    get_load_generation,
    process_prices_to_area,
    process_prices_to_daily_means,
)
from predict_np_spot_prices.pages_common import (
    area_selectbox,
    finnish_production_structure_chart,
    import_export_prices_chart,
    load_generation_chart,
    period_selector,
    price_daily_means_chart,
    production_structure_chart_yearly,
    price_quarterly_means_chart,
    basic_data_description,
    split_area_option_text,
    temperature_load_chart,
    temperature_prices_chart,
    wind_generation_chart,
    wind_prices_chart,
)
import streamlit as st

from predict_np_spot_prices.common import (
    DATA_DIR_PREPROCESSED,
    Area,
    get_compact_form,
    get_long_name,
)


st.set_page_config(page_title='Exploratory data analysis', layout='wide')
st.title('Exploratory data analysis')

st.markdown("""
    This is a mini-project I did for the
            [Introduction to Data Science](https://studies.helsinki.fi/courses/course-unit/otm-f1abc596-92c2-43ec-b42e-dc8114b5247d)
            course at the University of Helsinki in the fall of 2025. The purpose of the project
            was to analyze the formation of Finnish electricity spot prices on the Nordpool electricity
            exchange and provide forecasts that could help consumers plan their electricity usage for a
            longer period than just the next day.

    In this section, I will go through the exploratory data analysis I conducted before implementing the
            machine learning model.
""")


st.header('Spot electricity prices')

st.markdown("""
    The spot electricity prices are the property that I am trying to predict. There are many factors that
            affect it, the most important of which I will discuss in more detail below, but first we should
            look at some basic properties of price data.
""")

area = area_selectbox('prices', areas=Area.list_bidding_zones())
area_value, area_name = split_area_option_text(area)
df_prices_all = get_all_prices()
df_prices_area = process_prices_to_area(df_prices_all, area_value)
basic_data_description(df_prices_area)
price_quarterly_means_chart(df_prices_area)

daily_means_df = process_prices_to_daily_means(df_prices_area)
years = daily_means_df.index.year.unique()
year = st.selectbox(
    'Select year',
    options=years,
    key='daily_mean_prices',
)
price_daily_means_chart(daily_means_df, year, area_name)


st.header('Load, generation, import and export')

st.markdown("""
    Load and generation are clearly important factors in determining spot electricity prices. Load represents the demand
            and generation represents the supply. According to basic market logic, higher demand tends to push prices up,
            while higher supply tends to push prices down, though actual prices depend on the interaction of both and other
            market constraints.
            
    In an electricity network it is important to keep load and generation in balance, or many
            kinds of serious problems may arise. For this reason, there exists a balancing electricity market that is
            designed to direct the market participants to balance the situation in real-time. If there is overproduction,
            the imbalance prices are low or even negative, causing the producers to drive down production. On the other hand,
            if there is more demand than generation, the imbalance prices are high, causing producers to bring up reserves.

    Electricity may also be imported when the production outside is cheaper than the local production, and exported when it is
            the other way round, or there is excess production.
            

""")

st.subheader('Load and generation vs. prices')

st.markdown("""
    Here you can browse load and generation data vs. prices for different areas in different years.
            Finland's data contains also the netted commercial exchanges (export is calculated as negative).
""")

df_lg = get_load_generation()
df_exch = get_all_exchanges()
area, year, period = period_selector('lge', df_prices_all, include_area=True)
load_generation_chart(df_lg, df_exch, df_prices_all, area, period)


st.subheader(
    'Finnish imports and exports vs. price and supply-demand imbalance'
)
area, year, period = period_selector(
    'iep', df_prices_all, include_area=True, area_options=['EE', 'SE_1', 'SE_3']
)
import_export_prices_chart(df_exch, df_prices_all, df_lg, area, period)


st.header('Production structure')

st.subheader(
    'Average generation per production type per year in Finland and surrounding areas'
)

area = st.selectbox(
    'Select area',
    options=[
        f'{get_compact_form(code)} - {get_long_name(code)}'
        for code in Area.list_fetchable()
    ],
    key='production',
)
production_structure_chart_yearly(area)

st.subheader('A closer look at Finnish production structure')

year, period = period_selector('prod_structure', df_prices_all)
finnish_production_structure_chart(df_prices_all, period)


st.header('Weather \u2014 Wind and Temperature')

st.subheader('Wind')

# NOTICE! A SPECIFIC FILE IS LOADED HERE (A QUICK FIX BEFORE DL)
df_wind = pd.read_parquet(
    os.path.join(DATA_DIR_PREPROCESSED, 'wind_mean_2020-01-01-2025-11-30.pqt')
)
year, period = period_selector('wind_prices', df_prices_all)
wind_prices_chart(df_wind, df_prices_all, period)
wind_generation_chart(df_wind, df_lg, period)

st.subheader('Temperature')

# NOTICE! A SPECIFIC FILE IS LOADED HERE (A QUICK FIX BEFORE DL)
df_temperature = pd.read_parquet(
    os.path.join(
        DATA_DIR_PREPROCESSED, 'temperature_mean_2020-01-01-2025-11-30.pqt'
    )
)
year, period = period_selector('temperature_prices', df_prices_all)
temperature_prices_chart(df_temperature, df_prices_all, period)
temperature_load_chart(df_temperature, df_lg, period)
