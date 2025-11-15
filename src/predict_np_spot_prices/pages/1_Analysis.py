import streamlit as st
import altair as alt

from predict_np_spot_prices.eda import (
    get_archived_dfs,
    get_generation_mix_df,
    get_norway_generation_with_tuple_cols,
    process_to_generation_mix_df,
)
from predict_np_spot_prices.common import (
    Area,
    get_country,
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


def production_structure_chart(area, df):
    # melt data for Altair
    melted = df.melt('year', var_name='type', value_name='percentage')

    # Altair stacked area chart
    chart = (
        alt.Chart(melted)
        .mark_area()
        .encode(
            x='year:O',
            y='percentage:Q',
            color=alt.Color('type:N', scale=colors),
        )
        .properties(
            title=f'Generation per production type for {get_country(area[:2])}'
        )
    )

    st.altair_chart(chart, width='stretch')


def create_production_structure_chart(area):
    df = get_generation_mix_df(area[:2])
    print(df.attrs['name'])
    production_structure_chart(area, df)


st.set_page_config(page_title='Exploratory data analysis', layout='wide')
st.title('Exploratory data analysis')

st.markdown("""
    This is a mini-project I did for the
            [Introduction to Data Science](https://studies.helsinki.fi/courses/course-unit/otm-f1abc596-92c2-43ec-b42e-dc8114b5247d)
            course at the University of Helsinki in the fall of 2025. The purpose of the project
            is to predict the spot electricity prices in Finland. In this section, I will
            go through the exploratory data analysis I conducted before implementing the
            machine learning model. I also describe some of the slightly more specialized
            data cleaning steps I had to perform on the data.
""")


st.header('About the data')

st.markdown("""
    ENTSO-E [Transparency platform](https://transparency.entsoe.eu/) provides data on
            generation, transportation, consumption and prices of electricity in Europe.
            I started by downloading datasets related to generation, load, prices and
            transmission of electricity between Finland and the neighbouring areas. These
            were the particular datasets that I used:
    
    - [Energy Prices [12.1.D]](https://transparencyplatform.zendesk.com/hc/en-us/articles/16647234190100-Energy-Prices-12-1-D)
    - [Day-ahead & Actual Total Load Per Bidding Zone [6.1.A] & [6.1.B]](https://transparencyplatform.zendesk.com/hc/en-us/articles/16647979768084-Day-ahead-Actual-Total-Load-Per-Bidding-Zone-6-1-A-6-1-B)
    - [Actual Generation per Production Type [TR 16.1.B&C]](https://transparencyplatform.zendesk.com/hc/en-us/articles/16648290299284-Actual-Generation-per-Production-Type-16-1-B-C))
    - [Scheduled Commercial Exchanges [12.1.F]](https://transparencyplatform.zendesk.com/hc/en-us/articles/16583853635092-Scheduled-Commercial-Exchanges-12-1-F)
    
    For accessing the data I used the Python library [EnergieID/entsoe-py](https://github.com/EnergieID/entsoe-py)
            which loads the data through [ENTSO-E REST API](https://transparencyplatform.zendesk.com/hc/en-us/articles/15692855254548-Sitemap-for-Restful-API-Integration).
""")

st.subheader('Data cleaning')

st.markdown("""
    First, I wanted to look at the production structure in the areas that were exchanging
            electricity with Finland. I soon noticed, that Norway's generation data
            contains peculiarities where one production category has data in multiple
            columns. For example, for the category "Hydro Pumped Storage" there are three
            columns: "Hydro Pumped Storage", "('Hydro Pumped Storage', 'Actual Aggregated')"
            and "('Hydro Pumped Storage', 'Actual Consumption')". The latter two column names
            seem to be in a Python tuple format. I could not find a clear reason for this
            from the documentation, so I had to look at the data and figure out how to
            process it.

    By looking at the plot of yearly means containing only the columns with the the normal
            names (not the tuple formatted), we can see that the data seems to end after
            year 2024. All means go to zero for year 2025.
""")

df = get_archived_dfs('generation', 'NO_TUPLE')[0]
df = process_to_generation_mix_df(df)
production_structure_chart('NO', df)

st.markdown("""
    If we look at the plots of simply named columns against the "Actual Aggregated" tuple
            columns at the turn of 2024 and 2025, we can indeed see, that the data in the
            simply named columns ends at the end of 2024, while values in the tuple named
            columns start at the beginning of the 2025. (Empty columns have been dropped
            from the data below.)
""")

df, pairs = get_norway_generation_with_tuple_cols()
pair_labels = [f'{t}  vs.  {s}' for t, s in pairs]
choice = st.selectbox('Choose a column pair to plot:', pair_labels)

t, s = pairs[pair_labels.index(choice)]
chart_df = df[[t, s]]
chart_df.columns = ['Aggregated (tuple named)', 'Simple']

st.line_chart(chart_df)


st.markdown("""
    Based on the above, it seems that for some reason the naming of the columns has been
            changed from the beginning of the 2025 to tuple format, so that each production
            category can have multiple columns. After the change "Hydro Pumped Storage" and
            "Wind Offshore" have also columns for consumption in addition to the aggregated
            generation. These consumption readings should not affect the generation readings,
            which based on the data
            [description](https://transparencyplatform.zendesk.com/hc/en-us/articles/16648290299284-Actual-Generation-per-Production-Type-16-1-B-C)
            should hold the aggregated generation output. The consumption columns can
            therefore be dropped and the actual aggregated columns can be merged into the
            simply named columns. This allows us to form the proper production mix dataframe
            also for Norway.

    **Update November 13, 2025**: It seems that the problem was caused by the new data being
            originally in [MultiIndex](https://pandas.pydata.org/docs/user_guide/advanced.html)
            format (multi-level columns) but merged so that there ended up being, in addition to
            simply named columns, new single-level columns with multi-level column's tuple like
            representations as their names. As I mentioned, I have used
            [entsoe-py](https://github.com/EnergieID/entsoe-py)) library for fetching the data
            and I have not researched wether the issue was in the original data or in the library
            parsing the data from XML. As of at least November 13, 2025, data frames will now use
            multi-level columns. In my preprocessing, I will convert these to single-level columns
            and remove the consumption columns.
""")


st.header('Production structure')

area = st.selectbox(
    'Select area',
    options=[f'{code} - {get_country(code)}' for code in Area.list_fetchable()],
)

create_production_structure_chart(area)
