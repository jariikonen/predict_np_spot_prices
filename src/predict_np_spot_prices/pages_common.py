import streamlit as st
import altair as alt

from predict_np_spot_prices.eda import (
    get_generation_mix_df,
)
from predict_np_spot_prices.common import (
    get_area_value_from_compact_form,
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


def production_structure_chart(area, df):
    # melt data for Altair
    melted = df.melt('year', var_name='type', value_name='percentage')

    # Altair stacked area chart
    chart = (
        alt.Chart(melted)
        .mark_bar()
        .encode(
            x='year:O',
            y='percentage:Q',
            color=alt.Color('type:N', scale=colors),
        )
        .properties(
            title=f'Generation per production type for {get_long_name(area[:2])}'
        )
    )

    st.altair_chart(chart, width='stretch')


def create_production_structure_chart(area):
    df = get_generation_mix_df(
        get_area_value_from_compact_form(area.split(' ', 1)[0])
    )
    print(df.attrs['name'])
    production_structure_chart(area, df)
