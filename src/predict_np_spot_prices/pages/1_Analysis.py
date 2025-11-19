from predict_np_spot_prices.pages_common import (
    create_production_structure_chart,
)
import streamlit as st

from predict_np_spot_prices.common import (
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


st.header('Production structure')

area = st.selectbox(
    'Select area',
    options=[
        f'{get_compact_form(code)} - {get_long_name(code)}'
        for code in Area.list_fetchable()
    ],
)

create_production_structure_chart(area)
