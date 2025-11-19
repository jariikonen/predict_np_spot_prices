from predict_np_spot_prices.pages_common import production_structure_chart
import streamlit as st

from predict_np_spot_prices.eda import (
    get_archived_dfs,
    get_norway_generation_with_tuple_cols,
    process_to_generation_mix_df,
)

st.set_page_config(page_title='Data cleaning', layout='wide')
st.title('Data cleaning')

st.markdown("""
    This is a mini-project I did for the
            [Introduction to Data Science](https://studies.helsinki.fi/courses/course-unit/otm-f1abc596-92c2-43ec-b42e-dc8114b5247d)
            course at the University of Helsinki in the fall of 2025. The purpose of the project
            was to analyze the formation of Finnish electricity spot prices on the Nordpool electricity
            exchange and provide forecasts that could help consumers plan their electricity usage for a
            longer period than just the next day.
            
    In this section, I will go through some of the slightly more specialized data cleaning steps I had
            to perform on the data.
""")

st.header('Electricity market data from ENTSO-E')

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

st.subheader('Generation data')

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
            year 2024. All means go to zero for year 2025. (*Update November 19: To almost
            zero.*)
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
            simply named columns, new single-level columns with tuple like string representations
            of multi-level columns as their names. As I mentioned, I have used
            [entsoe-py](https://github.com/EnergieID/entsoe-py) library for fetching the data.
            I have not researched wether the issue was in the original data or in the library
            parsing the data from XML. However, I do not remember to have updated the library version
            after fetching the problematic data. As of at least November 13, 2025, data frames seem
            to now use multi-level columns. In my preprocessing, I will convert these to single-level
            columns and remove the consumption columns.
    
    **Update November 19, 2025**: It seems that the columns are still single-level with tuple-like
            names instead of multi-level. And it also seems that now the new data comes in the
            simply named columns as of November 14. Therefore, I have modified the preprocessing to merge
            the tuple named columns into the simply named columns using `pd.DataFrame.combine_first`
            method and not just by a simple cutoff date. I have also noticed that also Finnish
            generation data sometimes has these tuple columns.
""")
