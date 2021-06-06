#updated on 6 June 2021
import chart_desciption
import plotly.express as px
import streamlit as st
import pandas as pd

import altair as alt
from plotly import graph_objs as go

# set page layout
st.set_page_config(
    page_title="COMPX532A - Final Project Andrew CHOI",
    page_icon="🌍",
    initial_sidebar_state="expanded"
)


@st.cache
def load_data():
    dataframe = pd.read_csv('https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv')
    return dataframe


# clean the datasets
@st.cache
def trim_datasets(dataframe):
    # trim the dataframe
    df = dataframe.drop(columns=['co2_growth_prct', 'co2_growth_abs', 'consumption_co2',
                                 'consumption_co2_per_capita', 'share_global_co2', 'share_global_cumulative_co2',
                                 'co2_per_gdp', 'consumption_co2_per_gdp', 'co2_per_unit_energy',
                                 'cement_co2_per_capita',
                                 'coal_co2_per_capita', 'flaring_co2_per_capita', 'gas_co2_per_capita',
                                 'oil_co2_per_capita',
                                 'other_co2_per_capita', 'share_global_coal_co2', 'share_global_oil_co2',
                                 'share_global_gas_co2',
                                 'share_global_flaring_co2', 'share_global_cement_co2',
                                 'share_global_cumulative_coal_co2',
                                 'share_global_cumulative_oil_co2', 'share_global_cumulative_gas_co2',
                                 'share_global_cumulative_flaring_co2',
                                 'share_global_cumulative_cement_co2', 'total_ghg', 'ghg_per_capita', 'methane',
                                 'methane_per_capita', 'nitrous_oxide',
                                 'nitrous_oxide_per_capita', 'primary_energy_consumption', 'energy_per_capita',
                                 'energy_per_gdp'])
    # df = df[['iso_code','country','year','co2','co2_per_capita',
    #          'trade_co2','cement_co2','coal_co2','flaring_co2',
    #          'gas_co2','oil_co2','other_industry_co2','cumulative_co2',
    #          'cumulative_coal_co2','cumulative_oil_co2','cumulative_gas_co2',
    #          'cumulative_flaring_co2','cumulative_cement_co2','population','gdp']]
    return df


def get_df_year_mx_mi(dataframe):
    min_year = int(dataframe['year'].min())
    max_year = int(dataframe['year'].max())
    return min_year, max_year


# checkbox to show all data
def check_box_show(df):
    check_box_sd = st.checkbox('Show All Data')
    if check_box_sd:
        st.dataframe(df)
    return


# set up the data frame for choropleth map
def control_dataframe(dataframe):
    df = dataframe.loc[dataframe['year'] == select_year]
    df['population'] = df['population'].map('{:,}'.format)  # format the integer with comma
    df['gdp'] = df['gdp'].map('{:,}'.format)  # format the integer with comma
    df['population'] = df['population'].astype(str)  # change the data type to string for concatenation
    df['gdp'] = df['gdp'].astype(str)  # change the data type to string for concatenation
    df['text'] = df['country'] + '<br>Population: ' + df['population'] + '<br>GDP: ' + df['gdp']
    df_columns_list = df.columns.tolist()
    df_columns_list.remove('iso_code')
    df_columns_list.remove('country')
    df_columns_list.remove('year')
    df_columns_list.remove('population')
    df_columns_list.remove('gdp')
    return df, df_columns_list


@st.cache
def get_co2_choropleth_map(dataframe, year, select_sector):
    fig_choropleth = go.Figure(data=go.Choropleth(
        locations=dataframe['iso_code'],
        z=dataframe[select_sector],
        text=dataframe['text'],
        colorscale='agsunset',
        autocolorscale=False,
        reversescale=True,
        marker_line_color='darkgray',
        marker_line_width=1,
        colorbar_title='CO2 <br> in Country',
    ))

    fig_choropleth.update_layout(
        title_text='<b>' + 'CO2 Emission in ' + str(year) + ' (Sector in ' + select_sector + ')<b>',
        geo=dict(
            showframe=True,
            showcoastlines=True,
            projection_type="orthographic",
            showocean=True,
            oceancolor='deepskyblue',
            showlakes=True,
            lakecolor='lightblue'
        ),
    )

    fig_choropleth.update_layout(margin={"r": 30, "t": 30, "l": 30, "b": 30},
                                 title_font=dict(
                                     size=20,
                                     color='Blue')
                                 )
    return fig_choropleth


@st.cache
def animated_bar(dataframe, dataset_column):
    df_region_area = dataframe[dataframe['iso_code'].isnull()]
    df_region_area = df_region_area.pivot(index='year', columns='country', values=dataset_column)
    # create year as new column, some countries missing year in original datasets
    df_region_area['year'] = df_region_area.index
    # melt the data, now each country has the same year value
    df_region_area = df_region_area.melt(id_vars=['year'], var_name=['country'], value_name=dataset_column)
    cat_max_year = int(df_region_area[select_co2_sector].max()) * 1.1
    # plot an animated bar
    fig_animated = px.bar(df_region_area,
                          x='country',
                          y=dataset_column,
                          color='country',
                          animation_frame='year',
                          animation_group='country',
                          range_y=[0, cat_max_year],
                          height=600,
                          title="<b>Running bar chart of " + dataset_column + "<b>"
                          )
    return fig_animated


# show the bar chart of top 10 / lowest 10
def show_bar_top_low_10(dataframe, data_column, year):
    st.header("Which countries are the Top 10 or Lowest 10?")
    df = dataframe[dataframe['iso_code'].notnull()]
    df = df[df['iso_code'] != 'OWID_WRL']
    df_top10 = df.nlargest(10, data_column)
    df_low10 = df.nsmallest(10, data_column)
    high_or_low = st.radio('Select to display the Highest/Lowest 10 CO2 emission:',
                           ['Highest 10', 'Lowest 10'])  # .radio('Select to Highest 10 / Lowest 10', )
    fig_bar = go.Figure()
    if high_or_low == 'Highest 10':
        fig_bar = go.Figure(data=[go.Bar(
            x=df_top10['country'], y=df_top10[data_column],
            text=df_top10[data_column],
            textposition='auto',
        )])
    elif high_or_low == 'Lowest 10':
        fig_bar = go.Figure(data=[go.Bar(
            x=df_low10['country'], y=df_low10[data_column],
            text=df_low10[data_column],
            textposition='auto',
        )])

    fig_bar.update_traces(marker_color='rgb(51,255,149)', marker_line_color='rgb(8,48,107)',
                          marker_line_width=1.5, opacity=0.6)
    fig_bar.update_layout(title_text='<b>' + high_or_low
                                     + ' countries of CO2 emission in '
                                     + str(year)
                                     + ' (' + data_column + ')' + '<b>',
                          title_font=dict(size=18)
                          )
    return fig_bar


# scatter graph for GDP vs Population vs CO2 emission
def show_scatter_gdp_vs_pop(dataframe, data_column, year):
    # plot the scatter chart (GDP vs Population vs Selected category)
    st.header("Relationship between GDP, Population and CO2 emission")
    df_gdp_pop_cat = dataframe[['year', 'country', 'iso_code', data_column, 'gdp', 'population']]
    df_gdp_pop_cat = df_gdp_pop_cat[df_gdp_pop_cat['year'] == year]  # filter out the year
    df_gdp_pop_cat = df_gdp_pop_cat[df_gdp_pop_cat['iso_code'].notnull()]  # filter out no iso code area
    df_gdp_pop_cat = df_gdp_pop_cat[df_gdp_pop_cat['iso_code'] != 'OWID_WRL']  # filter out world data
    # create an altair chart dictionary
    scatter_chart = alt.Chart(df_gdp_pop_cat).mark_circle().encode(
        x='gdp', y='population',
        size=alt.Size(data_column, scale=alt.Scale(range=[100, 2000])),
        color=alt.Color('country', legend=alt.Legend(columns=2)),
        opacity='country',
        tooltip=['country', 'gdp', 'population', data_column]).properties(
        title='GDP vs Population vs ' + data_column + ' in ' + str(year)
    ).interactive()

    scatter_chart = alt.layer(scatter_chart).configure_title(
        fontSize=20, anchor="middle").configure_view(
        continuousHeight=500
    )
    # st.dataframe(df_gdp_pop_cat)
    return scatter_chart


# plot the line chart
def show_line_chart(dataframe):
    list_country = dataframe['country'].unique().tolist()
    st.header("Time Series of CO2 emission")
    col1, col2 = st.beta_columns(2)
    col1.subheader('Select Country')
    col2.subheader('Select Mode')

    select_list_country = col1.selectbox('', list_country, index=232)
    select_mode = col2.selectbox('', ['Trend', 'Cumulative'])

    df_world = pd.DataFrame()
    if len(select_list_country) == 0:
        df_world = dataframe[dataframe['iso_code'] == 'OWID_WRL']
    elif len(select_list_country) > 0:
        df_world = dataframe[dataframe['country'] == select_list_country]

    fig_line = go.Figure()
    if select_mode == 'Trend':
        fig_line.add_trace(go.Scatter(
            x=df_world['year'],
            y=df_world['co2'],
            name='World CO2',  # Style name/legend entry with html tags
            connectgaps=True,  # override default to connect the gaps
            line_shape='spline',
            fill='tonexty'
        ))
        fig_line.add_trace(go.Scatter(
            x=df_world['year'],
            y=df_world['trade_co2'],
            name='Trade CO2',
            line_shape='spline',
            line=dict(dash='dash'),
            fill='tonexty'
        ))
        fig_line.add_trace(go.Scatter(
            x=df_world['year'],
            y=df_world['cement_co2'],
            name='Cement CO2',
            line_shape='spline',
            line=dict(dash='dash'),
            fill='tonexty'
        ))
        fig_line.add_trace(go.Scatter(
            x=df_world['year'],
            y=df_world['coal_co2'],
            name='Coal CO2',
            line_shape='spline',
            line=dict(dash='dash'),
            fill='tonexty'
        ))
        fig_line.add_trace(go.Scatter(
            x=df_world['year'],
            y=df_world['flaring_co2'],
            name='Flaring CO2',
            line_shape='spline',
            line=dict(dash='dash'),
            fill='tonexty'
        ))
        fig_line.add_trace(go.Scatter(
            x=df_world['year'],
            y=df_world['gas_co2'],
            name='GAS CO2',
            line_shape='spline',
            line=dict(dash='dash'),
            fill='tonexty'
        ))
        fig_line.add_trace(go.Scatter(
            x=df_world['year'],
            y=df_world['oil_co2'],
            name='Oil CO2',
            line_shape='spline',
            line=dict(dash='dash'),
            fill='tonexty'
        ))
        fig_line.add_trace(go.Scatter(
            x=df_world['year'],
            y=df_world['other_industry_co2'],
            name='Other industry CO2',
            line_shape='spline',
            line=dict(dash='dash'),
            fill='tonexty'
        ))
    elif select_mode == 'Cumulative':
        fig_line.add_trace(go.Scatter(
            x=df_world['year'],
            y=df_world['cumulative_co2'],
            name='Cumulative CO2',  # Style name/legend entry with html tags
            connectgaps=True,  # override default to connect the gaps
            line_shape='spline',
            fill='tonexty'
        ))
        fig_line.add_trace(go.Scatter(
            x=df_world['year'],
            y=df_world['cumulative_coal_co2'],
            name='Cumulative Coal CO2',
            line_shape='spline',
            line=dict(dash='dash'),
            fill='tonexty'
        ))
        fig_line.add_trace(go.Scatter(
            x=df_world['year'],
            y=df_world['cumulative_cement_co2'],
            name='Cumulative cement CO2',
            line_shape='spline',
            line=dict(dash='dash'),
            fill='tonexty'
        ))
        fig_line.add_trace(go.Scatter(
            x=df_world['year'],
            y=df_world['cumulative_flaring_co2'],
            name='Cumulative flaring CO2',
            line_shape='spline',
            line=dict(dash='dash'),
            fill='tonexty'
        ))
        fig_line.add_trace(go.Scatter(
            x=df_world['year'],
            y=df_world['cumulative_gas_co2'],
            name='Cumulative GAS CO2',
            line_shape='spline',
            line=dict(dash='dash'),
            fill='tonexty'
        ))
        fig_line.add_trace(go.Scatter(
            x=df_world['year'],
            y=df_world['cumulative_oil_co2'],
            name='Cumulative Oil CO2',
            line_shape='spline',
            line=dict(dash='dash'),
            fill='tonexty'
        ))
    fig_line.update_layout(
        title='<b>Time series of CO2 emission with sub-sectors (' + select_list_country + ') (' + select_mode + ')<b>',
        legend=dict(y=1, font_size=14),
        xaxis=dict(rangeslider=dict(visible=True),
                   title_font=dict(size=20))
    )
    return fig_line


data1_co2 = load_data()

# page layout setting
st.header('COMPX532A - Final Project')
df = trim_datasets(data1_co2)
min_y, max_y = get_df_year_mx_mi(df)

# side bar setting
st.sidebar.header('Setting Option')
select_year = st.sidebar.slider('Year', min_y, max_y, max_y, 1)
select_country = st.sidebar.multiselect('Select Country:', df['country'].unique().tolist())
# st.sidebar.write(len(data1_co2['country'].unique().tolist()))

# filter dataframe by select country
if len(select_country) > 0:
    df = df[df['country'].isin(select_country)]
elif len(select_country) == 0:
    pass
check_box_show(df)
st.header("CO2 Emission in the world")
df, df_columns_list = control_dataframe(df)
df_columns_list.pop()
# add select box of select sector in side bar
select_co2_sector = st.sidebar.selectbox('Select sector in CO2 Emission', df_columns_list)
st.sidebar.write('🧠 Created by Andrew Choi. <br>©Copyright reserved.',unsafe_allow_html=True)

st.info(chart_desciption.data_information())

# plot the graph
st.plotly_chart(get_co2_choropleth_map(df, select_year, select_co2_sector))
with st.beta_expander("🗺️Choropleth Map Explanation"):
     st.markdown(chart_desciption.choropleth_map_explanation(), unsafe_allow_html=True)

#explanation of Bar chart
st.plotly_chart(show_bar_top_low_10(df, select_co2_sector, select_year))
with st.beta_expander("📊 Top 10/ Lowest 10 Bar chart Explanation"):
    st.markdown(chart_desciption.top_low_bar_chart_explanation(), unsafe_allow_html=True)

#explanation of Scatter chart
st.altair_chart(show_scatter_gdp_vs_pop(data1_co2, select_co2_sector, select_year), use_container_width=True)
with st.beta_expander("🔵🔴 Scatter chart (Relationship among CO2, GDP, Population) Explanation"):
    st.markdown(chart_desciption.scatter_chart_explanation(), unsafe_allow_html=True)

#explanation of line chart
st.plotly_chart(show_line_chart(data1_co2), use_container_width=True)
with st.beta_expander("📈 Line chart (trend/accumulated of CO2 emission) Explanation"):
    st.markdown(chart_desciption.line_chart_explanation(), unsafe_allow_html=True)

#Animated bar chart
st.header("Animated Bar chart (shows the volume/change over time)")
with st.spinner('Please wait, it is generating'):
    try:
        st.plotly_chart(animated_bar(data1_co2, select_co2_sector), use_container_width=True)
    except:
        st.error("You have select a sector that no animated graph.")

with st.beta_expander("🏃‍♂️🏃‍♀️📊Animated Bar chart (change over time) Explanation"):
    st.markdown(chart_desciption.animated_bar_explanation(), unsafe_allow_html=True)