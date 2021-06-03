import pathlib
import time

import matplotlib
import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from plotly import graph_objs as go
import matplotlib.pyplot as plt
# pd.set_option('display.max_columns', None)

import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,10
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense
from sklearn.preprocessing import MinMaxScaler

from fbprophet import Prophet
from fbprophet.plot import plot_plotly

# FILE_DIR = pathlib.Path.cwd()
# data1_co2 = FILE_DIR / 'datasets/owid-co2-data.csv'
# data2_co2 = FILE_DIR / 'datasets/GCB2020v18_MtCO2_flat.csv'

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
st.info("Data Information")
df = trim_datasets(data1_co2)
min_y, max_y = get_df_year_mx_mi(df)

# side bar setting
st.sidebar.header('About the data')
select_year = st.sidebar.slider('Year', min_y, max_y, max_y, 1)
select_country = st.sidebar.multiselect('Select Country:', df['country'].unique().tolist())

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

# plot the graph
st.plotly_chart(get_co2_choropleth_map(df, select_year, select_co2_sector))
with st.beta_expander("See explanation"):
     st.write("""
         The chart above shows some numbers I picked for you.
         I rolled actual dice for these, so they're *guaranteed* to
         be random.
     """)
     st.image("https://static.streamlit.io/examples/dice.jpg")

st.plotly_chart(show_bar_top_low_10(df, select_co2_sector, select_year))
st.altair_chart(show_scatter_gdp_vs_pop(data1_co2, select_co2_sector, select_year), use_container_width=True)
st.plotly_chart(show_line_chart(data1_co2), use_container_width=True)
with st.spinner('Please wait, it is generating'):
    st.plotly_chart(animated_bar(data1_co2, select_co2_sector), use_container_width=True)

# LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSTTTTTTTTTTTTTTTTTMMMMMMMMMMMMMMMMMMM

# data1_co2 = data1_co2[data1_co2['country'] == 'World']
# data1_co2 = data1_co2.reset_index()
#
#
#
# data = pd.DataFrame(index=range(0,len(data1_co2)),columns=['year','co2'])
# for i in range(0,len(data)):
#     data["year"][i]=data1_co2["year"][i]
#     data["co2"][i]=data1_co2["co2"][i]
#
# scaler=MinMaxScaler(feature_range=(0,1))
# data.index = data.year
# data.drop("year",axis=1,inplace=True)
# final_data = data.values
# # st.write(len(data))
# train_data=final_data[0:200,:]
# valid_data=final_data[200:,:]
# scaler=MinMaxScaler(feature_range=(0,1))
# scaled_data=scaler.fit_transform(final_data)
# x_train_data,y_train_data=[],[]
# for i in range(60,len(train_data)):
#     x_train_data.append(scaled_data[i-60:i,0])
#     y_train_data.append(scaled_data[i,0])

# x_train_data, y_train = np.array(x_train_data), np.array(y_train_data)
# x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))
# lstm_model=Sequential()
# lstm_model.add(LSTM(units=50,return_sequences=True,input_shape=(np.shape(x_train_data)[1],1)))
# lstm_model.add(LSTM(units=50))
# lstm_model.add(Dense(1))
# model_data=data[len(data)-len(valid_data)-60:].values
# model_data=model_data.reshape(-1,1)
# model_data=scaler.transform(model_data)
#
# lstm_model.compile(loss='mean_squared_error',optimizer='adam')
# lstm_model.fit(x_train_data,y_train_data,epochs=1,batch_size=1,verbose=2)
# X_test=[]
# for i in range(60,model_data.shape[0]):
#     X_test.append(model_data[i-60:i,0])
# X_test=np.array(X_test)
# X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
#
# predicted_stock_price=lstm_model.predict(X_test)
# predicted_stock_price=scaler.inverse_transform(predicted_stock_price)
#
# train_data=data[:200]
# valid_data=data[200:]
# valid_data['Predictions']=predicted_stock_price
# plt.plot(train_data["co2"])
# st.write(plt.plot(valid_data[['co2',"Predictions"]]))

# #prediction
# p_years = st.slider("Years of predcition", 1,10)
# period = p_years * 365
#
# select_country = st.selectbox('Select the Country to be display',df_org['country'].unique().tolist())
#
# #Forecasting
# df_train = df_org[df_org['country'] == 'World']
# df_train = df_train[['year','co2']]
# df_train = df_train.sort_values(by=['year'])

# df_train = df_train.rename(columns={"year":"ds","co2":"y"})
# st.dataframe(df_train)
#
# m = Prophet()
# m.fit(df_train)
# future = m.make_future_dataframe(periods=period)
#
# forecast = m.predict(future)
#
# st.subheader("Forecast Data")
# st.dataframe(forecast)
#
# st.write("Forecast Data")
# fig1 = plot_plotly(m, forecast)
# st.plotly_chart(fig1) #plot the forcast data

# from matplotlib.pylab import rcParams
# rcParams['figure.figsize']=20,10
# from keras.models import Sequential
# from keras.layers import LSTM,Dropout,Dense
# # from sklearn.preprocessing import MinMaxScaler
#
# df_train["year"] = pd.to_datetime(df_train.year, format="%Y")
# df_train = df_train.reset_index(drop=True)
#
# data = pd.DataFrame(index=range(0,len(df_train)),columns=['year','co2'])
# for i in range(0,len(data)):
#     data["year"][i]=df_train['year'][i]
#     data["co2"][i]=df_train["co2"][i]
#
# st.write(data.head())
# st.write(df_train)

# line
# fig_scatter.add_trace(go.Line(
#     mode="markers", x=df_sca["population"], y=df_sca["gdp"],
#     text=df_sca['country'], marker=dict(size=20,colorscale='Viridis')))


# df_sum = df_org[['country', 'co2']]
# df_sum = df_sum.groupby(['country']).sum()
# st.dataframe(df_sum)
# import plotly.express as px
# fig_tree = px.treemap(data_frame=df_sum, values='co2')
# st.plotly_chart(fig_tree)

# import altair as alt
# source = df_org
#
# selection = alt.selection_multi(fields=['series'], bind='legend')
#
# fig_a = alt.Chart(source).mark_area().encode(
#     alt.X('cumulative_co2', axis=alt.Axis(domain=False, format='%Y', tickSize=0)),
#     alt.Y('year', stack='center', axis=None),
#     alt.Color('series:N', scale=alt.Scale(scheme='country')),
#     opacity=alt.condition(selection, alt.value(1), alt.value(0.2))
# ).add_selection(
#     selection
# )
# st.altair_chart(fig_a)

# import plotly.express as px
# df_org = df_org.fillna(0)
# df_org = df_org[df_org['co2'] > 0]
#
# fi_bar = go.Bar(x=df_org["country"][0:10],y=df_org["co2"][0:10], marker=dict(color="co2"), showlegend=True)
#
# st.plotly_chart(fi_bar)


# import plotly.express as px
# import plotly.io as pio
# df1 = pd.read_csv(data1_co2)
# df = df.drop(df[df.score < 50].index)
# st.dataframe(df1)
# df1 = df1[df1['country'] != 'World']
# df1 = df1.fillna(value=0)
# df1 = df1.sort_values(by=['co2'], ascending=False)
# st.write(df1["co2"].max())
# st.write(df1["co2"].min())

# @st.cache
# def plot_animated_bar():
#     fig2 = px.bar(df1, x='co2', y='country', color='country',
#                   animation_frame='year', animation_group='country', hover_name="country",
#                   range_x=[-2,21000])
#     fig2.update_xaxes(rangeslider_visible=True)
#
#     fig2.show()
#     st.plotly_chart(fig2)
# plot_animated_bar()

# df_year = df1['year'].tolist()
# df_year = list(dict.fromkeys(df_year))
# df_year.sort()
#
# dict_keys = []
# index = 1
# for i in range(len(df_year)):
#     dict_keys.append(str(index))
#     index += 1
#
# #['one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen',
# #            'fourteen','fifteen','sixteen','seventeen','eighteen','nineteen','twenty','twentyone','twentytwo',
# #            'twentythree','twentyfour','twentyfive','twentysix','twentyseven','twentyeight']
# #
# # years=[1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,
# #        2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017]
# #
# n_frame={}
#
# for y, d in zip(df_year, dict_keys):
#     dataframe=df1[(df1['year']==y)&(df1['country'])]
#     dataframe=dataframe.nlargest(n=5,columns=['co2'])
#     dataframe=dataframe.sort_values(by=['year','co2'])
#
#     n_frame[d]=dataframe
# pd.set_option('display.max_columns', None)
# # print(dataframe)
# print(n_frame['1'])
#
# fig = go.Figure(
#     data=[
#         go.Bar(
#         x=n_frame['1']['co2'], y=n_frame['1']['country'],orientation='h',
#         text=n_frame['1']['co2'], texttemplate='%{text:.3s}',
#         textfont={'size':18}, textposition='inside', insidetextanchor='middle',
#         width=0.9, marker={'color':n_frame['1']['co2']})
#     ],
#     layout=go.Layout(
#         xaxis=dict(range=[-2, 21000], autorange=False, title=dict(text='co2',font=dict(size=18))),
#         yaxis=dict(range=[-0.5, 5.5], autorange=False,tickfont=dict(size=14)),
#         title=dict(text='Top 5 co2',font=dict(size=28),x=0.5,xanchor='center'),
#         # Add button
#         updatemenus=[dict(
#             type="buttons",
#             buttons=[dict(label="Play",
#                           method="animate",
#                           # https://github.com/plotly/plotly.js/blob/master/src/plots/animation_attributes.js
#                           args=[None,
#                           {"frame": {"duration": 1000, "redraw": True},
#                           "transition": {"duration":250,
#                           "easing": "linear"}}]
#             )]
#         )]
#     ),
#     frames=[
#             go.Frame(
#                 data=[
#                         go.Bar(x=value['co2'], y=value['country'],
#                         orientation='h',text=value['co2'],
#                         marker={'color':value['co2']})
#                     ],
#                 layout=go.Layout(
#                         xaxis=dict(range=[0, 21000], autorange=False),
#                         yaxis=dict(range=[-0.5, 5.5], autorange=False,tickfont=dict(size=14)),
#                         title=dict(text='CO2 Top5 over year: '+str(value['year'].values[0]),
#                         font=dict(size=28))
#                     )
#             )
#         for key, value in n_frame.items()
#     ]
# )
# pio.show(fig)
