def animated_bar_explanation():
    animated_bar_explain = '''
           <li>The animated bar chart shows the change of (selected) CO2 emission over time.
           Only continents shows in the graph (with some special regions).
           <br><br>
           Select the sector of \"cumulative_XX_co2\" in the sidebar would receive better data exploration.
           Simply click the <b><i>play</i></b> button to start the animation. 
           <br><br>
           From the animated bar chart (in cumulative sector), we observed that Asia rapidly growth in recent year.
           The International Transport suddenly appears since 2017, believing no statistic in previous year.
           <br><br>
           <b>It takes time to generate a new animated bar chart when filter changes in the side bar.</b>
           </li>
           <br>'''
    return animated_bar_explain


def line_chart_explanation():
    line_chart_explain = '''
       <li>The static line chart shows the trend/accumulated CO2 emission across different sectors.
       The default graph shows the trend of CO2 emission but <i>user could change to other country or 
       switch to accumulated mode</i>.
       <br><br>
       The trend mode shows the change over years whereas the accumulated mode shows the 
       speed of CO2 emission over years. From the trend mode, the CO2 emission stayed at stable and low
       volume emission before 1900. It then slightly increase after then. After 1950, the CO2 emission 
       upsurged since then which had not declined until present.
       <br><br>
       A time series filter underneath the line chart allows user to zoom into the time range.
       <br><br>
       <b>The filters at the sidebar are not applicable to this graph.</b>
       </li>
       <br>'''
    return line_chart_explain


def scatter_chart_explanation():
    scatter_chart_explain = '''
       <li>The scatter chart shows <i>the relationship among GDP, Population and CO2 emission</i> 
       (based on the selected sector) in which are are able to find the cluster of the countries.
       <br><br>
       From the scatter chart, we discovered that most of the small bubble scatter around the 
       left bottom corner reflecting that the countries with 
       <b>relatively low GDP and Population contributed small volume of CO2 emission</b>.
       <br><br>
       The chart will dynamically change if filters of countries and year applied.
       </li>
       <br>'''
    return scatter_chart_explain


def top_low_bar_chart_explanation():
    top_low_bar_chart_explain = '''
       <li>The bar chart shows the top/lowest 10 of the selected year about the CO2 mission 
       (based on the selected sector). The chart will dynamically change if filter of countries and year applied.
       </li>
       <br>'''
    return top_low_bar_chart_explain


def choropleth_map_explanation():
    choropleth_map_explain = '''
        <li>The choropleth map above shows all countries of CO2 emission according to the filter 
        at the side bar. You are able to adjust the year slider explore the data in particular year. 
        <b>Multiple countries</b> could also be selected if you would like to explore the target 
        data sets.</li>
        <br>
        <li>In addition, the dataset consists of different CO2 emission sectors, including Oil, Cement,
        Trade, Gas, ect. You could try to <b>select the sector</b> to dive into the data. The map will automatically
        adjust the colour and legend of your choice. 
        </li>
        <br>'''
    return choropleth_map_explain


def data_information():
    data_info = '''The data is from \"Our World in Data\" covering CO2 emission at country level.
    It covers over 250 years CO2 emission data with 236 sovereign state (including continents or 
    special region data) which consists of several sectors of CO2 source including general CO2 
    emission, oil, gas, cement, trade, flaring, coal and those sectors data derived into cumulative 
    figure, and per capita figure. The data is open access under the Creative Commons BY license. Data source:
    
    https://github.com/owid/co2-data
    '''
    return data_info

def project_background():
    project_intro = '''
    Carbon dioxide emission is a primary concern of climate change in the world.
    
    Most of us understand reducing CO2 emission would avoid the deterioration of global 
    warming. However, do you know how human activities impact CO2 emission and how much 
    was your living place or other countries contribute the CO2 emission 200 years ago, 
    100 years ago or 10 years ago?
    
    When we have the historical record, it is time-consuming to mining useful data at the 
    spreadsheet full with a digit and over 23 thousand rows.
    
    The visualisation drives the aforesaid problem by illustrating the annual CO2 emission into 
    a map-based graph with pointing out the severe and minor 10 countries in a particular year. 
    In addition, some graphs help to determine the relationship between GDP, Population and CO2 
    emission and the trend in time series. The last animated bar chart provides a clear view to 
    watch the annual change of CO2 emission across the different continents.
    '''
    return project_intro