import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
import folium

from collections import Counter
from streamlit_folium import folium_static

def app():

    def ScattergeoWithTraceMap(df1):

        df1['text'] = df1['Наименование ПС'] + '<br>' + df1['тип ЭОБ']
        # limits = [(0,2),(3,10),(11,20),(21,50),(50,3000)]
        colors = ["royalblue","crimson","lightseagreen","orange","lightgrey"]
        cities = []
        scale = 5000

        fig = go.Figure()
        fig.add_trace(go.Scattergeo(
            # locationmode = 'USA-states',
            lon = df1['lon'],
            lat = df1['lat'],
            hoverinfo = 'text',
            text = df1['text'],
            mode = 'markers',
            showlegend = True,
            marker = dict(
                size = 6,
                # color = 'тип ЭОБ',
                # color = 'rgb(255, 0, 0)',
                line = dict(
                    width = 2,
                    # color = 'rgba(68, 68, 68, 0)'
                )
            )))
        # flight_paths = []
        for i in range(len(df1)):
            fig.add_trace(go.Scattergeo(
                    # locationmode = 'USA-states',
                    lon = [df1['ilon'][i], df1['lon'][i]],
                    lat = [df1['ilat'][i], df1['lat'][i]],
                    mode = 'lines',
                    line = dict(width = 1,color = 'red'),
                    # opacity = float(df_flight_paths['cnt'][i]) / float(df_flight_paths['cnt'].max()),
                )
            )
        fig.update_layout(
            title_text = 'География проектов АО "Профотек"',
            showlegend = False,
            geo = dict(
                scope = 'world',
                # projection_type = 'azimuthal equal area',
                # projection_type="orthographic",
                showcountries = True,
                showland = True,
                landcolor = 'rgb(243, 243, 243)',
                countrycolor = 'rgb(204, 204, 204)',
            ),
        )
        # fig.show()
        st.plotly_chart(fig, use_container_width=True)

    # df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_us_cities.csv')

    def FoliumMap(df1,Option,OptionName):

        if not OptionName:
            OptionName = 'Наименование ПС'
            # Option =

        if df1[OptionName].nunique() != 1:
            m = folium.Map()
        else:
            m = folium.Map(location=[float(df1[df1[OptionName] == Option][['lat']].iloc[0]), float(df1[df1[OptionName] == Option][['lon']].iloc[0])], zoom_start=3)


        colors = ["#c85cdb","#5cc6db","#67db5c","#dbbd5c","#db5c5c"]
        voltages = [110, 220, 330, 500, 1000]
        colors_list = []

        # for ind, row in df1.iterrows():
        #     row['Класс напряжения']

        # colors_list = []
        # for ind, row in df1.iterrows():
        #     if row['Класс напряжения'] <= 110:
        #         colors_list.append("c85cdb")
        #     elif row['Класс напряжения'] > 110 and row['Класс напряжения'] <= 220:
        #         colors_list.append("5cc6db")
        #     elif row['Класс напряжения'] > 220 and row['Класс напряжения'] <= 330:
        #         colors_list.append("67db5c")
        #     elif row['Класс напряжения'] > 330 and row['Класс напряжения'] <= 500:
        #         colors_list.append("dbbd5c")
        #     elif row['Класс напряжения'] > 500 and row['Класс напряжения'] <= 1000:
        #         colors_list.append("db5c5c")


        for i in range(len(df1)):
           # folium.CircleMarker(
           #    location=[df1.iloc[i]['lat'], df1.iloc[i]['lon']],
           #    popup=df1.iloc[i]['Наименование ПС'],
           #    # radius=float(df1.iloc[i]['Класс напряжения']/10),
           #    # color = colors_list[i],
           #    # color= ,
           #    fill=True,
           #    fill_color='#c85cdb'
           # ).add_to(m)

          folium.Marker(
             location=[df1.iloc[i]['lat'], df1.iloc[i]['lon']],
             popup=df1.iloc[i]['Наименование ПС'],
          ).add_to(m)

        folium.Marker(
            location=[55.70875238638579, 37.72187262683586],
            popup="АО Профотек",
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(m)

        folium_static(m)

    def ScattergeoMap(df1):

        SubstationCouter = Counter(df1['название проекта'])

        df_by_Supstation = df1.groupby(by=['название проекта']).first()
        df_by_Supstation['Размер'] = SubstationCouter.values()
        df_by_Supstation['Размер'] = df_by_Supstation['Размер']*5

        fig = px.scatter_geo(df_by_Supstation,
                 lat="lat",
                 lon="lon",
                 color="Класс напряжения",
                 hover_name="тип ЭОБ",
#                      hover_data = ["серийный номер"],
                 text = "Наименование ПС",
                 size= 'Размер'
                 )

        fig.update_layout(
            # title_text = 'География проектов АО "Профотек"',
            showlegend = True,
            geo = dict(
                scope = 'world',
                # projection_type = 'azimuthal equal area',
                showcountries = True,
                showcoastlines = True,
                showsubunits = True,
                showland = True,
                landcolor = 'rgb(243, 243, 243)',
                countrycolor = 'rgb(204, 204, 204)',
            ),
        )

        # fig.show()
        st.plotly_chart(fig, use_container_width=True)


    df1 = pd.read_excel(r'./data/Журнал учета серийных номеров.xlsx',
                    sheet_name='ЭОБ',
                    header=0,
                    usecols=['серийный номер','название проекта','Наименование ПС', 'Принадлежность', 'Класс напряжения', 'Локация','тип ЭОБ', 'дата', 'серийный номер трансформатора, в который входит ЭОБ'],
                   )
    # df1.dropna(inplace=True)
    df1 = df1[df1['Локация'].notna()]
    df1.drop_duplicates(inplace=True)
    df1.reset_index(drop=True,inplace=True)
    # df1
    lst = df1['Локация'].tolist()
    lon = []
    lat = []
    for i in lst:
        lat.append(i.split(',')[0])
        lon.append(i.split(',')[1][1:])
    df1['lon'] = lon
    df1['lat'] = lat

    ilon = []
    ilat = []

    for i in range(len(df1)):
        ilon.append(37.72187262683586)
        ilat.append(55.70875238638579)

    dic = {'ilon':ilon, 'ilat':ilat}
    df_lines = pd.DataFrame(dic)
    df1 = pd.concat([df1, df_lines], axis=1)

    DeviceCouter = Counter(df1['тип ЭОБ'])

    st.write('Общее кол-во приборов по типа:')
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(label="Кол-во ТТЭО:", value=DeviceCouter['ТТЭО'])
    with col2:
        st.metric(label="Кол-во ТТЭО-Г:", value=DeviceCouter['ТТЭО-Г'])
    with col3:
        st.metric(label="Кол-во ДНЕЭ:", value=DeviceCouter['ДНЕЭ'])
    with col4:
        st.metric(label="Кол-во ТТНК:", value=DeviceCouter['ТТНК'])
    with col5:
        st.metric(label="Кол-во ВМ:", value=DeviceCouter['Модуль питания ВМ'])


    # df1.to_excel(r'C:\Users\testingcenter\Downloads\map.xlsx')

    with st.expander('Посмотреть таблицу'):
        st.dataframe(df1)

    SubstationName = ''

    col1, col2 = st.columns(2)
    with col1:
        MapOption = st.radio('Выбирите тип карты для отображения', options=['Folium', 'Scattergeo', 'Scattergeo with trace'])

    with col2:
        if 'Folium' in MapOption:
            FoliumMap(df1, Option=[],OptionName=[])
        elif 'Scattergeo with trace' in MapOption:
            ScattergeoWithTraceMap(df1)
        elif 'Scattergeo' in MapOption:
            ScattergeoMap(df1)

    col1, col2 = st.columns(2)
    with col1:
        OptionList = st.selectbox('Осуществить поиск по: ', options = ['По проекту', 'По наименованию ПС', 'По сер.№', 'По принадлежности'])

    with col2:
        if OptionList == 'По проекту':
            Option = st.selectbox('Список проектов:', options = df1['название проекта'].drop_duplicates())
            OptionName = 'название проекта'
            Selected_df = df1[df1['название проекта'] == Option ]
            Selected_df.set_index('серийный номер', inplace=True)

        elif OptionList == 'По наименованию ПС':
            Option = st.selectbox('Список ПС:', options = df1['Наименование ПС'].drop_duplicates() )
            OptionName = 'Наименование ПС'
            Selected_df = df1[df1['Наименование ПС'] == Option ]
            Selected_df.set_index('серийный номер', inplace=True)

        elif OptionList == 'По сер.№':
            Option = st.selectbox('Список серийных номеров:', options = df1['серийный номер'].drop_duplicates() )
            OptionName = 'серийный номер'
            Selected_df = df1[df1['серийный номер'] == Option ]
            Selected_df.set_index('дата', inplace=True)
            
        elif OptionList == 'По принадлежности':
            Option = st.selectbox('Список проектов:', options = df1['Принадлежность'].drop_duplicates())
            OptionName = 'Принадлежность'
            Selected_df = df1[df1['Принадлежность'] == Option ]
            Selected_df.set_index('серийный номер', inplace=True)

    # SNName = st.selectbox('Серийные номера', options = df1['серийный номер'].drop_duplicates())
    # st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.write(Selected_df.groupby(level=0).first()[['название проекта','Наименование ПС','Класс напряжения', 'тип ЭОБ']])
    with col2:
        if 'Folium' in MapOption:
            FoliumMap(Selected_df,Option,OptionName)
        elif 'Scattergeo with trace' in MapOption:
            pass
            # ScattergeoWithTraceMap(Selected_df)
        elif 'Scattergeo' in MapOption:
            ScattergeoMap(Selected_df)
