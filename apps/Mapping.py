import streamlit as st
import pandas as pd
import numpy as np
# import os
# import plotly.express as px
# import plotly.graph_objects as go

import folium
from folium import plugins
from folium.plugins import HeatMap
from folium.plugins import MarkerCluster

from streamlit_folium import folium_static
import leafmap.foliumap as leafmap

from streamlit_echarts import st_echarts

from collections import Counter

from PIL import Image
from datetime import datetime, date
from st_aggrid import AgGrid, GridUpdateMode
from st_aggrid.grid_options_builder import GridOptionsBuilder



def app():
    # st.sidebar.subheader('Описание:')
    # with st.sidebar.expander("Описание"):
    st.info('Приложение предназначено для анализа развития Цифровых ПС в компании [ПАО «Россети»](https://rosseti.ru). Приложение выполнено в рамках образовательной программы «Лидеры энергетики» совместно с [Skoltech](https://www.skoltech.ru/?lang=ru).')

    def FoliumMap(df1,Option,OptionName,AddOptions):

#         if not OptionName:
#             OptionName = 'Наименование ПС'

#         if df1[OptionName].nunique() != 1:
        m = folium.Map(location=[63, 77], zoom_start=3.25)

        dic = {
            10:'darkblue',
            35:'green',
            110:'orange',
            220:'red',
            330:'blue',
            500:'darkgreen'}

        HeatMapData = []
        Locations = []
        for x,y,w in zip(df1['lat'],df1['lon'],df1['Класс напряжения, кВ']):
            HeatMapData.append([x,y,w])
            Locations.append([str(x),str(y)])

        if 'По классу напряжения' in AddOptions:
            for i in range(len(df1)):
                folium.Marker(
                     location=[df1.iloc[i]['lat'], df1.iloc[i]['lon']],
                     popup=df1.iloc[i]['Наименование ПС'],
                     icon=folium.Icon(color=str(dic[df1.iloc[i]['Класс напряжения, кВ']])),
                  ).add_to(m)
            # folium.LayerControl().add_to(m)

        elif 'Тепловая карта' in AddOptions:
            HeatMap(data=HeatMapData).add_to(m)
            # HeatMap(data=list(zip(df1['lat'].values, df1['lon'].values, df1['Класс напряжения, кВ'].values ))).add_to(m)
            # folium.LayerControl().add_to(m)

        elif 'Кластеризация' in AddOptions:
            # FastMarkerCluster(data=list(zip(df1['lat'].values, df1['lon'].values))).add_to(m)
            # folium.LayerControl().add_to(m)
            # MarkerCluster(locations=Locations)
            mc = MarkerCluster()
            for index, row in df1.iterrows():
               mc.add_child(folium.Marker(location=[str(row['lat']),str(row['lon'])],
               popup=row['Наименование ПС'],
               # tooltip="Put anything you want here!",
               icon=folium.Icon(color=dic[row['Класс напряжения, кВ']], icon="info-sign", clustered_marker=True)))

               # folium.Circle(location=[str(row['Lat']), str(row['Long_'])],   color='Red', fill=True, fill_color='Red',
               #            radius=int(row['Confirmed']) * 30).add_to(m)
            m.add_child(mc)


        # folium.Marker(
        #     location=[55.70875238638579, 37.72187262683586],
        #     popup="АО Профотек",
        #     icon=folium.Icon(color="red", icon="info-sign"),
        # ).add_to(m)

        folium_static(m)


    df1 = pd.read_excel('./data/Реестор ЦПС.xlsx',
                    sheet_name='Sheet1',
                    header=0, parse_dates=['Дата ввода в эксплуатацию']
                    )

    df1["Дата ввода в эксплуатацию"] = df1["Дата ввода в эксплуатацию"].dt.date
    df1.sort_values('Дата ввода в эксплуатацию', inplace=True)

    df1 = df1[df1['Локация (Широта, Долгота)'].notna()]
    df1.drop_duplicates(inplace=True)
    df1.reset_index(drop=True,inplace=True)
    # df1
    lst = df1['Локация (Широта, Долгота)'].tolist()
    lon = []
    lat = []
    for i in lst:
        lat.append(i.split(',')[0])
        lon.append(i.split(',')[1][1:])
    df1['lon'] = lon
    df1['lat'] = lat


    VoltClassCouter = Counter(df1['Класс напряжения, кВ'])
    ArchetchCouter = Counter(df1['Архитектура построения ПС'])
    StageCouter = Counter(df1['Стадия реализации'])
    DZOCouter = Counter(df1['ДЗО ПАО Россети'])

    # st.write(DZOCouter)

    a = StageCouter['Введена в работу']
    b = StageCouter['СМР']
    c = StageCouter['ПИР']
    d = StageCouter['ОПЭ']

    # st.write(f'Всего в России {df1.shape[0]} ЦПС из которых {a} введены в работу, {b} находяться на стадии СМР, {c} на стадии ПИР и оставщиеся {d} на ОПЭ.')


    st.metric(label="Всего ЦПС:", value=df1.shape[0])

    st.write('Кол-во ЦПС по классу напряжения:')
    col1, col2, col3, col4, col5, col6, = st.columns(6)
    with col1:
        st.metric(label="ПС 10 кВ:", value=VoltClassCouter[10])
    with col2:
        st.metric(label="ПС 35 кВ:", value=VoltClassCouter[35])
    with col3:
        st.metric(label="ПС 110 кВ:", value=VoltClassCouter[110])
    with col4:
        st.metric(label="ПС 220 кВ:", value=VoltClassCouter[220])
    with col5:
        st.metric(label="ПС 330 кВ:", value=VoltClassCouter[330])
    with col6:
        st.metric(label="ПС 500 кВ:", value=VoltClassCouter[500])


    gd = GridOptionsBuilder.from_dataframe(df1[['Дата ввода в эксплуатацию','ДЗО ПАО Россети','Наименование ПС','Архитектура построения ПС','Стадия реализации','Производитель РЗА']])
    gd.configure_pagination(enabled=True)
    gd.configure_default_column(editable=False, groupable=True)
    # gd.configure_selection(selection_mode='multiple', use_checkbox=True)
    gridoptions = gd.build()

    with st.expander('Посмотреть таблицу'):
        # st.dataframe(df1)
        Grid_table = AgGrid(df1[['Дата ввода в эксплуатацию','ДЗО ПАО Россети','Наименование ПС','Архитектура построения ПС','Стадия реализации','Производитель РЗА']], gridOptions=gridoptions, update_mode=GridUpdateMode.SELECTION_CHANGED, height=400, theme = 'streamlit')
        # sel_rows = Grid_table['selected_rows']

    Selected_df = pd.DataFrame()

    st.markdown('---')

    AddOptions = []
    col1, col2 = st.columns(2)
    with col1:
        OptionList = st.selectbox('Осуществить поиск по категории: ', options = ['Все ЦПС','ДЗО ПАО Россети', 'По наименованию ПС', 'По классу напряжения', 'По архитектуре', 'Стадия реализации'])

    with col2:
        if OptionList == 'Все ЦПС':
            AddOptions = st.selectbox('Варианты отображения:', options = ['По классу напряжения','Тепловая карта','Кластеризация'])

        elif OptionList == 'ДЗО ПАО Россети':
            AddOptions = 'По классу напряжения'
            OptionName = 'ДЗО ПАО Россети'
            Option = st.multiselect('Список ДЗО ПАО Россети:', options = df1[OptionName].drop_duplicates())
            Selected_df = df1[df1[OptionName].isin(Option) ]
            # Selected_df.set_index('серийный номер', inplace=True)

        elif OptionList == 'По наименованию ПС':
            AddOptions = 'По классу напряжения'
            OptionName = 'Наименование ПС'
            Option = st.multiselect('Список ПС:', options = df1[OptionName].drop_duplicates() )
            # Selected_df = df1[df1[OptionName] == Option ]
            Selected_df = df1[df1[OptionName].isin(Option) ]
            # Selected_df.set_index('серийный номер', inplace=True)

        elif OptionList == 'По классу напряжения':
            AddOptions = 'По классу напряжения'
            OptionName = 'Класс напряжения, кВ'
            Option = st.multiselect('Список напряжений, кВ:', options = df1[OptionName].drop_duplicates().sort_values() )
            Selected_df = df1[df1[OptionName].isin(Option) ]
            # Selected_df.set_index('дата', inplace=True)

        elif OptionList == 'По архитектуре':
            AddOptions = 'По классу напряжения'
            OptionName = 'Архитектура построения ПС'
            Option = st.multiselect('Список проектов:', options =['II','III','IV'] )
            Selected_df = df1[df1[OptionName].isin(Option)]
            # Selected_df.set_index('серийный номер', inplace=True)

        elif OptionList == 'Стадия реализации':
            AddOptions = 'По классу напряжения'
            OptionName = 'Стадия реализации'
            Option = st.multiselect('Список проектов:', options = df1[OptionName].drop_duplicates())
            Selected_df = df1[df1[OptionName].isin(Option) ]
            # Selected_df.set_index('серийный номер', inplace=True)
        else:
            pass

    # YearRange = [i.year for i in df1.groupby(by=['Дата ввода в эксплуатацию'])['Наименование ПС'].count().index.tolist()]
    # SubStationAmountByYear_10 = df1[df1['Класс напряжения, кВ']==10].groupby(by=['Дата ввода в эксплуатацию'])['Наименование ПС'].count().tolist()
    # SubStationAmountByYear_35 = df1[df1['Класс напряжения, кВ']==35].groupby(by=['Дата ввода в эксплуатацию'])['Наименование ПС'].count().tolist()
    # SubStationAmountByYear_110 = df1[df1['Класс напряжения, кВ']==110].groupby(by=['Дата ввода в эксплуатацию'])['Наименование ПС'].count().tolist()
    # SubStationAmountByYear_220 = df1[df1['Класс напряжения, кВ']==220].groupby(by=['Дата ввода в эксплуатацию'])['Наименование ПС'].count().tolist()
    # SubStationAmountByYear_330 = df1[df1['Класс напряжения, кВ']==330].groupby(by=['Дата ввода в эксплуатацию'])['Наименование ПС'].count().tolist()
    # SubStationAmountByYear_500 = df1[df1['Класс напряжения, кВ']==330].groupby(by=['Дата ввода в эксплуатацию'])['Наименование ПС'].count().tolist()

    dic = {
    '2010':[0,0,0,0,0,0],
    '2011':[0,0,0,0,0,0],
    '2012':[0,0,0,0,0,0],
    '2013':[0,0,0,0,0,0],
    '2014':[0,0,0,0,0,0],
    '2015':[0,0,0,0,0,0],
    '2016':[0,0,0,0,0,0],
    '2017':[0,0,0,0,0,0],
    '2018':[0,0,0,0,0,0],
    '2019':[0,0,0,0,0,0],
    '2020':[0,0,0,0,0,0],
    '2021':[0,0,0,0,0,0],
    '2022':[0,0,0,0,0,0],
    }


    df2 = df1[['Дата ввода в эксплуатацию','Класс напряжения, кВ']]
    df3 = df2[df2['Дата ввода в эксплуатацию']<=date(2022,12,31)].value_counts().reset_index().sort_values('Дата ввода в эксплуатацию').reset_index(drop=True)
    # df3['Дата ввода в эксплуатацию'] = df3['Дата ввода в эксплуатацию'].dt.year

    for _, row in df3.iterrows():
        if row["Класс напряжения, кВ"] == 10:
            dic[str(row["Дата ввода в эксплуатацию"].year)][0] += row[0]
        elif row["Класс напряжения, кВ"] == 35:
            dic[str(row["Дата ввода в эксплуатацию"].year)][1] += row[0]
        elif row["Класс напряжения, кВ"] == 110:
            dic[str(row["Дата ввода в эксплуатацию"].year)][2] += row[0]
        elif row["Класс напряжения, кВ"] == 220:
            dic[str(row["Дата ввода в эксплуатацию"].year)][3] += row[0]
        elif row["Класс напряжения, кВ"] == 330:
            dic[str(row["Дата ввода в эксплуатацию"].year)][4] += row[0]
        elif row["Класс напряжения, кВ"] == 500:
            dic[str(row["Дата ввода в эксплуатацию"].year)][5] += row[0]

    df4 = pd.DataFrame(dic)

    option_5 = {
    'title': {
        'top': '0%',
        'left': 'center',
        'text': 'Кол-во ЦПС по классу напряжения'
      },
      'tooltip': {
        'trigger': 'axis',
        'axisPointer': {
          'type': 'shadow'
        }
      },
      'legend': {
        'top': '90%',
        'data': ['10 кВ', '35 кВ', '110 кВ', '220 кВ', '330 кВ','500 кВ']
      },
      'toolbox': {
        'show': 'true',
        'orient': 'vertical',
        'left': 'right',
        'top': 'center',
        'feature': {
          'mark': { 'show': 'true' },
          'dataView': { 'show': 'true', 'readOnly': 'false' },
          'magicType': { 'show': 'true', 'type': ['line', 'bar', 'stack'] },
          'restore': { 'show': 'true' },
          'saveAsImage': { 'show': 'true' }
        }
      },
      'xAxis': [
        {
          'type': 'category',
          'axisTick': { 'show': 'false' },
          'data': df4.columns.tolist()
        }
      ],
      'yAxis': [
        {
          'type': 'value'
        }
      ],
      'series': [
        {
          'name': '10 кВ',
          'type': 'bar',
          'barGap': 0,
          'label': 'labelOption',
          'emphasis': {
            'focus': 'series'
          },
          'data': df4.iloc[0].tolist()
        },
        {
          'name': '35 кВ',
          'type': 'bar',
          'label': 'labelOption',
          'emphasis': {
            'focus': 'series'
          },
          'data': df4.iloc[1].tolist()
        },
        {
          'name': '110 кВ',
          'type': 'bar',
          'label': 'labelOption',
          'emphasis': {
            'focus': 'series'
          },
          'data': df4.iloc[2].tolist()
        },
        {
          'name': '220 кВ',
          'type': 'bar',
          'label': 'labelOption',
          'emphasis': {
            'focus': 'series'
          },
          'data': df4.iloc[3].tolist()
        },
        {
          'name': '330 кВ',
          'type': 'bar',
          'label': 'labelOption',
          'emphasis': {
            'focus': 'series'
          },
          'data': df4.iloc[4].tolist()
        },
        {
          'name': '500 кВ',
          'type': 'bar',
          'label': 'labelOption',
          'emphasis': {
            'focus': 'series'
          },
          'data': df4.iloc[5].tolist()
        }
      ]
    };

    if not Selected_df.empty:
        YearRange = [i.year for i in Selected_df.groupby(by=['Дата ввода в эксплуатацию'])['Наименование ПС'].count().index.tolist()]
        SubStationAmountByYear = Selected_df.groupby(by=['Дата ввода в эксплуатацию'])['Наименование ПС'].count().tolist()

        option_6 = {
        'title': {
            'top': '0%',
            'left': 'center',
            'text': 'Кол-во ЦПС по годам'
              },
          'xAxis': {
            'type': 'category',
            'data': YearRange
          },
          'yAxis': {
            'type': 'value'
          },
          'series': [
            {
              'data': SubStationAmountByYear,
              'type': 'bar'
            }
          ]
        };

    col0, col1 = st.columns(2)
    
    with col0:
        if not Selected_df.empty:
            # with st.expander("See explanation", expanded=True):
            with st.expander("Карта:",expanded=True):
                FoliumMap(Selected_df, Option,OptionName, AddOptions)
        else:
            # with st.expander("See explanation", expanded=True):
            with st.expander("Карта:",expanded=True):
                FoliumMap(df1, Option=[],OptionName=[], AddOptions=AddOptions)
            # col1.subheader('Кол-во ЦПС по годам:')
    
    
    with col1:
        if not Selected_df.empty:
            with st.expander("Посмотреть таблицу"):
                Grid_table = AgGrid(Selected_df[['Дата ввода в эксплуатацию','ДЗО ПАО Россети','Наименование ПС','Архитектура построения ПС','Стадия реализации','Производитель РЗА']], key='1' ,gridOptions=gridoptions, update_mode=GridUpdateMode.SELECTION_CHANGED, height=250, theme = 'streamlit')
                # st.write(Selected_df[['Дата ввода в эксплуатацию','ДЗО ПАО Россети','Наименование ПС','Архитектура построения ПС','Стадия реализации','Производитель РЗА']])
            st_echarts(options=option_6)
        else:
            with st.expander("Аналитика:",expanded=True):
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                st_echarts(options=option_5)
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                st.write('')

    option_1 = {
      'tooltip': {
        'trigger': 'item'
      },
      'legend': {
        'top': '85%',
        'left': 'center'
      },
      'series': [
        {
          'name': 'Access From',
          'type': 'pie',
          'radius': ['40%', '70%'],
          'avoidLabelOverlap': 'false',
          'itemStyle': {
            'borderRadius': 10,
            'borderColor': '#fff',
            'borderWidth': 2
          },
          'label': {
            'show': 'false',
            'position': 'center'
          },
          'emphasis': {
            'label': {
              'show': 'true',
              'fontSize': '40',
              'fontWeight': 'bold'
            }
          },
          'labelLine': {
            'show': 'false'
          },
          'data': [
            { 'value': VoltClassCouter[10], 'name': 'ПС 10 кВ' },
            { 'value': VoltClassCouter[35], 'name': 'ПС 35 кВ' },
            { 'value': VoltClassCouter[110], 'name': 'ПС 110 кВ' },
            { 'value': VoltClassCouter[220], 'name': 'ПС 220 кВ' },
            { 'value': VoltClassCouter[330], 'name': 'ПС 330 кВ' },
            { 'value': VoltClassCouter[500], 'name': 'ПС 500 кВ' },
          ]
        }
      ]
    }

    # st_echarts(options=option_1)

    # st.write('Общее кол-во ЦПС по типу архитектуры:')
    # col0, col1, col2 = st.columns(3)
    # with col0:
    #     st.metric(label="II-я архитектура", value=ArchetchCouter['II'])
    # with col1:
    #     st.metric(label="III-я архитектура",value=ArchetchCouter['III'])
    # with col2:
    #     st.metric(label="IV-я архитектура", value=ArchetchCouter['IV'])

    option_2 = {
      'tooltip': {
        'trigger': 'item'
      },
      'legend': {
        'top': '85%',
        'left': 'center'
      },
      'series': [
        {
          'name': 'Access From',
          'type': 'pie',
          'radius': ['40%', '70%'],
          'avoidLabelOverlap': 'false',
          'itemStyle': {
            'borderRadius': 10,
            'borderColor': '#fff',
            'borderWidth': 2
          },
          'label': {
            'show': 'false',
            'position': 'center'
          },
          'emphasis': {
            'label': {
              'show': 'true',
              'fontSize': '40',
              'fontWeight': 'bold'
            }
          },
          'labelLine': {
            'show': 'false'
          },
          'data': [
            { 'value': ArchetchCouter['II'], 'name': 'II' },
            { 'value': ArchetchCouter['III'], 'name': 'III' },
            { 'value': ArchetchCouter['IV'], 'name': 'IV' },
          ]
        }
      ]
    }
    # st_echarts(options=option_2)

    option_3 = {
      'tooltip': {
        'trigger': 'item'
      },
      'legend': {
        'top': '85%',
        'left': 'center'
      },
      'series': [
        {
          'name': 'Access From',
          'type': 'pie',
          'radius': ['40%', '70%'],
          'avoidLabelOverlap': 'false',
          'itemStyle': {
            'borderRadius': 10,
            'borderColor': '#fff',
            'borderWidth': 2
          },
          'label': {
            'show': 'false',
            'position': 'center'
          },
          'emphasis': {
            'label': {
              'show': 'true',
              'fontSize': '40',
              'fontWeight': 'bold'
            }
          },
          'labelLine': {
            'show': 'false'
          },
          'data': [
            { 'value': StageCouter['Введена в работу'], 'name': 'Введена в работу' },
            { 'value': StageCouter['СМР'], 'name': 'СМР' },
            { 'value': StageCouter['ПИР'], 'name': 'ПИР' },
            { 'value': StageCouter['ОПЭ'], 'name': 'ОПЭ' },
          ]
        }
      ]
    }
    # st_echarts(options=option_3)

    option_4 = {
      'tooltip': {
        'trigger': 'item'
      },
      'legend': {
        'top': '85%',
        'left': 'center'
      },
      'series': [
        {
          'name': 'Access From',
          'type': 'pie',
          'radius': ['40%', '70%'],
          'avoidLabelOverlap': 'false',
          'itemStyle': {
            'borderRadius': 10,
            'borderColor': '#fff',
            'borderWidth': 2
          },
          'label': {
            'show': 'false',
            'position': 'center'
          },
          'emphasis': {
            'label': {
              'show': 'true',
              'fontSize': '40',
              'fontWeight': 'bold'
            }
          },
          'labelLine': {
            'show': 'false'
          },
          'data': [
            { 'value': DZOCouter['Россети "Сибирь"'], 'name': 'Сибирь' },
            { 'value': DZOCouter['Россети "Московский регион"'], 'name': 'Московский регион' },
            { 'value': DZOCouter['Россети "Центр и Приволжье"'], 'name': 'Центр и Приволжье' },
            { 'value': DZOCouter['Россети "Центр"'], 'name': 'Центр' },
            { 'value': DZOCouter['Россети "Урал"'], 'name': 'Урал' },
            { 'value': DZOCouter['Россети "Волга"'], 'name': 'Волга' },
            { 'value': DZOCouter['Россети "Томск"'], 'name': 'Томск' },
            { 'value': DZOCouter['Россети "Юг"'], 'name': 'Юг' },
            { 'value': DZOCouter['Россети "Тюмень"'], 'name': 'Тюмень' },
          ]
        }
      ]
    }
    # st_echarts(options=option_3)


    st.write('')
    st.markdown('---')
    st.subheader('Кол-во ЦПС по категориям')
    col0, col1 = st.columns(2)
    with col0:
        with st.expander("По классу напряжения:"):
            # st.write('По классу напряжения:')
            st_echarts(options=option_1)
        with st.expander("По типу архитектуры:"):
        # st.write('По типу архитектуры:')
            st_echarts(options=option_2)
    with col1:
        with st.expander('По ДЗО ПАО "Россети":'):
        # st.write('По ДЗО ПАО "Россети:')
            st_echarts(options=option_4)

        with st.expander("По стадии реализации:"):
        # st.write('По стадии реализации:')
            st_echarts(options=option_3)

    st.markdown('---')

    m = leafmap.Map(center=(62,66), zoom=3.25)
    col1, col2 = st.columns(2)
    with col1:
        m.to_streamlit(height=0)
