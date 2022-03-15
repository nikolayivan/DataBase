import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import datetime
import sqlite3

def app():
    st.title('Система сбора данных и мониторинг 📈')
    st.sidebar.write('')
#     st.sidebar.info('About: \n This is a demo version of web application designed to recode and analyse parameters from EPU. All rights belongs to JSC Profotech.')
    FolderPath = r'./data'
    FileName = '/DataBase.db'
#     FileName = '/Таблица Данных.xlsx'

    @st.experimental_memo
    def LoadDataBase(FolderPath,FileName, ColOption):
        con = sqlite3.connect(FolderPath + FileName)
        cur = con.cursor()
        DataBase_df = pd.read_sql('SELECT * FROM data', con, parse_dates=['Дата'])
        con.close()
        # DataBase_df = pd.read_excel(FolderPath + FileName, header = 0, parse_dates=['Дата'])
        DataBase_df["Дата"] = DataBase_df["Дата"].dt.date

        if ColOption:
            nan_value = float("NaN")
            DataBase_df.replace("", nan_value, inplace=True)
            DataBase_df.dropna(how='all', axis=1, inplace=True)
        return DataBase_df

    
    ColOption = False
    DataBase_df = LoadDataBase(FolderPath,FileName,ColOption)
    DataBase_df.sort_values('Дата', inplace=True, ignore_index=True)
    
    con1 = st.container()
    
    st.write('Обновить таблицу:')
    if st.button('Обновить'):
        LoadDataBase.clear()
        DataBase_df = LoadDataBase(FolderPath,FileName,ColOption)
        DataBase_df.sort_values('Дата', inplace=True, ignore_index=True)
        # st.session_state['DataBase_df'] = DataBase_df.to_dict()

    con1.write(f"Общее кол-во записей: `{DataBase_df.shape[0]}`")
    con1.write(f"Общее кол-во столбцов: `{DataBase_df.shape[1]}`" )
    
    col1, _ = st.columns([1,5])
    Number = col1.number_input('Укажите кол-во строк для отображения:', value=5)
    # st.write('Показать последние 10 записей из общей таблицы:')
    with st.expander("Посмотреть таблицу"):
        st.dataframe(DataBase_df.tail(Number))

    Plot_df = DataBase_df
    Plot_df.sort_values('Дата', inplace=True)

    with st.form("form_2"):
        st.write('Анализ данных по выбранному параметру:')
        con_2 = st.container()

        if st.checkbox('Выбрать все ЭОБ'):
            RowOptionList = list(dict.fromkeys(Plot_df['Данные ЭОБ: Зав. номер трансформатора'].tolist()))
            RowOption = RowOptionList
        else:
            RowOption = list(dict.fromkeys(Plot_df['Данные ЭОБ: Зав. номер трансформатора'].tolist()))
            RowOptionList = None

        con_3 = st.container()

        if st.checkbox('Выбрать все параметры'):
            ColOptionList = Plot_df.columns.tolist()
            ColOptions = ColOptionList
        else:
            ColOptionList = None
            ColOptions = [
            'Диагностические параметры (EOM Фаза А): Контраст',
            'Диагностические параметры (EOM Фаза А): Umod',
            'Диагностические параметры (EOM Фаза А): Max. ADC',
            'Диагностические параметры (EOM Фаза В): Контраст',
            'Диагностические параметры (EOM Фаза В): Umod',
            'Диагностические параметры (EOM Фаза В): Max. ADC',
            'Диагностические параметры (EOM Фаза С): Контраст',
            'Диагностические параметры (EOM Фаза С): Umod',
            'Диагностические параметры (EOM Фаза С): Max. ADC',
            'Диагностические параметры (EOM Фаза С): Tin',
            'Лазерный излучатель: Ток Лазерного Излучателя']

        AddColOptions = [
        'Дата',
        'Данные ЭОБ: Зав. номер трансформатора',
        'Оптические параметры (EOM Фаза А): Част. модуляции',
        'Оптические параметры (EOM Фаза В): Част. модуляции',
        'Оптические параметры (EOM Фаза С): Част. модуляции',
        ]

        con_4 = st.container()
        Rows = con_2.multiselect('Выбрать ЭОБ:', options=RowOption , default = RowOptionList)
        MainColumns = con_3.multiselect('Выбрать основные параметры:', options=ColOptions, default = ColOptionList)
        AddColumns = con_4.multiselect('Выбрать дополнительные параметры:', options=AddColOptions[3:])

        Columns = AddColOptions[:3] + MainColumns + AddColumns

        Columns = list(dict.fromkeys(Columns))

        submitted = st.form_submit_button("Ввод")

        if submitted:
            pass
        else:
            st.stop()

    if not Rows:
        st.warning('Выберите ряды!')
        st.stop()

    if not Columns:
        st.warning('Выберите столбцы!')
        st.stop()

    selected_df = Plot_df.loc[Plot_df['Данные ЭОБ: Зав. номер трансформатора'].isin(Rows), Columns]
    # st.write(selected_df.info())
    # selected_df = Plot_df

    datetimes = pd.to_datetime(selected_df["Дата"])
    selected_df["Дата"] = datetimes
    selected_df.sort_values('Дата', inplace=True)

    # st.line_chart(selected_df)
    st.header('')
    st.header('Результаты в абсолютных единицах')

    with st.expander("Посмотреть результирующую таблицу"):
        st.dataframe(selected_df)

    if Rows:
        if MainColumns:
            df = selected_df[selected_df['Оптические параметры (EOM Фаза А): Част. модуляции'] < 50 ]
            if not df.empty:
                # st.write(df)
                st.subheader('Частота модуляции 38 кГц')
                fig = px.scatter(df, x="Дата", y=MainColumns, hover_name=df['Данные ЭОБ: Зав. номер трансформатора'])
                fig.update_layout(legend_title_text=None,
                        legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1))
                st.plotly_chart(fig, use_container_width=True)

            df = selected_df[selected_df['Оптические параметры (EOM Фаза А): Част. модуляции'] > 50 ]
            if not df.empty:
                # st.write(df)
                st.subheader('Частота модуляции 66 кГц')
                fig = px.scatter(df, x="Дата", y=MainColumns, hover_name=df['Данные ЭОБ: Зав. номер трансформатора'])
                fig.update_layout(legend_title_text=None,
                        legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1))
                st.plotly_chart(fig, use_container_width=True)

            # marginal_y="violin",
            # marginal_x="box", trendline="ols", template="simple_white")

            # fig3 = px.scatter(grouped_df, y=Columns[1:])
            # st.plotly_chart(fig3, use_container_width=True)
            #
            # fig3 = px.box(grouped_df, y=Columns[1:], points="all")
            # st.plotly_chart(fig3, use_container_width=True)
            #
            # fig3 = px.violin(grouped_df, y=Columns[1:], points="all")
            # st.plotly_chart(fig3, use_container_width=True)


    # st.stop()

    selected_df_5 = pd.DataFrame()

    for Row in Rows:
        selected_df_6  = pd.DataFrame()
        selected_df_2  = selected_df.loc[selected_df['Данные ЭОБ: Зав. номер трансформатора'].isin([Row]), :]

        # st.write(selected_df_2)
        # st.stop()

        if type(selected_df_2) == type(pd.Series([1,2,3])):
            selected_df_2  = selected_df_2.to_frame().T
            # selected_df_2 = selected_df_2.reset_index()
            # selected_df_2.rename(columns={'index': 'Данные ЭОБ: Зав. номер трансформатора'}, inplace=True)
        # else:
        #     selected_df_2 = selected_df_2.reset_index()

        for Ind, Col in enumerate(selected_df_2.columns):
            Series = selected_df_2.loc[:,Col]
            if Col == 'Дата':
                selected_df_6[Col] = Series.sub(Series.iloc[0]).dt.days
                # selected_df_6[Col] = selected_df_2.loc[:,Col].sub(selected_df_2.iloc[0,Ind]).dt.days
            elif Col == 'Данные ЭОБ: Зав. номер трансформатора':
                selected_df_6[Col] = selected_df_2[Col]
            elif Col == 'Оптические параметры (EOM Фаза А): Част. модуляции':
                selected_df_6[Col] = selected_df_2[Col]
            else:
                for i in Series.index:
                    if not pd.isna(Series[i]):
                        selected_df_6[Col] = Series.div(Series[i])
                        break
                    # else:
                    #     selected_df_6[Col] = Series.div(Series.iloc[1])

        selected_df_5 = pd.concat([selected_df_5, selected_df_6], ignore_index=False)
        # selected_df_5.sort_values('Дата', inplace=True)


    st.header('')
    st.header('Результаты в относительных единицах')

    with st.expander('Показать таблицу'):
        st.write(selected_df_5)

    df = selected_df_5[selected_df_5['Оптические параметры (EOM Фаза А): Част. модуляции'] < 50]
    if not df.empty:
        st.subheader('Частота модуляции 38 кГц')
        fig2 = px.scatter(df, x='Дата', y=MainColumns, hover_name='Данные ЭОБ: Зав. номер трансформатора')
        fig2.update_layout(legend_title_text=None,
                legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1))
        st.plotly_chart(fig2, use_container_width=True)

    df = selected_df_5[selected_df_5['Оптические параметры (EOM Фаза А): Част. модуляции'] > 50]
    if not df.empty:
        st.subheader('Частота модуляции 66 кГц')
        fig2 = px.scatter(df, x='Дата', y=MainColumns, hover_name='Данные ЭОБ: Зав. номер трансформатора')
        fig2.update_layout(legend_title_text=None,
                legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1))
        st.plotly_chart(fig2, use_container_width=True)
