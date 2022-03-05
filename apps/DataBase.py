import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

def app():
    st.title('Система сбора данных и мониториг 📈')
    st.sidebar.write('')
    st.sidebar.info('About: \n This is a demo version of web application designed to recode and analyse parameters from EPU. All rights belongs to JSC Profotech.')
    FolderPath = r'./data'
    FileName = '/Таблица Данных.xlsx'

    if 'DataBase_df' not in st.session_state:
        st.session_state['DataBase_df'] = {}

    # @st.experimental_memo
    def LoadDataBase(FolderPath,FileName, ColOption):
        DataBase_df = pd.read_excel(FolderPath + FileName, header = 0, parse_dates=['Дата'])
        if ColOption:
            nan_value = float("NaN")
            DataBase_df.replace("", nan_value, inplace=True)
            DataBase_df.dropna(how='all', axis=1, inplace=True)
        return DataBase_df

    con_1 = st.container()
    ColOption = st.checkbox('Исключить пустые столбцы')

    if not st.session_state['DataBase_df']:
        DataBase_df = LoadDataBase(FolderPath,FileName, ColOption)
        DataBase_df.sort_values('Дата', inplace=True, ignore_index=True)
        st.session_state['DataBase_df'] = DataBase_df.to_dict()
    else:
        DataBase_df = pd.DataFrame(st.session_state['DataBase_df'])

    st.write("Общее кол-во записей: ", DataBase_df.shape[0] )
    st.write("Общее кол-во столбцов: ", DataBase_df.shape[1] )

    with st.expander("Посмотреть общую таблицу"):
        st.dataframe(DataBase_df)

    con_1.write('Обновить таблицу:')
    con_1.button('Обновить', on_click=LoadDataBase, args=(FolderPath,FileName,ColOption),)

    Plot_df = DataBase_df
    Plot_df.sort_values('Дата', inplace=True)

    with st.form("form_2"):
        st.write('Анализ данных по выбранному параметру.')

        con_2 = st.container()

        if st.checkbox('Выбрать все трансформаторы'):
            RowOptionList = list(dict.fromkeys(Plot_df['Данные ЭОБ: Зав. номер трансформатора'].tolist()))
            RowOption = RowOptionList
        else:
            RowOption = list(dict.fromkeys(Plot_df['Данные ЭОБ: Зав. номер трансформатора'].tolist()))
            RowOptionList = None

        con_3 = st.container()

        if st.checkbox('Выбрать все столбцы'):
            ColOptionList = Plot_df.columns.tolist()
            ColOptions = ColOptionList
        else:
            ColOptionList = None
            ColOptions = [
            'Дата',
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
        'Данные ЭОБ: Зав. номер трансформатора',
        'Оптические параметры (EOM Фаза А): Част. модуляции',
        'Оптические параметры (EOM Фаза В): Част. модуляции',
        'Оптические параметры (EOM Фаза С): Част. модуляции',
        ]

        con_4 = st.container()

        Rows = con_2.multiselect('Выбрать ряды:', options=RowOption , default = RowOptionList)
        MainColumns = con_3.multiselect('Выбрать основные столбцы:', options=ColOptions, default = ColOptionList)
        AddColumns = con_4.multiselect('Выбрать дополнительные столбцы:', options=AddColOptions, default = AddColOptions[:2])

        Columns = MainColumns + AddColumns

        submitted = st.form_submit_button("Ввести")

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

    st.write(selected_df)

    # # st.stop()
    #
    # # selected_df.to_excel(FolderPath + 'output.xlsx', sheet_name='Sheet1', index = False)
    #
    # # st.stop()
    #
    # # selected_df[Columns[1:]] = selected_df[Columns[1:]].astype(float)
    #
    # # selected_df.dropna( how = 'all', subset=selected_df.columns[1:-1], inplace=True )
    # # st.write(selected_df)
    # # if 'Диагностические параметры (EOM Фаза А): Umod' or 'Диагностические параметры (EOM Фаза В): Umod' or 'Диагностические параметры (EOM Фаза С): Umod' in ColOptions:
    #
    # selected_df = selected_df[['Данные ЭОБ: Зав. номер трансформатора','Диагностические параметры (EOM Фаза А): Umod','Диагностические параметры (EOM Фаза В): Umod','Диагностические параметры (EOM Фаза С): Umod']]
    # selected_df.set_index('Данные ЭОБ: Зав. номер трансформатора', inplace=True)
    #
    # st.write(selected_df)
    #
    # # st.stop()
    #
    # grouped_df = selected_df.groupby(level=0).mean()
    # grouped_df['Mean'] = grouped_df.sum(axis=1) / grouped_df.count(axis=1)
    #
    # def f(x):
    #     if x['Mean'] <= 7:
    #         return 66
    #     elif x['Mean'] > 7:
    #         return 38
    #
    # grouped_df['Freq'] = grouped_df.apply(f,axis=1)
    #
    # lst5 = []
    # for ind1, row1 in selected_df.iterrows():
    #     for ind2, row2 in grouped_df.iterrows():
    #         if ind1 == ind2:
    #             lst5.append(row2['Freq'])
    #
    # selected_df['Freq'] = lst5
    #
    # with st.expander('Показать таблицу'):
    #     st.table(selected_df)
    #
    # selected_df.to_excel(FolderPath + 'output_freq.xlsx', sheet_name='Sheet1', index = False)
    # st.write(selected_df)
    # st.stop()

    # st.line_chart(selected_df)
    if Rows:
        if 'Дата' in Columns:
            if len(Columns) >= 2:

                df = selected_df[selected_df['Оптические параметры (EOM Фаза А): Част. модуляции'] < 50 ]
                if not df.empty:
                    # st.write(df)
                    st.subheader('Частота модуляции 38 Гц')
                    fig = px.scatter(df, x="Дата", y=MainColumns, hover_name=df['Данные ЭОБ: Зав. номер трансформатора'])
                    st.plotly_chart(fig, use_container_width=True)

                df = selected_df[selected_df['Оптические параметры (EOM Фаза А): Част. модуляции'] > 50 ]
                if not df.empty:
                    # st.write(df)
                    st.subheader('Частота модуляции 66 Гц')
                    fig = px.scatter(df, x="Дата", y=MainColumns, hover_name=df['Данные ЭОБ: Зав. номер трансформатора'])
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
        selected_df_6 = pd.DataFrame()
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
            if Col == 'Дата':
                Series = selected_df_2.loc[:,Col]
                selected_df_6[Col] = Series.sub(Series.iloc[0]).dt.days
                # selected_df_6[Col] = selected_df_2.loc[:,Col].sub(selected_df_2.iloc[0,Ind]).dt.days
            elif Col == 'Данные ЭОБ: Зав. номер трансформатора':
                selected_df_6[Col] = selected_df_2[Col]
            elif Col == 'Оптические параметры (EOM Фаза А): Част. модуляции':
                selected_df_6[Col] = selected_df_2[Col]
            else:
                Series = selected_df_2.loc[:,Col]
                selected_df_6[Col] = Series.div(Series.iloc[0])

        selected_df_5 = pd.concat([selected_df_5, selected_df_6], ignore_index=True)
        selected_df_5.sort_values('Дата', inplace=True)

    with st.expander('Показать таблицу'):
        st.write(selected_df_5)

    df = selected_df_5[selected_df_5['Оптические параметры (EOM Фаза А): Част. модуляции'] < 50]
    if not df.empty:
        st.subheader('Частота модуляции 38 Гц')
        fig2 = px.scatter(df, x='Дата', y=MainColumns, hover_name='Данные ЭОБ: Зав. номер трансформатора')
        st.plotly_chart(fig2, use_container_width=True)

    df = selected_df_5[selected_df_5['Оптические параметры (EOM Фаза А): Част. модуляции'] > 50]
    if not df.empty:
        st.subheader('Частота модуляции 66 Гц')
        fig2 = px.scatter(df, x='Дата', y=MainColumns, hover_name='Данные ЭОБ: Зав. номер трансформатора')
        st.plotly_chart(fig2, use_container_width=True)
