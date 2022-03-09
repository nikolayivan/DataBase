import streamlit as st
import numpy as np
import pandas as pd

import openpyxl
from datetime import datetime

def app():
    st.title('Система сбора данных и мониториг 📈')
    st.sidebar.write('')
    st.sidebar.info('About: This is a demo version of web application designed to recode and analyse parameters from EPU. All rights belongs to JSC Profotech.')
    FolderPath = r'./data'
    FileName = '/Таблица Данных.xlsx'

    @st.experimental_memo
    def LoadDataBase(FolderPath,FileName):
        DataBase_df = pd.read_excel(FolderPath + FileName, header = 0, parse_dates=['Дата'])
        return DataBase_df


    DataBase_df = LoadDataBase(FolderPath,FileName)
    DataBase_df.sort_values('Дата', inplace=True, ignore_index=True)

    st.write('Обновить таблицу:')
    if st.button('Обновить'):
        LoadDataBase.clear()
        DataBase_df = LoadDataBase(FolderPath,FileName)
        DataBase_df.sort_values('Дата', inplace=True, ignore_index=True)
        st.session_state['DataBase_df'] = DataBase_df.to_dict()

    st.write("Общее кол-во записей: ", DataBase_df.shape[0] )
    st.write("Общее кол-во столбцов: ", DataBase_df.shape[1] )

    st.write('Показать последние 10 записей из общей таблицы:')
    with st.expander("Посмотреть таблицу"):
        st.dataframe(DataBase_df.tail(10))

    con_1 = st.container()
    # ColOption = st.checkbox('Исключить пустые столбцы')

    uploaded_file = st.file_uploader("Загрузить файл")

    if uploaded_file is not None:

        with st.form("form_1"):

            Date = st.date_input("Укажите дату",datetime.today())
            if not Date:
                st.warning('Укажите Дату!')

            df0 = pd.read_csv(uploaded_file, sep=';', header=None, delimiter = None, names=['Параметр','Значение'], encoding="cp1251", na_filter=True, skip_blank_lines=True)
            df0.drop_duplicates(subset = ['Параметр'], keep = 'last', inplace=True)

            row_1_s = pd.Series({'Параметр': 'Дата', 'Значение': str(Date)})
            row_1_df = pd.DataFrame([row_1_s])
            df0 = pd.concat([row_1_df, df0], ignore_index=True)

            Columns = df0.iloc[:,0].dropna().tolist()
            Columns = [Col for Col in Columns if Col[0] != "=" if Col[0] != ":" if Col[0] != ""]
            df1 = df0.loc[df0.iloc[:,0].isin(Columns)].reset_index(drop=True)

            for row, name in enumerate(df1['Параметр']):
                df1.iloc[row,0] = ' '.join(name.split())

            SN = df1.loc[ df1['Параметр'] == 'Данные ЭОБ: Зав. номер трансформатора', 'Значение'].reset_index(drop=True)[0]

            # if len(SN) < 2:
            SN = st.text_input("Укажите серийный номер ЭОБ", value=SN)
            if len(SN) < 2:
            # if not SN:
                st.warning('Укажите серийный номер ЭОБ!')
            else:
                df1.loc[ df1['Параметр'] == 'Данные ЭОБ: Зав. номер трансформатора', 'Значение'] = SN

            with st.expander("Посмотреть загруженную таблицу"):
                st.write(df1)

            final_df = df1.T
            final_df.columns = final_df.iloc[0]
            final_df = final_df.drop('Параметр').reset_index(drop=True)

            result = pd.concat([DataBase_df, final_df], ignore_index=True, sort=False)

            df_to_save = result.iloc[-1]
            df_to_save = df_to_save.to_frame().T

                # with st.expander("Посмотреть таблицу для записи"):
                #     st.write(df_to_save)
            st.write('Записать в общую таблицу:')
            submitted_1 = st.form_submit_button('Записать')

            if submitted_1:
                if uploaded_file is not None:
#                     append_df_to_excel(FolderPath + FileName, df_to_save, sheet_name='Sheet1',header=0, index=False)
#                     st.success('Данные успешно записаны!')
                    with pd.ExcelWriter(FolderPath + FileName, mode="a", engine="openpyxl", if_sheet_exists="overlay",) as writer:
                        df_to_save.to_excel(writer, sheet_name="Sheet1", startrow=writer.sheets['Sheet1'].max_row, index = False,header= False)
                        st.success('Данные успешно записаны!')
                else:
                    st.warning("Файл не загружен! Загрузите файл...")

        # final_df.to_excel(FolderPath + '\output.xlsx', sheet_name='Sheet1', index = False)


        if st.checkbox('Поиск данных'):
            with st.form("form_2"):
                st.write('Выбрать определенный набор данных из загруженной таблицы для просмотра:')
                # if st.checkbox('Выбрать определенный набор данных из загруженной таблицы для просмотра:'):
                Rows = st.multiselect('Выбрать ряды', options=df1.iloc[:,0])

                submitted = st.form_submit_button("Вывести данные")

                if submitted:
                    selected_df = df1.loc[df1['Параметр'].isin(Rows)]

            if Rows:
                st.write(selected_df)
