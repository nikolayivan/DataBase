import streamlit as st

import numpy as np
import cvxpy as cp
import pandas as pd
import math, random
import scipy as sc
import time
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

def app():

    st.title('Расчёт и оптимизация суточного профиля потребления посёлка с электромобилями 🏘️🚗')

    with st.expander('Описание:'):
        st.markdown("""
        **Задача оптимизации** — рассчитать суточный график зарядки электромобилей, с учётом ограничения по установленной мощности подстанции, который:
        - максимизирует мощность зарядки каждого электромобиля,
        - стремится сгладить профиль зарядки каждого электромобиля,
        - учитывает требования владельцев электромобилей — время прибытия и отправления, желаемый конечный уровень заряда батареи.
        """)
        st.markdown("---")
        st.write("""
        Пользователь программы задаёт следующие параметры:
        ###### Обязательные параметры:
        - ограничение по установленной мощности подстанции, кВА;
        - максимальная мощность зарядки электромобиля, кВт — одно значение для всех электромобилей;
        - количество электромобилей;
        - количество домов/квартир/домохозяйств;

        ###### Опциональные параметры:
        - указатель со значением 0 или 1, который определяет выбор типа файла с данными по потреблению домов, который пользователь загрузит в программу.
           - 0 — файл с данными по суточному потреблению всех домов в посёлке;
           - 1 — файл с данными по суточному потреблению одного дома. На основе этих данных программа рассчитает примерный профиль потребления желаемого количества домов превышение предела установленной мощности, в %;
        - указание на то, учитывать ли три быстрые зарядки по 50 кВт:
           - зарядка 1 - с `12:00` до `14:00`; с `16:00` до `17:00`; с `19:00` до `21:00`
           - зарядка 2 - с `13:00` до `15:00`; с `15:30` до `17:00`; с `18:00` до `20:00`; с `10:00` до `11:00`
           - зарядка 3 - с `12:00` до `13:00`; с `13:00` до `15:00`; с `16:00` до `18:00`; с `21:00` до `23:00`; с `11:00` до `12:00`
              - 1 — учитывать быстрые зарядки;
              - 0 — не учитывать;

        По умолчанию в программе используются следующие значения параметров и файлы:
        - предел по установленной мощности подстанции — `570 кВА`;
        - максимальная мощность зарядки электромобиля — `3.6 кВт`;
        - количество электромобилей — `10`;
        - превышение предела по установленной мощности подстанции в задаче оптимизации — `0 %`;
        - не используются быстрые зарядки;
        - файл с данными по суточному потреблению `190` домов;
        - файл с данными о `171` электромобиле;

        Пользователь программы предоставляет:
        - файл с данными по электропотреблению домов в течение суток с 12:00 (включительно) по 12:00 (не включая) следующего дня;
        - файл с данными об электромобилях;

        """)
        st.markdown("---")
        st.write("""
        ###### Описание данных:
        - Данные по потреблению домов/квартир/домохозяйств:
           - Файл формата `*.csv`;
           - Размер файла — [количество домов x 48] (т.к. 48 периодов по 30 мин);
           - Файл должен быть без строки заголовка;
           - В строке - потребление дома (кВт*ч/2) в течение суток, которые начинаются в `12:00` и заканчиваются в `11:30` следующего дня;

        Пример файла доступен для скачивания ниже.

        - Данные об электромобилях:
        - Файл формата `*.csv`;
        - Размер файла — [количество электромобилей x 8];
        - Данные для каждого электромобиля:
           - порядковый номер электромобиля;
           - паспортная ёмкость батареи (кВт*ч);
           - фактическая ёмкость батареи (%);
           - остаток заряда (%);
           - время подключения (с точностью до часа — например, `17:00` или `22:00`. Считаем, что EV могут подключаться с `12:00` до `23:00` в первый день);
           - планируемое время выезда (с точностью до часа — например, 9 или 22. Считаем, что EV выезжают не позже `10:00` второго дня);
           - желаемый верхний порог зарядки (%, 80-100);
           - приемлемый нижний порог зарядки (пока не используется! Можно не заполнять — оставьте в Excel пустой столбец с заголовком);

        Пример файла доступен для скачивания ниже.
        """)
        st.markdown("---")
        st.write("""
        ##### Результаты:
        Результат работы программы — три графика:
        - общее потребление без учёта электромобилей (кВА) в течение суток;
        - общее потребление с учётом электромобилей (кВА) в течение суток без оптимизации;
        - общее потребление с учётом электромобилей (кВА) в течение суток с оптимизацией.


        Общее потребление с учётом электромобилей без оптимизации рассчитывается в предположении, что все электромобили начинают заряжаться на максимальной мощности сразу по прибытии и время их зарядки не ограничено (т. е. все электромобили могут заряжаться до 11:30 второго дня).

        Задача оптимизации будет иметь решение (в виде графиков зарядки электромобилей) не при всех значениях параметров. В частности, задача не решается, если максимальная мощность зарядки электромобиля и возможное время зарядки таковы, что все электромобили не успевают зарядиться до требуемого уровня за доступное время. Возможное решение — увеличить мощность зарядки электромобиля и доступное время зарядки, не разряжать батарею ниже определённого уровня.

        Также важно понимать, что при задании количества электромобилей они выбираются из всех имеющихся электромобилей случайным образом при каждом запуске программы. Каждый такой набор электромобилей будет характеризоваться своими запросами по зарядке и доступным временем для зарядки. Поэтому задача оптимизации может иметь решение для одного запуска программы — с одним набором электромобилей — и не иметь решения при следующем запуске программы — с другим набором электромобилей.
        """)
        st.markdown("---")
        col1,col2,col3 = st.columns([2,2,6])
        with open(r"C:\Users\testingcenter\Downloads\data_EV_example.csv", "rb") as file:
            col1.write("Скачать шаблон для электромобилей:")
            col2.download_button(label="⬇️ Скачать", key='1temp', data=file, file_name='data_EV_example.csv', mime="csv",)

        with open(r"C:\Users\testingcenter\Downloads\data_houses_example.csv", "rb") as file:
            col1.write("Скачать шаблон для домов:")
            col2.download_button(label="⬇️ Скачать", key='2temp',data=file, file_name='data_houses_example.csv', mime="csv",)

    # def main_EV_scheduling(P_max, S_upper_limit,  S_upper_upper_limit, use_lighting, use_quick_charges, house_koeff, EV_koeff, key, data_houses, data_EVs, n_EVs):
    def main_EV_scheduling(P_max, S_upper_limit, S_percent, use_quick_charges, key_houses, data_houses, data_1_house, data_EVs, n_EVs, n_houses):

        alpha = 1 # коэфф, требующий, чтобы заряжались на максимально возможной P
        beta = 0.5 # коэфф, требующий гладкий профиль зарядки

        # Подгружаем данные по домам
        def make_houses_dataset(house_koeff):
            if data_houses is not None:
                house_data = pd.read_csv(data_houses, encoding="windows-1251", header = None)
            rows = house_data.iloc[:, 0]
            house_data.index = rows
            house_data = house_data.T

            n_houses = house_data.shape[1]
            n_int = house_data.shape[0] # кол-во 30-минутных интервалов

            houses_E = house_koeff*house_data.sum(axis = 1) # общее потребление домов с учетом коэфф. одновременности

            return house_data, n_int, houses_E

        def make_1_house_dataset(house_koeff):
            house_df = pd.read_csv(data_1_house, header = None)
            house_data_arr = [round(num, 3) for num in list(house_df.iloc[0,:])]
            n_int_ = len(house_data_arr)
            #max_h = max(house_data_arr)

            house_list = [house_data_arr]
            for i in range(n_houses-1):
                arr = [0]*n_int_

                for j in range(n_int_):
                    arr[j] = house_data_arr[j] + np.random.normal(0, 0.15*house_data_arr[j], size = None)

                arr = [round(num, 3) for num in arr]
                house_list.append(arr)

            house_data = pd.DataFrame(house_list)
            house_data = house_data.transpose()

            houses_E = house_koeff*house_data.sum(axis = 1) # общее потребление домов с учетом коэфф. одновременности
            n_int = house_data.shape[0] # кол-во 30-минутных интервалов

            return house_data, n_int, houses_E

        def koeff_simultaneity(n_houses):
            keys = [1, 0.51, 0.38, 0.32, 0.29, 0.26, 0.24, 0.2, 0.18, 0.16, 0.14, 0.13]
            values = [list(range(1, 6)),
                      [6, 7, 8],
                      [9, 10, 11],
                      [12, 13, 14],
                      [15, 16, 17],
                      list(range(18, 24)),
                      list(range(24, 40)),
                      list(range(40, 60)),
                      list(range(60, 100)),
                      list(range(100, 200)),
                      list(range(200, 400)),
                      list(range(400, 600))
                     ]
            mydict = dict(zip(keys, values))

            if n_houses < 1:
                print('Ошибка: количество домов должно быть > 0!')
                return
            elif n_houses >= 600:
                return 0.11
            else:
                a = [k for k, v in mydict.items() if n_houses in v]
                return a[0]

        def n_EVs_(n_EVs):
            if data_EVs is not None:
                EVs_df = pd.read_csv(data_EVs, encoding="windows-1251")
            EVs_df = EVs_df.iloc[list(np.random.choice(range(EVs_df.shape[0]), n_EVs, replace=False)), :]
            return EVs_df

        # Подгружаем данные по EVs - время прибытия и отправления, определяем требуемое кол-во энергии
        def EVs_data(EVs_df):
            keys = list(range(12, 24)) + list(range(12))
            values = list(range(0, 48, 2))
            dictionary = dict(zip(keys, values))


            arrival_times = np.array([dictionary[key] for key in EVs_df['Время подключения, ч']])

            departure_times = [48 - dictionary[key] for key in EVs_df['Планируемое время выезда, ч']]

            required_energies = list(EVs_df['Паспортная емкость батареи, кВт*ч']* EVs_df['Фактическая емкость батареи, %']/100*\
            (EVs_df['Желаемый верхний порог зарядки, %'] - EVs_df['Остаток заряда, %'])/100)


            arrival_times, departure_times, required_energies  = \
            zip(*sorted(zip(arrival_times, departure_times, required_energies)))

            return arrival_times, departure_times, required_energies

        # Вычисляем профили зарядки EVs в самом простом случае - без оптимизации
        # EV приезжает - сразу начинает заряжаться на P_max
        def schedule_simple(n_int,n_EVs):
            A0 = np.zeros((n_int,n_EVs)) # n_int - кол-во 30-минутных интервалов с 12:00 до 11:30 след. дня

            for j in range(n_EVs):
                start = arrival_times[j]
                duration = math.ceil(required_energies[j]/P_max*2) # кол-во 30-минуток, чтобы зарядить EV на P_max

                A0[start:start + duration, j] = [1]*duration

            return A0

        # Оптимизация
        def optimization():
            S_limit = S_upper_limit * (1 + S_percent/100)
            diags = [[1]*n_int, [-1]*(n_int-1)]
            D = sc.sparse.diags(diags, [0, 1]).toarray()[:-1, :] # матрица для вычисления не-гладкости профиля зарядки

            X = cp.Variable((n_int, n_EVs)) # variable

            constr = [required_energies == cp.sum(X, axis=0) * P_max / 2,
              cp.sum(X, axis=1) * P_max / 0.95 <= list([S_limit]*n_int - houses_E*2/0.95 - np.array([x/0.95 for x in sum_quick])),
              X <= np.ones((n_int, n_EVs)),
              np.zeros((n_int, n_EVs)) <= X]

            for j in range(n_EVs):
                dep_time = departure_times[j]
                arr_time = arrival_times[j]
                if dep_time > 0:
                    constr += [X[-dep_time:, j] == 0]
                if arr_time > 0:
                    constr += [X[:arr_time, j] == 0]

            sum_1_norms = sum([cp.norm1(D @ X[:, j]) for j in range(n_EVs)])

            cost = 0

            for j in range(n_EVs):
                summ = 0
                for ll in range(n_int):
                    summ += (X[ll, j] - 1)**2
                cost += summ * alpha     # ограничение - чтобы заряжались на бОльших мощностях

            cost += beta * sum_1_norms   # ограничение на не-гладкость профиля

            prob = cp.Problem(cp.Minimize(cost),constr)

            prob.solve()

            if prob.status in ["infeasible", "unbounded"]:
                print('Задача оптимизации не имеет решения!!')
                return

            X_profile = X.value

            return X_profile


        def plot_houses_consumption(houses_E):
            st.write('##### Суммарное потребление э/э без учёта электромобилей')
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(n_int)), y=houses_E*2/0.95, name="Потребление без учёта электромобилей", line_shape='hv', fill='tozeroy',legendrank=2))
            fig.add_trace(go.Scatter(x=list(range(0, n_int)), y=[S_upper_limit]*n_int, name="Ограничение по установленной мощности",legendrank=1))
            fig.update_traces(hoverinfo='x+y', mode='lines')
            fig.update_layout(
                # title='Суммарное потребление э/э без учёта электромобилей',
                xaxis_title='Время, ч', yaxis_title='S, kVA',
                legend=dict(y=0.5, font_size=16),
                    xaxis = dict(
                    tickmode = 'array',
                    tickvals = np.arange(0, 49, step=2).tolist(),
                    ticktext = list(range(12, 24)) + list(range(0, 13))),
                    )
            st.plotly_chart(fig, use_container_width=True)


        def plot_schedule_simple(A0):
            st.write('##### Общее потребление э/э без оптимизации, kVA')

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(n_int)), y=houses_E*2/0.95, name="Потребление без учёта электромобилей", line_shape='hv', fill='tozeroy',legendrank=2))
            fig.add_trace(go.Scatter(x=list(range(0, n_int)), y=[S_upper_limit]*n_int, name="Ограничение по установленной мощности",legendrank=1))
            fig.add_trace(go.Scatter(x=list(range(0, n_int)), y=P_max/0.95*np.sum(A0, axis = 1) + houses_E*2/0.95 + np.array([x/0.95 for x in sum_quick]), name="Потребление с учётом электромобилей",line_shape='hv', fill='tozeroy',legendrank=3))
            fig.add_trace(go.Scatter(x=list(range(0, n_int)), y=P_max/0.95*np.sum(A0, axis = 1) + np.array([x/0.95 for x in sum_quick]), name="Потребление эл. энергии электромобилями",line_shape='hv', fill='tozeroy',legendrank=4))

            fig.update_traces(hoverinfo='x+y', mode='lines')
            fig.update_layout(
                # title='Общее потребление без оптимизации, kVA',
                xaxis_title='Время, ч', yaxis_title='S, kVA',
                legend=dict(y=0.5, font_size=16),
                    xaxis = dict(
                    tickmode = 'array',
                    tickvals = np.arange(0, 49, step=2).tolist(),
                    ticktext = list(range(12, 24)) + list(range(0, 13))),
                    )
            st.plotly_chart(fig, use_container_width=True)

        def plot_total_profile_optimized(X_profile):
            st.write('##### Общее потребление э/э с оптимизацией, kVA')
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(n_int)), y=houses_E*2/0.95, name="Потребление без учёта электромобилей", line_shape='hv', fill='tozeroy',legendrank=2))
            fig.add_trace(go.Scatter(x=list(range(0, n_int)), y=[S_upper_limit]*n_int, name="Ограничение по установленной мощности",legendrank=1))
            fig.add_trace(go.Scatter(x=list(range(0, n_int)), y=np.sum(X_profile, axis = 1)* P_max / 0.95 + houses_E*2/0.95 + np.array([x/0.95 for x in sum_quick]), name="Потребление с учётом электромобилей",line_shape='hv', fill='tozeroy',legendrank=3))
            fig.add_trace(go.Scatter(x=list(range(0, n_int)), y=np.sum(X_profile, axis = 1)* P_max / 0.95 + np.array([x/0.95 for x in sum_quick]), name="Потребление эл. энергии электромобилями",line_shape='hv', fill='tozeroy',legendrank=4))

            fig.update_traces(hoverinfo='x+y', mode='lines')
            fig.update_layout(
                # title='Общее потребление с оптимизацией, kVA',
                xaxis_title='Время, ч', yaxis_title='S, kVA',
                legend=dict(y=0.5, font_size=16),
                    xaxis = dict(
                    tickmode = 'array',
                    tickvals = np.arange(0, 49, step=2).tolist(),
                    ticktext = list(range(12, 24)) + list(range(0, 13))),
                    )
            st.plotly_chart(fig, use_container_width=True)

        # И наконец - запускаем всё!!
        house_koeff = koeff_simultaneity(n_houses)

        if key_houses == 0:
            house_data, n_int, houses_E = make_houses_dataset(house_koeff)
        elif key_houses == 1:
            house_data, n_int, houses_E = make_1_house_dataset(house_koeff)
        else:
            st.error('Ошибка: указатель выбора файла с данными принимает значения 0 или 1!')
            return


        EVs_df = n_EVs_(n_EVs)
        arrival_times, departure_times, required_energies = EVs_data(EVs_df)
        A0 = schedule_simple(n_int,n_EVs)

        st.write("")

        st.subheader('Результат')

        st.write('Пик потребления эл. энергии без электромобилей: ', round(max(houses_E*2/0.95),2), 'кВА')

        with st.expander("Посмотреть данные по домам:"):
            Names = [i for i in range(house_data.shape[1])]
            house_data.columns = Names
            st.write(house_data)
        with st.expander("Посмотреть данные по электромобилям:"):
            st.write(EVs_df.sort_index())

        if use_quick_charges == 1:
            quick_1 = [50]*4 + [0]*4 + [50]*2 + [0]*4 + [50]*4 + [0]*30
            quick_2 = [0]*2 + [50]*4 + [0]*1 + [50]*3 + [0]*2 + [50]*4 + [0]*28 + [50]*2 + [0]*2
            quick_3 = [50]*6 + [0]*2 + [50]*4 + [0]*6 + [50]*4 + [0]*24 + [50]*2
            sum_quick = [x+ y+ z for (x, y, z) in zip(quick_1, quick_2, quick_3)]
        elif use_quick_charges == 0:
            sum_quick = [0]*n_int
        else:
            st.error('Ошибка! use_quick_charges = 0 или 1!')
            return

        with st.spinner('Идёт рассчет...'):
            X_profile = optimization()

        plot_houses_consumption(houses_E)
        plot_schedule_simple(A0)
        plot_total_profile_optimized(X_profile)


    # plt.show

    st.subheader('Параметры:')
    with st.form("Form1"):

        st.write('##### Обязательные параметры:')
        col1,col2,col3,col4 = st.columns(4)

        with col1:
            S_upper_limit = st.number_input('Ограничение по установленной мощности ПС, кВА', min_value=0, max_value=None, value=570)
        with col2:
            P_max = st.number_input('Макс. мощность зарядки электромобиля, кВт', min_value=0.0, max_value=None, value=3.6) # kW - max мощность зарядки EV
        with col3:
            n_EVs = st.number_input('Кол-во электромобилей, шт', min_value=0, max_value=None, value=170)
        with col4:
            n_houses = st.number_input('Кол-во домов/квартир/домохозяйств, шт', min_value=0, max_value=None, value=190)

        st.markdown("---")

        st.write('##### Опциональные параметры:')
        col1,col2,col3,col4 = st.columns(4)
        with col1:
            S_percent = st.number_input('Превышение предела установленной мощности, в %',min_value=0.0, max_value=None, value=0.0, format='%f') # ограничение сверху на общее энергопотребление - EVs + дома
        with col3:
            key_houses = st.checkbox('Данные по одному дому')
        with col4:
            use_quick_charges = st.checkbox('Быстрые зарядки')

        for option in [use_quick_charges,key_houses]:
            if option:
                option = 1
            else:
                option = 0

        st.markdown("---")
        st.write('##### Файлы с исходными данными:')

        col1,col2 = st.columns(2)
        with col1:
            data_houses = st.file_uploader('Загрузите данные по потреблению домов за сутки:', type='csv', key='1', accept_multiple_files=False)
        with col2:
            data_EVs = st.file_uploader('Загрузите данные об электромобилях:', type='csv', key='2', accept_multiple_files=False)

        if st.checkbox('Использовать demo файлы'):
            data_houses = "./data/190 houses.csv"
            data_EVs = "./data/data EVs.csv"

        data_1_house = "./data/1 house 24 hours.csv"
        # st.info('EVs приезжают рандомно с 19:00 до 23:00, могут заряжаться до 07:00')
        st.write("")
        # st.markdown("---")

        submitted = st.form_submit_button("🚀Расчитать")

    if submitted:
        main_EV_scheduling(P_max, S_upper_limit, S_percent, use_quick_charges, key_houses, data_houses, data_1_house, data_EVs, n_EVs, n_houses)

