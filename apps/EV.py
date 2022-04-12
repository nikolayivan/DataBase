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

    st.title('Расчёт и оптимизация суточного профиля потребления посёлка с электромобилями')

    with st.expander('Описание:'):
        st.write("""
        Задача оптимизации — рассчитать суточный график зарядки электромобилей (EV), с учётом ограничения по установленной мощности подстанции (S_upper_limit, кВА), который:
        максимизирует мощность зарядки каждого EV,
        стремится сгладить профиль зарядки каждого EV,
        учитывает требования владельцев EV — время прибытия и отправления, желаемый конечный уровень заряда батареи EV.
        """)


    def main_EV_scheduling(P_max, S_upper_limit,  S_upper_upper_limit, use_lighting, use_quick_charges, house_koeff, EV_koeff, key, data_houses, data_EVs, n_EVs):

        alpha = 1 # коэфф, требующий, чтобы заряжались на максимально возможной P
        beta = 0.5 # коэфф, требующий гладкий профиль зарядки

        # Подгружаем данные по домам
        def make_houses_dataset():

            if data_houses is not None:
                house_data = pd.read_csv(data_houses, encoding="windows-1251", header = None)
            # house_data = pd.read_csv(data_houses, encoding="windows-1251", header = None)
            rows = house_data.iloc[:, 0]
            house_data = house_data.iloc[:, 1:]
            house_data.index = rows
            house_data = house_data.T

            n_houses = house_data.shape[1]
            n_int = house_data.shape[0] # кол-во 30-минутных интервалов с 12:00 до 12:00 след. дня

            if use_lighting == 1:
                house_data['lighting'] = [0]*10 + [3.6]*32 + [0]*6
            elif  use_lighting != 0:
                st.error('Ошибка! use_lighting = 0 или 1!')
                return

            houses_E = house_koeff*house_data.sum(axis = 1) # общее потребление домов

            return house_data, n_int, houses_E

        def n_EVs_(n_EVs):
            if data_EVs is not None:
                EVs_df = pd.read_csv(data_EVs, encoding="windows-1251")
            # EVs_df = pd.read_csv(data_EVs, encoding="windows-1251")
            EVs_df = EVs_df.iloc[:, [0, 2, 4, 5, 6, 7, 8, 9]]

            EVs_df = EVs_df.iloc[list(np.random.choice(range(EVs_df.shape[0]), n_EVs, replace=False)), :]
            #print(EVs_df.shape)

            return EVs_df

        # Подгружаем данные по EVs - время прибытия и отправления, определяем требуемое кол-во энергии
        def EVs_data(EVs_df):

            keys = list(range(12, 24)) + list(range(12))
            values = list(range(0, 48, 2))
            dictionary = dict(zip(keys, values))


            arrival_times = np.array([dictionary[key] for key in EVs_df['Время подключения']])

            departure_times = [48 - dictionary[key] for key in EVs_df['Планируемое время выезда']]

            required_energies = list(EV_koeff*EVs_df['Паспортная емкость батареи']* EVs_df['Фактическая емкость батареи, %']/100*\
            (EVs_df['Желаемый верхний порог зарядки'] - EVs_df['Остаток заряда'])/100)


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
        def optimization(key):

            if key == 0:
                S_limit = S_upper_limit
            elif key == 1:
                st.warning('Оптимизируем с выходом за пределы установленной мощности. Новый предел - ', np.round(S_upper_upper_limit, 3), 'кВА')
                S_limit = S_upper_upper_limit
            else:
                st.error('Ошибка! key = 0 или 1!')
                return

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
                st.error('Optimization is infeasible!!')
                return

            X_profile = X.value

            return X_profile


        def plot_houses_consumption(houses_E):
            # plt.figure(figsize=(15, 6.5))
            # fig = plt.figure()
            # plt.step(list(range(n_int)), houses_E*2/0.95, linewidth=2.5)
            # plt.plot(list(range(0, n_int)), [S_upper_limit]*n_int, color = 'darkred', label = 'Ограничение по установленной мощности', linewidth=2.5)
            # #plt.legend(loc='center left', bbox_to_anchor=(1.2, 0.9))
            # plt.legend(fontsize=15, loc=0, labelspacing = 0.06)
            # plt.title("Суммарное потребление домов", fontsize = 17)
            # plt.ylabel("S, kVA", fontsize=17)
            # plt.xlabel("время, ч", fontsize=17)
            # plt.grid(True)
            # plt.xticks(np.arange(0, 49, step=2),  list(range(12, 24)) + list(range(0, 13)), fontsize=17)
            # _ = plt.yticks(fontsize=17)

            # st.pyplot(fig)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(n_int)), y=houses_E*2/0.95, name="Дома", line_shape='hv'))
            fig.add_trace(go.Scatter(x=list(range(0, n_int)), y=[S_upper_limit]*n_int, name="Ограничение по установленной мощности"))
            fig.update_traces(hoverinfo='x+y', mode='lines')
            fig.update_layout(
                title='Суммарное потребление домов',
                xaxis_title='Время, ч', yaxis_title='S, kVA',
                legend=dict(y=0.5, traceorder='reversed', font_size=16),
                    xaxis = dict(
                    tickmode = 'array',
                    tickvals = np.arange(0, 49, step=2).tolist(),
                    ticktext = list(range(12, 24)) + list(range(0, 13))),
                    )
            st.plotly_chart(fig, use_container_width=True)


        def plot_schedule_simple(A0):

            # plt.figure(figsize=(15, 6.5))
            # fig = plt.figure()
            # plt.step(list(range(n_int)), houses_E*2/0.95, label="Дома", linewidth=2.5)
            # plt.plot(list(range(0, n_int)), [S_upper_limit]*n_int, color = 'darkred', label = 'Ограничение по установленной мощности', linewidth=2.5)
            # plt.step(list(range(0, n_int)), P_max/0.95*np.sum(A0, axis = 1) + houses_E*2/0.95 + np.array([x/0.95 for x in sum_quick]), label="Дома + электромобили", linewidth=2.5, color = 'orange')
            # plt.step(list(range(0, n_int)), P_max/0.95*np.sum(A0, axis = 1) + np.array([x/0.95 for x in sum_quick]), label="Электромобили", linewidth=2.5, color = 'magenta')
            #
            # plt.legend(loc='center left', bbox_to_anchor=(1, 0.7))
            # plt.title("Общее потребление без оптимизации, kVA", fontsize = 17)
            # plt.ylabel("S, kVA", fontsize=17)
            # plt.xlabel("время, ч", fontsize=17)
            # plt.grid(True)
            # plt.xticks(np.arange(0, 49, step=2),  list(range(12, 24)) + list(range(0, 13)), fontsize=17)
            # _ = plt.yticks(fontsize=17)

            # st.pyplot(fig)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(n_int)), y=houses_E*2/0.95, name="Дома", line_shape='hv'))
            fig.add_trace(go.Scatter(x=list(range(0, n_int)), y=[S_upper_limit]*n_int, name="Ограничение по установленной мощности"))
            fig.add_trace(go.Scatter(x=list(range(0, n_int)), y=P_max/0.95*np.sum(A0, axis = 1) + houses_E*2/0.95 + np.array([x/0.95 for x in sum_quick]), name="Дома + электромобили",line_shape='hv'))
            fig.add_trace(go.Scatter(x=list(range(0, n_int)), y=P_max/0.95*np.sum(A0, axis = 1) + np.array([x/0.95 for x in sum_quick]), name="Электромобили",line_shape='hv'))

            fig.update_traces(hoverinfo='x+y', mode='lines')
            fig.update_layout(
                title='Общее потребление без оптимизации, kVA',
                xaxis_title='Время, ч', yaxis_title='S, kVA',
                legend=dict(y=0.5, font_size=16),
                    xaxis = dict(
                    tickmode = 'array',
                    tickvals = np.arange(0, 49, step=2).tolist(),
                    ticktext = list(range(12, 24)) + list(range(0, 13))),
                    )
            st.plotly_chart(fig, use_container_width=True)

        def plot_total_profile_optimized(X_profile):
            # plt.figure(figsize=(15, 6.5))
            # fig = plt.figure()
            # plt.step(list(range(0, n_int)), houses_E*2/0.95, label="Дома", linewidth=2.5)
            # plt.step(list(range(0, n_int)), np.sum(X_profile, axis = 1)* P_max / 0.95 + houses_E*2/0.95 + np.array([x/0.95 for x in sum_quick]), label="Дома + электромобили", linewidth=2.5, color = 'orange')
            # plt.step(list(range(0, n_int)), np.sum(X_profile, axis = 1)* P_max / 0.95 + np.array([x/0.95 for x in sum_quick]), label="Электромобили", linewidth=2.5, color = 'magenta')
            #
            # plt.plot(list(range(0, n_int)), [S_upper_limit]*n_int, color = 'darkred', label = 'Ограничение по установленной мощности', linewidth=2.5)
            # #plt.legend(fontsize=15, loc=0, labelspacing = 0.06)
            # plt.legend(loc='center left', bbox_to_anchor=(1, 0.7))
            # plt.title("Общее потребление с оптимизацией, kVA", fontsize = 17)
            # plt.ylabel("S, kVA", fontsize=17)
            # plt.xlabel("время, ч", fontsize=17)
            # plt.grid(True)
            # plt.xticks(np.arange(0, 49, step=2),  list(range(12, 24)) + list(range(0, 13)), fontsize=17)
            # _ = plt.yticks(fontsize=17)
            # _ = plt.yticks(fontsize=17)

            # st.pyplot(fig)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(n_int)), y=houses_E*2/0.95, name="Дома", line_shape='hv'))
            fig.add_trace(go.Scatter(x=list(range(0, n_int)), y=[S_upper_limit]*n_int, name="Ограничение по установленной мощности"))
            fig.add_trace(go.Scatter(x=list(range(0, n_int)), y=np.sum(X_profile, axis = 1)* P_max / 0.95 + houses_E*2/0.95 + np.array([x/0.95 for x in sum_quick]), name="Дома + электромобили",line_shape='hv'))
            fig.add_trace(go.Scatter(x=list(range(0, n_int)), y=np.sum(X_profile, axis = 1)* P_max / 0.95 + np.array([x/0.95 for x in sum_quick]), name="Электромобили",line_shape='hv'))

            fig.update_traces(hoverinfo='x+y', mode='lines')
            fig.update_layout(
                title='Общее потребление с оптимизацией, kVA',
                xaxis_title='Время, ч', yaxis_title='S, kVA',
                legend=dict(y=0.5, font_size=16),
                    xaxis = dict(
                    tickmode = 'array',
                    tickvals = np.arange(0, 49, step=2).tolist(),
                    ticktext = list(range(12, 24)) + list(range(0, 13))),
                    )
            st.plotly_chart(fig, use_container_width=True)

        # И наконец - запускаем всё!!
        house_data, n_int, houses_E = make_houses_dataset()
        #print('house_data.shape, n_int, houses_E', house_data.shape, n_int, houses_E)
        EVs_df = n_EVs_(n_EVs)
        #print('n_EVs, n_int, EVs_df', n_EVs, n_int, EVs_df.shape)
        arrival_times, departure_times, required_energies = EVs_data(EVs_df)
        #print('arrival_times, departure_times, required_energies ', arrival_times, departure_times, required_energies)

        A0 = schedule_simple(n_int,n_EVs)
        st.subheader('Результаты')

        # with st.expander('Посмотреть исходные таблицы:'):
        #     st.write(house_data)
        #     st.write(EVs_df)

        st.write('Пик потребления домов: ', round(max(houses_E*2/0.95),2), 'кВА')

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

        X_profile = optimization(key)

        # %matplotlib inline
        plot_houses_consumption(houses_E)
        plot_schedule_simple(A0)
        plot_total_profile_optimized(X_profile)


    # plt.show

    st.subheader('Параметры:')
    with st.form("Form1"):

        st.write('Обязательные параметры:')
        col1,col2,col3 = st.columns(3)

        with col1:
            S_upper_limit = st.number_input('Ограничение по установленной мощности ПС', min_value=None, max_value=None, value=570)
            S_upper_upper_limit = 1.035*S_upper_limit
        with col2:
            n_EVs = st.number_input('Кол-во EVs', min_value=None, max_value=None, value=10)
        with col3:
            P_max = st.number_input('Максимальная мощность зарядки EV, кВт', help='Одно значение для всех EV', min_value=None, max_value=None, value=3.6) # kW - max мощность зарядки EV

        col1,col2 = st.columns(2)
        with col1:
            data_houses = st.file_uploader('Загрузите данные по потреблению домов за сутки:', type='csv', key='1', accept_multiple_files=False)
        with col2:
            data_EVs = st.file_uploader('Загрузите данные об электромобилях:', type='csv', key='2', accept_multiple_files=False)

        if st.checkbox('Использовать demo файлы'):
            data_houses = ".data/data houses 22 23 February 95 houses.csv"
            data_EVs = ".data/data EVs.csv"

        st.markdown("---")

        st.write('Опциональные параметры:')
        col1,col2,col3,col4,col5 = st.columns([0.25,0.25,0.133,0.133,0.133])

        with col1:
            house_koeff = st.number_input('Ограничение на общее энергопотребление', help='Коэфф. для пропорционального увеличения/уменьшения общего потребления домов',min_value=None, max_value=None, value=1) # ограничение сверху на общее энергопотребление - EVs + дома
        with col2:
            EV_koeff = st.number_input('Сколько заряда есть', help='Коэфф. для пропорционального увеличения/уменьшения общего требуемого потребления EV', min_value=None, max_value=None, value=1) # сколько заряда есть
        with col3:
            use_lighting = st.checkbox('Затраты э/э на освещение') #   сколько заряда требуется
        with col4:
            key = st.checkbox('Увеличенный предел')  # (Выбор тарифа. 0 - трёхставочный, 1 - почасовой)
        with col5:
            use_quick_charges = st.checkbox('Быстрые зарядки')


        for option in [use_lighting,key,use_quick_charges]:
            if option:
                option = 1
            else:
                option = 0

        # st.info('EVs приезжают рандомно с 19:00 до 23:00, могут заряжаться до 07:00')
        st.write("")
        # st.markdown("---")

        submitted = st.form_submit_button("🚀Расчитать")

    if submitted:
        main_EV_scheduling(P_max, S_upper_limit, S_upper_upper_limit, use_lighting, use_quick_charges, house_koeff, EV_koeff, key, data_houses, data_EVs, n_EVs)
