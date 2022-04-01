import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
import folium

from collections import Counter
from streamlit_folium import folium_static
from streamlit_echarts import st_echarts
from PIL import Image
from datetime import datetime
from st_aggrid import AgGrid, GridUpdateMode
from st_aggrid.grid_options_builder import GridOptionsBuilder

def app():
    # st.sidebar.subheader('Описание:')
    with st.sidebar.expander("Описание"):
        st.info('Приложение предназначено для анализа развития Цифровых ПС в компании [ПАО «Россети»](https://rossetimr.ru/#/). Приложение выполнено в рамках образовательной программы «Лидеры энергетики» совместно с [Skoltech](https://www.skoltech.ru/?lang=ru).')

    st.write('Функционал находиться в стадии разработки...')
