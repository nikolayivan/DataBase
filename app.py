import streamlit as st
# import streamlit_authenticator as stauth
from PIL import Image

from multiapp import MultiApp
from apps import DataBase, WriteIn, Mapping

st.set_page_config(page_title = 'Калибровка', layout = 'wide', page_icon = '🔰')

app = MultiApp()

with st.sidebar:
    image = Image.open(r'./img/logo-en.png')
    st.image(image)

    # st.info('About: This web application ...')
    # names = ['123']
    # usernames = ['123']
    # passwords = ['123']
    # hashed_passwords = stauth.hasher(passwords).generate()
    # authenticator = stauth.authenticate(names,usernames,hashed_passwords,
    #     'some_cookie_name','some_signature_key',cookie_expiry_days=30)
    #
    # name, authentication_status = authenticator.login('Login','main')
    #
    # if authentication_status:
    #     st.write('Вы вошли как: *%s*' % (name))
    #     # st.sidebar.title('Some content')
    # elif authentication_status == False:
    #     st.error('Логин/пароль неверны')
    #     st.stop()
    # elif authentication_status == None:
    #     st.warning('Укажите свой логин и пароль')
    #     st.stop()

# Add all your application here
app.add_app("Запись новых данных", WriteIn.app)
app.add_app("Анализ данных", DataBase.app)

# The main app
app.run()
