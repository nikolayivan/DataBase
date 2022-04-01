import streamlit as st
# import streamlit_authenticator as stauth
from PIL import Image

from multiapp import MultiApp
from apps import Mapping, Analysis


st.set_page_config(page_title = 'Рееcтор ЦПС', layout = 'wide', page_icon = '📈')
st.title('Реестор Цифровых ПС в ПАО "Россети"')
app = MultiApp()

with st.sidebar:
    ImgPath = r'./img'
    image1 = Image.open(ImgPath + '/' + 'rosseti_logo.png')
    image2 = Image.open(ImgPath + '/' + 'skoltech_logo.png')
    col1, col2 = st.columns(2)
    col1.image(image1)
    col2.image(image2)

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
    #     st.error('Username/password is incorrect')
    #     st.stop()
    # elif authentication_status == None:
    #     st.warning('Please enter your username and password')
    #     st.stop()

# Add all your application here
# app.add_app("Реестор ЦПС", Mapping.app)
app.add_app("Анализ ЦПС", Analysis.app)

# The main app
app.run()
