import streamlit as st

st.set_page_config(
    page_title = "Titulo da pagina",
    layout = "wide",
    menu_items = {
        'About': "TESTE DO ABOUT"
    }
)

nome = st.text_input('Nome')
if nome != '':
    st.markdown(f'Oi, {nome}. O STREAMLIT RODOU!')
else:
    st.markdown('O STREAMLIT RODOU!')
