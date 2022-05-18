import streamlit as st

st.set_page_config(
    page_title = "Titulo da pagina",
    layout = "wide",
    menu_items = {
        'About': "TESTE DO ABOUT"
    }
)

nome = st.text_input('Nome')
if nome == '':
    st.markdown('O STREAMLIT RODOU!')
else:
    st.markdown(f'''
        Oi, <b style="color:#0033FF; text-transform:uppercase">{nome}</b>.<br>
        O STREAMLIT RODOU!<br>
        E ele aceita <code>HTML</code> e <code>CSS</code>!
    ''', unsafe_allow_html=True)
