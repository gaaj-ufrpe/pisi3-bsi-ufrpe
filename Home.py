import streamlit as st

st.set_page_config(
    page_title = "Teste do Streamlit por Gabriel Alves",
    layout = "wide",
    menu_items = {
        'About': "TESTE DO ABOUT"
    }
)

st.markdown(f'''
    <h1>Testes com o Streamlit</h1>
    <br>
    Este branch do projeto utiliza páginas e a integração com o YData Profiling para analisar dados de um dataframe (página <i>Profiling</i>).
    <br>
    Além disso, alguns comandos de agrupamento de dataframe também são explorados (página <i>Grouping</i>).
''', unsafe_allow_html=True)
