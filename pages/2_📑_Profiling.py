import pandas as pd
import os
from pandas_profiling import ProfileReport
import streamlit.components.v1 as components
import streamlit as st

#Data:
#https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
#http://dados.recife.pe.gov.br/dataset/acidentes-de-transito-com-e-sem-vitimas
#

def df_names():
    result = []
    dir_iter = os.scandir('data')
    for f in dir_iter:
        if f.name.endswith('.csv'):
            result.append(f.name[0:-4])
    return sorted(result)

def read_df(file):
    try:
        df = pd.read_csv(file, sep=',')
    except:
        df = pd.read_csv(file, sep=';')
    return df

def profile():
    df_name = st.session_state.dataset
    if df_name in st.session_state:
        return 
    df = read_df(f'data/{df_name}.csv')
    profile = ProfileReport(df, title=f"{df_name} Dataset")
    profile.to_file(f"reports/{df_name}.html")
    st.session_state[df_name] = df

def print_report():
    df_name = st.session_state.dataset
    if df_name not in st.session_state:
        return
    st.write(f'Dataset: <i>{df_name}</i>', unsafe_allow_html=True)
    report_file = open(f'reports/{df_name}.html', 'r', encoding='utf-8')
    source_code = report_file.read() 
    components.html(source_code, height=400, scrolling=True)

def print_controls():
    col1, col2 = st.columns([.3,.7])
    col1.selectbox('Selecione o Dataset', df_names(), label_visibility='collapsed', key='dataset')
    button_placeholder = col2.empty()
    if button_placeholder.button('Analisar'):
        #O container 'col2.empty()' é utilizado para que se substitua o seu conteúdo.
        #Se usar o container diretamente, os conteúdos são adicionados ao invés de serem substituídos.
        button_placeholder.button('Analisando...', disabled=True)
        profile()
        st.experimental_rerun()

print_controls()
print_report()