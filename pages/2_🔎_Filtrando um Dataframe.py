import streamlit as st
import numpy as np
from utils import df_names, read_df

def initial_query(df):
    col_name = df.columns[0]
    value = df[col_name][0]
    if not isinstance(value,np.number):
        value = f'"{value}"'
    return f'{col_name} == {value}'

def print_df(container, df, query_str, sort_values):
    filtered_df = df
    if len(query_str) > 0:
        filtered_df = df.query(query_str).copy()
    if len(sort_values) > 0:
        filtered_df.sort_values(by=sort_values, inplace=True)
    container.dataframe(filtered_df, use_container_width=True)

def build_header():
    text ='<h1>Filtrando Dados de um Dataframe</h1>'+\
    '''<p>O pandas disponibiliza diversas formas de filtragem de dados. A utilização do método <code>query</code> é o que mais se 
    aproxima de uma consulta SQL.</p>
    <p>
    Este método permite que se passe uma string que represente a condição equivalente à cláusula <code>WHERE</code> de um comando SQL.
    Neste texto, é possível usar a notação <code>df.query('col == @v')</code> para se filtrar as linhas do dataframe para as quais
    a coluna <code>col</code> possui o valor atribuído à variável <code>v</code>, encontrada no escopo da query. Também é possível 
    passar direto o valor sem fazer uso da notação <code>@variavel</code>.
    </p>
    <p>Para "like" em colunas do tipo string, utilizar a notação <code>col.str.contains('parte do conteúdo')</code>. Para um "like" case-insensitive, 
    usar a notação <code>col.str.lower().str.contains("parte do conteúdo")</code>.
    </p>'''
    st.write(text, unsafe_allow_html=True)

def build_body():
    col1, col2 = st.columns([.3,.7])
    df_name = col1.selectbox('Dataset', df_names())
    df = read_df(df_name)
    sort_values = col2.multiselect('Ordenar', options=df.columns, default=df.columns[0])
    tooltip ='Insira neste campo a condição de filtragem.'
    query_str = st.text_area('Query', height=5, value=initial_query(df), help=tooltip)
    print_df(st, df, query_str, sort_values)

build_header()
build_body()
