import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils import read_df

def build_header():
    text ='<h1>Visualização de Dados (Dataviz)</h1>'+\
    '''<p>Esta página apresenta alguns gráficos a partir da base de dados do 
    Titanic¹ (https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv).</p>
    '''
    st.write(text, unsafe_allow_html=True)

    with st.expander('¹Notas sobre o dataset'):
        st.write(
        '''
        <table>
            <tr><th>COLUNA ORGIGINAL</th><th>COLUNA</th><th>DESCRIÇÃO</th></tr>
            <tr><td>PassengerId</td><td>id</td><td>O Id do passageiro.</td></tr>
            <tr><td>Name</td><td>nome</td><td>Nome do passageiro.</td></tr>
            <tr><td>Survived</td><td>sobreviveu</td><td>Sobreviveu: 0 = Não, 1 = Sim</td></tr>
            <tr><td>Pclass</td><td>classe</td><td>Classe do Passageiro: 1 = 1a. classe, 2 = 2a classe, 3 = 3a classe</td></tr>
            <tr><td>Sex</td><td>sexo</td><td>Sexo</td></tr>
            <tr><td>Age</td><td>idade</td><td>Idade em anos</td></tr>
            <tr><td>SibSp</td><td>irmaos</td><td># de irmãos/esposa no Titanic.</td></tr>
            <tr><td>Parch</td><td>pais</td><td># de pais/filhos no Titanic.</td></tr>
            <tr><td>Ticket</td><td>id_passagem</td><td>Número da passagem.</td></tr>
            <tr><td>Fare</td><td>tarifa</td><td>Tarifa da passagem.</td></tr>
            <tr><td>Cabin</td><td>cabine</td><td>Número da cabine</td></tr>
            <tr><td>Embarked</td><td>embarque</td><td>Local de embarque C = Cherbourg, Q = Queenstown, S = Southampton</td></tr>
        </table>
        <br>
        Notas:<br>
        Pclass: Classe do navio, relacionada à socio-econômica:<br>
        1a classe = Classe Alta<br>
        2a classe = Classe Média<br>
        3a classe = Classe Baixa<br>
        Age: Fracionada se for menor que 1. Se estimada, está no formato xx.5<br>
        Parch: Algumas crianças viajaram com babás, assim este atributo fica zerado para elas (Parch=0).
        ''', unsafe_allow_html=True)

def ingest_data() -> pd.DataFrame:
    df = read_df('titanic')
    df.rename(columns={
        'PassengerId':'id','Name':'nome',
        'Survived':'sobreviveu_val','Pclass':'classe_val',
        'Sex':'sexo','Age':'idade','SibSp':'irmaos',
        'Parch':'pais','Ticket':'id_passagem',
        'Fare':'tarifa','Cabin':'cabine','Embarked':'embarque'
    }, inplace=True)
    return df

def transform_data(df:pd.DataFrame) -> pd.DataFrame:
    df['sobreviveu'] = df['sobreviveu_val'].map({
        0: 'Não', 1: 'Sim',
    })
    df['classe'] = df['classe_val'].map({
        1: 'Primeira', 2: 'Segunda', 3: 'Terceira',
    })
    return df

def plot_df(df:pd.DataFrame):
    st.write('<h2>Dados do Titanic</h2>', unsafe_allow_html=True)
    st.dataframe(df)

def plot_idade(df:pd.DataFrame):
    st.markdown('<h2>Dados Relativos à idade</h2>', unsafe_allow_html=True)

def plot_histograma(df:pd.DataFrame, container):
    st.markdown('<h3>Histograma</h3>', unsafe_allow_html=True)
    c1, c2 = st.columns([.3,.7])
    cols = ['sobreviveu', 'classe', 'sexo', 'embarque']
    serie_col = c1.selectbox('Série', options=cols)
    stacked = c1.checkbox('Stacked', value=True)
    separar = False
    fig = None
    if stacked:
        separar = c1.selectbox('Separar', options=['Não Separar']+cols)
        facet_col = separar if separar != 'Não Separar' else None 
        fig = px.histogram(df, x='idade', color=serie_col, opacity=.75, facet_row=facet_col)
    else:
        fig = go.Figure()
        opacity = 1
        serie_vals = ordered_vals(df, serie_col)
        for val in serie_vals:
            df_aux = df.query(f'{serie_col}==@val').copy()
            hist = go.Histogram(name=str(val),x=df_aux['idade'], opacity=opacity)
            fig.add_trace(hist)
            opacity -= .1
        fig.update_layout(barmode='overlay', legend_title_text=serie_col)
    c2.plotly_chart(fig)

def ordered_vals(df:pd.DataFrame, col:str) -> list:
    result = df[['id',col]].groupby(by=col).count()
    result = result.sort_values(by='id', ascending=False).reset_index().copy()
    return result[col].to_list()

def plot_boxplot(df:pd.DataFrame):
    st.markdown('<h3>Diagrama de Caixa (<i>Boxplot</i>)</h3>', unsafe_allow_html=True)
    c1, c2 = st.columns([.3,.7])
    cols = ['sobreviveu', 'classe', 'sexo', 'embarque']
    serie_col = c1.selectbox('Série', options=cols)
    inverter = c1.checkbox('Inverter Eixos', True)
    cols = [serie_col, 'idade']
    if inverter:
        cols.reverse()
    df_plot = df[cols]
    fig = px.box(df_plot,x=cols[0],y=cols[1])
    c2.plotly_chart(fig)

def build_body():
    df = ingest_data()
    df = transform_data(df)
    plot_df(df)
    plot_idade(df)
    plot_boxplot(df)

build_header()
build_body()