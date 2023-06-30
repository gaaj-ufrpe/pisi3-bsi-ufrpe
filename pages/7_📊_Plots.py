import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pages.util.plot_pages_util import read_titanic_df, build_nota_titanic, build_dataframe_section, get_color_sequence_names, get_color_sequence

def build_page():
    build_header()
    build_body()

def build_header():
    text ='<h1>Plots com a base do Titanic</h1>'+\
    '<p>Esta página apresenta alguns gráficos a partir da base de dados do '+\
    'Titanic¹ (https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv).</p>'
    st.write(text, unsafe_allow_html=True)
    build_nota_titanic()

def build_body():
    df = read_titanic_df()
    build_dataframe_section(df)
    st.markdown('<h2>Gráficos Relativos à Idade</h2>', unsafe_allow_html=True)
    build_scatter_section(df)
    build_bubble_section(df)
    build_histograma_section(df)
    build_boxplot_section(df)

def build_scatter_section(df:pd.DataFrame):
    #see: https://plotly.com/python/marker-style/
    st.markdown('<h3>Scatter</h3>', unsafe_allow_html=True)
    #ordenamento para bater com a ordem das sequencias de simbolo e cor
    df.sort_values(by=['sexo','classe'], ascending=[False, True], inplace=True)
    fig = px.scatter(df, x='idade', y='tarifa', color='sexo', symbol='classe', 
                     opacity=.75, color_discrete_sequence=['#99ccff','#ffb3b3'], 
                     symbol_sequence=['circle','square','triangle-down'])
    fig.update_traces(marker_size=8, marker_line_width=1)
    st.plotly_chart(fig, use_container_width=True)

def faixa_etaria(idade:int) -> str:
    result = '60+'
    if idade is None or np.isnan(idade):
        result = None
    elif idade < 60:
        first = int(f'{int(idade/10)}0')
        second = first+9
        result = f'{first}-{second}'
    return result

def build_bubble_section(df:pd.DataFrame):
    st.markdown('<h3>Bubble</h3>', unsafe_allow_html=True)
    df_aux = df[['sexo','classe','idade','sobreviveu','id']].copy()
    df_aux['faixa_etaria'] = df_aux['idade'].apply(faixa_etaria)
    df_aux.drop('idade',axis=1,inplace=True)
    df_aux = df_aux.groupby(by=['sexo','classe','sobreviveu','faixa_etaria']).count().reset_index()
    df_aux['sobreviveu'] = df_aux['sobreviveu'].map({'Não':'mortos','Sim':'vivos'})
    df_aux.rename(columns={'id':'qtd'},inplace=True)
    df_aux = df_aux.pivot(index=['sexo','classe','faixa_etaria'], 
                          columns='sobreviveu', values='qtd').reset_index().\
                    sort_values(by=['sexo','classe'], ascending=[False, True])
    df_aux.fillna(0, inplace=True)
    df_aux['qtd'] = df_aux['vivos'] + df_aux['mortos']
    fig = px.scatter(df_aux, x='vivos', y='mortos', text='faixa_etaria',
                     color='sexo', symbol='classe', size='qtd', size_max=80, opacity=.75, 
                     color_discrete_sequence=['#99ccff','#ffb3b3'],
                     symbol_sequence=['circle','square','triangle-down'])
    st.plotly_chart(fig, use_container_width=True)
    
def build_histograma_section(df:pd.DataFrame):
    st.markdown('<h3>Histograma</h3>', unsafe_allow_html=True)
    c1, c2 = st.columns([.3,.7])
    cols = ['sobreviveu', 'classe', 'sexo', 'embarque']
    series_col = c1.selectbox('Série*', options=cols, key='serie_1')
    stacked = c1.checkbox('Empilhado', value=True)
    color_sequence_name = c1.selectbox('Escala de Cor', options=get_color_sequence_names())
    color_sequence = get_color_sequence(color_sequence_name)
    if stacked:
        fig = create_histograma_stacked(df, series_col, color_sequence)
    else:
        fig = create_histograma_unstacked(df, series_col, color_sequence)
    c2.plotly_chart(fig, use_container_width=True)
    fig.update_layout(title=f'Histograma de "{series_col}" por faixa etária.', 
                      legend_title_text=series_col, xaxis_title_text='Idade', yaxis_title_text='Quantidade')

def create_histograma_stacked(df:pd.DataFrame, series_col:str, color_sequence) -> go.Figure:
    df = df.query(f'{series_col}.notna()').copy()
    df.sort_values(by=series_col, inplace=True)
    return px.histogram(df, x='idade', nbins=20, color=series_col, 
                        opacity=.75, color_discrete_sequence=color_sequence)

def create_histograma_unstacked(df:pd.DataFrame, series_col:str, color_sequence) -> go.Figure:
    # em alguns casos, pode ser interessante ou mesmo necessário usar a api graph objects do plotly: https://plotly.com/python/graph-objects/
    # esta api se baseia na inclusão de 'traces' sobre uma figura. ademais, propriedades e eixos da figura e dos traces podem ser customizados
    # apesar de mais complexo, o uso destes elementos diretamente permite que cada elemetno do gráfico seja ajustado individualmente.
    series_vals = ordered_vals(df, series_col)
    fig = go.Figure()
    color_idx = 0
    for val in series_vals:
        query = f'{series_col}==@val'
        df_aux = df.query(query).copy()
        opacity = .5 * (1+color_idx/10)
        str_series = str(val)
        #showlegend e o legendgroup são usados para mostrar apenas a legenda da primeira linha
        #e permitir que a interação com a legenda altere o estado de todos os subplots
        color = f'{color_sequence[color_idx]}'
        hist = go.Histogram(name=str_series, x=df_aux['idade'], 
                            xbins=dict(start=0,end=80,size=5), legendgroup=val, showlegend=True,
                            marker={'color': color, 'opacity':opacity})
        fig.add_trace(hist)
        color_idx += 1
    fig.update_layout(barmode='overlay', legend_title_text=series_col)
    return fig

def ordered_vals(df:pd.DataFrame, col:str) -> list:
    result = df[['id',col]].groupby(by=col).count().\
        reset_index().copy()
    return result[col].to_list()

def build_boxplot_section(df:pd.DataFrame):
    st.markdown('<h3>Diagrama de Caixa (<i>Boxplot</i>)</h3>', unsafe_allow_html=True)
    c1, c2 = st.columns([.3,.7])
    cols = ['sobreviveu', 'classe', 'sexo', 'embarque']
    serie_col = c1.selectbox('Série*', options=cols, key='serie_2')
    inverter = c1.checkbox('Inverter Eixos', True)
    cols = [serie_col, 'idade']
    if inverter:
        cols.reverse()
    df_plot = df[cols]
    fig = px.box(df_plot,x=cols[0],y=cols[1])
    c2.plotly_chart(fig, use_container_width=True)
    st.text('''
    *Estes elementos de input tÊm os mesmos valores e mesmo nome. Por isso, é necessario informar 
    o atributo "key" destes elementos com valores diferentes. Caso contrário, o streamlit entende que 
    o mesmo componente está sendo inserido duas vezes na mesma página, dando um erro.
    ''')

build_page()