import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pages.util.plot_pages_util import read_titanic_df, build_nota_titanic, build_dataframe_section

def plot_idade(df:pd.DataFrame):
    st.markdown('<h2>Gráficos Relativos à Idade</h2>', unsafe_allow_html=True)
    plot_histograma_idade(df)
    plot_boxplot_idade(df)

def get_sequence(color_sequence):
    #código usado para alternar as cores, para aumentar a diferença da tonalidade
    return [x for idx, x in enumerate(color_sequence) if idx%2!=0]

def plot_histograma_idade(df:pd.DataFrame):
    st.markdown('<h3>Histograma</h3>', unsafe_allow_html=True)
    c1, c2 = st.columns([.3,.7])
    cols = ['sobreviveu', 'classe', 'sexo', 'embarque']
    series_col = c1.selectbox('Série*', options=cols, key='serie_1')
    stacked = c1.checkbox('Empilhado', value=True)
    color_sequences = {'Azul (reverso)': get_sequence(px.colors.sequential.Blues_r),
                       'Azul': get_sequence(px.colors.sequential.Blues),
                       'Plasma (reverso)': get_sequence(px.colors.sequential.Plasma_r),
                       'Plasma': get_sequence(px.colors.sequential.Plasma),
                       'Vermelho (reverso)': get_sequence(px.colors.sequential.Reds_r),
                       'Vermelho': get_sequence(px.colors.sequential.Reds),}
    color_sequence_key = c1.selectbox('Escala de Cor', options=color_sequences.keys())
    color_sequence = color_sequences[color_sequence_key]
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

def plot_boxplot_idade(df:pd.DataFrame):
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

def build_body():
    df = read_titanic_df()
    build_dataframe_section(df)
    plot_idade(df)

def build_header():
    text ='<h1>Visualização de Dados (Dataviz)</h1>'+\
    '<p>Esta página apresenta alguns gráficos a partir da base de dados do '+\
    'Titanic¹ (https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv).</p>'
    st.write(text, unsafe_allow_html=True)
    build_nota_titanic()

build_header()
build_body()