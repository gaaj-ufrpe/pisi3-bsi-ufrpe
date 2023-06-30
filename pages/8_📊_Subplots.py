import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pages.util.plot_pages_util import read_titanic_df, build_nota_titanic, build_dataframe_section

def build_page():
    build_header()
    build_body()

def build_header():
    st.write('<h1>Criação de Subplots</h1>', unsafe_allow_html=True)

def build_body():
    df = read_titanic_df()
    c1, _, c2 = st.columns([.25,.05,.7])
    min_idade = int(np.min(df.idade))
    max_idade = int(np.max(df.idade))
    idade_range, linha, coluna, disposicao = build_controls(c1, [min_idade,max_idade])
    df_aux = query(df, linha, coluna, idade_range)
    plot_histogramas_px(c2, linha, coluna, disposicao, df_aux)
    plot_histogramas_go(c2, linha, coluna, disposicao, df_aux)

def build_controls(c,idade_range):
    i0,i1 = c.slider('Idade', value=(idade_range[0],idade_range[1]),
                      min_value=idade_range[0], max_value=idade_range[1])
    cols = ['classe', 'sexo', 'embarque']
    linha = c.selectbox('Linhas',cols)
    coluna = c.selectbox('Colunas',[x for x in cols if x != linha])
    disposicao = c.selectbox('Barras',['stack','overlay','group'])
    return [i0,i1], linha, coluna, disposicao

def query(df, col_linhas, col_colunas, idade_range):
    i0 = idade_range[0]
    i1 = idade_range[1]
    df_aux = df[['id','idade','sobreviveu',col_linhas,col_colunas]].query(f'idade>=@i0 and idade<=@i1').copy()
    df_aux.dropna(inplace=True)
    df_aux.sort_values(by=['sobreviveu'], inplace=True)
    return df_aux

def plot_histogramas_px(container, col_linhas, col_colunas, disposicao, df_aux):
    fig = px.histogram(df_aux, x='idade', nbins=20, opacity=.75,
        color='sobreviveu', color_discrete_sequence=['#ff6666','#66ff66'],
        facet_row=col_linhas, facet_row_spacing=.15,
        facet_col=col_colunas, facet_col_spacing=.15, )
    fig.update_layout(barmode=disposicao, title=f'Subplots de mortes ({col_linhas} X {col_colunas}) com "Plotly Express"', legend_title_text='Sobreviveu')
    container.plotly_chart(fig, use_container_width=True)
    
def plot_histogramas_go(container, col_linhas, col_colunas, disposicao, df_aux):
    col_linhas_vals = df_aux[col_linhas].unique() if col_linhas is not None else ['']
    col_colunas_vals = df_aux[col_colunas].unique() if col_colunas is not None else ['']
    fig = make_subplots(rows=len(col_linhas_vals), cols=len(col_colunas_vals), vertical_spacing=.3,)
    for row_idx, linha_val in enumerate(col_linhas_vals):
        for col_idx, col_val in enumerate(col_colunas_vals):
            ridx = row_idx+1            
            cidx = col_idx+1
            showlegend = (ridx==1 and cidx==1)
            add_subplot_histograma(col_linhas, col_colunas, linha_val, col_val, df_aux, fig, ridx, cidx, showlegend)
            fig.add_annotation(xref='x domain',yref='y domain', x=0, y=1.2, showarrow=False,
                               text=f'{col_linhas}: {linha_val} & {col_colunas}: {col_val}', row=ridx, col=cidx)
            fig.update_xaxes(matches='x', row=ridx, col=cidx)
            fig.update_yaxes(title='', matches='y', row=ridx, col=cidx)
            
    fig.update_layout(barmode=disposicao, title=f'Subplots de mortes ({col_linhas} X {col_colunas}) com "Graph Objects"', legend_title_text='Sobreviveu')
    # fig = px.histogram(df_aux, x='idade', nbins=20, opacity=.75,
    #     color='sobreviveu', color_discrete_sequence=['#ff6666','#66ff66'],
    #     facet_row=col_linhas, facet_row_spacing=.15,
    #     facet_col=col_colunas, facet_col_spacing=.15, )
    container.plotly_chart(fig, use_container_width=True)
    # fig.update_layout(title=title, legend_title_text=series_col)
    # fig.update_xaxes(title_text=xtitle)
    # fig.update_yaxes(title_text=ytitle)

def add_subplot_histograma(col_linhas, col_colunas, col_linhas_val, col_colunas_val, df, fig, row_idx, col_idx, showlegend):
    query = f'{col_linhas}=="{col_linhas_val}" and {col_colunas}=="{col_colunas_val}"'
    df_aux = df.query(f'{query} and sobreviveu=="Não"').copy()
    subfig = go.Histogram(name='Não',x=df_aux.idade, xbins=dict(start=0,end=80,size=5), 
                          marker={'color': '#ff6666', 'opacity':.75}, legendgroup='Não', showlegend=showlegend)
    fig.add_trace(subfig, row=row_idx, col=col_idx)
    df_aux = df.query(f'{query} and sobreviveu=="Sim"').copy()
    subfig = go.Histogram(name='Sim',x=df_aux.idade, xbins=dict(start=0,end=80,size=5), 
                          marker={'color': '#66ff66', 'opacity':.75}, legendgroup='Sim', showlegend=showlegend)
    fig.add_trace(subfig, row=row_idx, col=col_idx)


#                 hist = go.Histogram(name=str_series, x=df_aux['idade'], 
#                                     xbins=dict(start=0,end=80,size=5), legendgroup=val, showlegend=showlegend,
#                                     marker={'color': color, 'opacity':opacity})

build_page()



# def create_histograma_unstacked(df:pd.DataFrame, series_col:str, facet_col:str, color_sequence, title: str, xtitle: str, ytitle: str) -> go.Figure:
#     # em alguns casos, pode ser interessante ou mesmo necessário usar a api graph objects do plotly: https://plotly.com/python/graph-objects/
#     # esta api se baseia na inclusão de 'traces' sobre uma figura. ademais, propriedades e eixos da figura e dos traces podem ser customizados
#     # apesar de mais complexo, o uso destes elementos diretamente permite que cada elemetno do gráfico seja ajustado individualmente.
#     if facet_col is None:
#         facets = ['']
#         query_start = ''
#     else:
#         facets = ordered_vals(df,facet_col)
#         query_start = f'{facet_col}==@facet and'
#     facets_len = len(facets)
#     fig = make_subplots(facets_len, 1, vertical_spacing=.3, x_title=xtitle, y_title=ytitle)
#     series_vals = ordered_vals(df, series_col)
#     series_color = {}
#     row_idx = 0
#     for facet in facets:
#         row_idx += 1
#         color_idx = 0
#         for val in series_vals:
#             query = f'{query_start} {series_col}==@val'
#             df_aux = df.query(query).copy()
#             if len(df_aux)!=0:
#                 opacity = .5 * (1+color_idx/10)
#                 str_series = str(val)
#                 #showlegend e o legendgroup são usados para mostrar apenas a legenda da primeira linha
#                 #e permitir que a interação com a legenda altere o estado de todos os subplots
#                 if str_series in series_color.keys():
#                     color = series_color[str_series] 
#                     showlegend = False
#                 else:
#                     color = f'{color_sequence[color_idx]}'
#                     series_color[str_series] = color
#                     showlegend = True
#                 hist = go.Histogram(name=str_series, x=df_aux['idade'], 
#                                     xbins=dict(start=0,end=80,size=5), legendgroup=val, showlegend=showlegend,
#                                     marker={'color': color, 'opacity':opacity})
#                 fig.add_trace(hist, row=row_idx, col=1)
#                 fig.add_annotation(xref='x domain',yref='y domain',x=0, y=1.1, showarrow=False,
#                                 text=f'{facet_col}: {facet}', row=row_idx, col=1)
#                 fig.update_xaxes(matches='x', row=row_idx, col=1)
#                 fig.update_yaxes(title='', matches='y', row=row_idx, col=1)
#                 color_idx += 1
#     fig.update_layout(barmode='overlay', title=title, legend_title_text=series_col)
    # return fig
