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

def build_body():
    df = read_titanic_df()
    build_histograma(df)

def build_histograma(df):
    c1, _, c2 = st.columns([.25,.05,.7])
    min_idade = int(np.min(df.idade))
    max_idade = int(np.max(df.idade))
    i0,i1 = c1.slider('Idade', value=(min_idade,max_idade),
                      min_value=min_idade, max_value=max_idade)
    cols = [None, 'classe', 'sexo', 'embarque']
    linha = c1.selectbox('Linhas',cols)
    coluna = c1.selectbox('Colunas',cols)
    df_aux = df.query('idade>=@i0 and idade<=@i1').copy()
    df_aux.sort_values(by=['sobreviveu'], inplace=True)
    fig = px.histogram(df_aux, x='idade', nbins=20, opacity=.75,
        color='sobreviveu', color_discrete_sequence=['red','green'],
        facet_row=linha, facet_row_spacing=.15,
        facet_col=coluna, facet_col_spacing=.15, )
    c2.plotly_chart(fig, use_container_width=True)
    # fig.update_layout(title=title, legend_title_text=series_col)
    # fig.update_xaxes(title_text=xtitle)
    # fig.update_yaxes(title_text=ytitle)

def build_header():
    st.write('<h1>Criação de Subplots</h1>', unsafe_allow_html=True)

build_page()