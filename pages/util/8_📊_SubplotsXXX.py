import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    df.sort_values(by=['classe_val','idade','id'], inplace=True)
    return df

def plot_df(df:pd.DataFrame):
    st.write('<h2>Dados do Titanic</h2>', unsafe_allow_html=True)
    st.dataframe(df)

def plot_idade(df:pd.DataFrame):
    st.markdown('<h2>Gráficos Relativos à idade</h2>', unsafe_allow_html=True)
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
    separar = c1.selectbox('Separar', options=['Não Separar']+cols)
    facet_col = separar if separar != 'Não Separar' else None 
    color_sequences = {'Azul': get_sequence(px.colors.sequential.Blues),
                      'Azul (reverso)': get_sequence(px.colors.sequential.Blues_r),
                      'Plasma': get_sequence(px.colors.sequential.Plasma),
                      'Plasma (reverso)': get_sequence(px.colors.sequential.Plasma_r),
                      'Vermelho': get_sequence(px.colors.sequential.Reds),
                      'Vermelho (reverso)': get_sequence(px.colors.sequential.Reds_r),}
    color_sequence_key = c1.selectbox('Escala de Cor', options=color_sequences.keys())
    color_sequence = color_sequences[color_sequence_key]
    complemento_titulo = f' separado por "{facet_col}"' if facet_col != 'Não Separar' else ''
    title = f'Histograma de "{series_col}" por faixa etária{complemento_titulo}.'
    xtitle ='Idade'
    ytitle='Qtd.'
    if stacked:
        fig = create_histograma_stacked(df, series_col, facet_col, color_sequence, title, xtitle, ytitle)
    else:
        fig = create_histograma_unstacked(df, series_col, facet_col, color_sequence, title, xtitle, ytitle)
    c2.plotly_chart(fig, use_container_width=True)

def create_histograma_stacked(df:pd.DataFrame, series_col:str, facet_col:str, color_sequence, title: str, xtitle: str, ytitle: str) -> go.Figure:
    if facet_col == 'Não Separar':
        order = [series_col]
        query_start = ''
    else:
        order = [series_col, facet_col]
        query_start = f'{facet_col}.notna() and'
    query = f'{query_start} {series_col}.notna()'
    df = df.query(query).copy()
    df.sort_values(by=order, inplace=True)
    fig = px.histogram(df, x='idade', nbins=20, color=series_col, opacity=.75, facet_row=facet_col,
        facet_row_spacing=.15, color_discrete_sequence=color_sequence)
    fig.update_layout(title=title, legend_title_text=series_col)
    fig.update_xaxes(title_text=xtitle)
    fig.update_yaxes(title_text=ytitle)
    return fig

def create_histograma_unstacked(df:pd.DataFrame, series_col:str, facet_col:str, color_sequence, title: str, xtitle: str, ytitle: str) -> go.Figure:
    # em alguns casos, pode ser interessante ou mesmo necessário usar a api graph objects do plotly: https://plotly.com/python/graph-objects/
    # esta api se baseia na inclusão de 'traces' sobre uma figura. ademais, propriedades e eixos da figura e dos traces podem ser customizados
    # apesar de mais complexo, o uso destes elementos diretamente permite que cada elemetno do gráfico seja ajustado individualmente.
    if facet_col is None:
        facets = ['']
        query_start = ''
    else:
        facets = ordered_vals(df,facet_col)
        query_start = f'{facet_col}==@facet and'
    facets_len = len(facets)
    fig = make_subplots(facets_len, 1, vertical_spacing=.3, x_title=xtitle, y_title=ytitle)
    series_vals = ordered_vals(df, series_col)
    series_color = {}
    row_idx = 0
    for facet in facets:
        row_idx += 1
        color_idx = 0
        for val in series_vals:
            query = f'{query_start} {series_col}==@val'
            df_aux = df.query(query).copy()
            if len(df_aux)!=0:
                opacity = .5 * (1+color_idx/10)
                str_series = str(val)
                #showlegend e o legendgroup são usados para mostrar apenas a legenda da primeira linha
                #e permitir que a interação com a legenda altere o estado de todos os subplots
                if str_series in series_color.keys():
                    color = series_color[str_series] 
                    showlegend = False
                else:
                    color = f'{color_sequence[color_idx]}'
                    series_color[str_series] = color
                    showlegend = True
                hist = go.Histogram(name=str_series, x=df_aux['idade'], 
                                    xbins=dict(start=0,end=80,size=5), legendgroup=val, showlegend=showlegend,
                                    marker={'color': color, 'opacity':opacity})
                fig.add_trace(hist, row=row_idx, col=1)
                fig.add_annotation(xref='x domain',yref='y domain',x=0, y=1.1, showarrow=False,
                                text=f'{facet_col}: {facet}', row=row_idx, col=1)
                fig.update_xaxes(matches='x', row=row_idx, col=1)
                fig.update_yaxes(title='', matches='y', row=row_idx, col=1)
                color_idx += 1
    fig.update_layout(barmode='overlay', title=title, legend_title_text=series_col)
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
    df = ingest_data()
    df = transform_data(df)
    plot_df(df)
    plot_idade(df)

build_header()
build_body()