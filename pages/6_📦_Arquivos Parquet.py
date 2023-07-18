import streamlit as st
import pandas as pd
from utils import  read_df
import numpy as np
from datetime import datetime
import os

FILE_NAME = 'amostra_microdados_censo_superior_2021'

def build_header():
    text ='''<h1>PARQUET</h1>
    <p>
    Uma alternativa ao carregamento e armazenamento dos dados em arquivos '.csv' é utilizar o formato '.parquet'.
    O formato Parquet (Apache Parquet) faz parte do ecossistema de processamento de Big Data, Hadoop 
    (<a href="https://parquet.apache.org/docs/overview/motivation/">https://parquet.apache.org/docs/overview/motivation/</a>).
    Este é um formato de armazenamento de dados baseado em colunas. Isto significa que é possível carregar apenas algumas colunas 
    dos dados, ao invés de ter que carregar todas as colunas, melhorando o desempenho
    </p>
    <p>
    Uma vez que este formato foi criado para o processamento de Big Data, ele é bastante útil para o processamento paralelo.
    Por ser um formato binário, não é possível conferir o conteúdo de um arquivo neste formato em editores de texto plano.
    Por fim, comparado a dados disponibilizados em arquivos .csv, um .parquet ocupa menos espaço e é carregado mais rápido.
    Em um estudo empírico, disponível em <a href="https://dzone.com/articles/how-to-be-a-hero-with-powerful-parquet-google-and">https://dzone.com/articles/how-to-be-a-hero-with-powerful-parquet-google-and</a>,
    um arquivo parquet ocupou 87% menos espaço em disco e foi carregado 34x mais rápido. Em cenários nos quais estes dados são armazenados em nuvens,
    onde se paga por espaço utilizado, tem-se também uma economia financeira.
    </p>
    <p>Neste exemplo, estamos usando um arquivo com os dados do censo da educação superior, disponibilizado pelo INEP em 
    <a href="https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/censo-da-educacao-superior">
    https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/censo-da-educacao-superior
    </a>, para o ano de 2021. Devido ao tamanho original do arquivo (~250MB) foi necessário remover os últimos registros devido 
    à restrição do Github de o arquivo ter no máximo 50MB.
    </p>
    '''
    st.write(text, unsafe_allow_html=True)

def get_executions(extension):
    return st.session_state.get(extension,[])

def read_data(container, times, extension):
    executions = []
    for i in range(1,times+1):
        container.write(f'Leitura do {extension.upper()}. Execução {i} de {times}...')
        start_time = datetime.now()
        df = read_df(FILE_NAME, extension=extension, encoding='latin-1', low_memory=False)
        end_time = datetime.now()
        time_delta = (end_time - start_time)
        time_delta = time_delta.total_seconds()*1000
        executions.append(time_delta)
    st.session_state[extension] = executions

def build_body():
    col1, _, col2, _ = st.columns([.4,.05,.3,.25])
    times = col1.slider('Execuções', min_value=1, max_value=50, value=5)
    extensions = ['csv','parquet']
    selected_extensions = col2.multiselect('Extensões', options=extensions, default=extensions)
    if len(selected_extensions)==0:
        selected_extensions=extensions
    col1, col2, _ = st.columns([.2,.4,.5])
    button_placeholder = col1.empty()
    if button_placeholder.button('Executar'):
        button_placeholder.button('Executando...', disabled=True)
        message_holder = col2.empty()
        for extension in selected_extensions:
            read_data(message_holder, times, extension)
        st.experimental_rerun()
    st.write(f'<h3>RESULTADOS</h3>', unsafe_allow_html=True)
    execution_df = create_execution_df(selected_extensions)
    hide_index_column()
    st.table(execution_df)

def hide_index_column():
    #por algum motivo o uso do style.hide(axis='index') esconde ao gerar o html (imprimindo no terminal),
    #mas ao chamar o st.table a coluna do indice é gerada. por isso é necessário remover a coluna via css.
    #see: https://docs.streamlit.io/knowledge-base/using-streamlit/hide-row-indices-displaying-dataframe
    # CSS to inject contained in a string
    hide_table_row_index = """
                <style>
                thead tr th:first-child {display:none}
                tbody th {display:none}
                </style>
                """
    # Inject CSS with Markdown
    st.markdown(hide_table_row_index, unsafe_allow_html=True)

def create_execution_df(extensions):
    results = pd.DataFrame({
        'MÉTRICA': ['Execuções', 'Tamanho', 'Média','Desvio Padrão', 'Mediana', 'Q1', 'Q3', 'IQR'],
    })
    cols = []
    for extension in extensions:
        executions = get_executions(extension)
        if len(executions) == 0:
            continue
        col = extension.upper()
        cols.append(col)
        results[col] = [
            len(executions), 
            get_size(extension), 
            np.mean(executions), 
            np.std(executions), 
            np.median(executions), 
            np.quantile(executions,.25), 
            np.quantile(executions,.75), 
            np.quantile(executions,.75)-np.quantile(executions,.25)
        ]
    results.reset_index(drop=True, inplace=True)
    #as linhas abaixo são utilizadas para formatar a tabela. algumas destas formatações podem não funcionar caso se 
    #utilize o componente st.dataframe ao invés do componente st.table
    try:
        max_size = results.loc[1,cols].max().max()
        max_time = results.loc[2:,cols].max().max()
    except:
        max_size = max_time = None

    sdf = results.style
    sdf.bar(cmap='Blues', subset=(results.index[1],cols), vmin=0, vmax=max_size)
    sdf.bar(cmap='Blues', subset=(results.index[2:],cols), vmin=0, vmax=max_time)
    sdf.format('{:.0f}', subset=(results.index[0:], cols))
    sdf.format('{:.1f} MB', subset=(results.index[1:], cols))
    sdf.format('{:.2f} ms', subset=(results.index[2:], cols))
    sdf.hide(axis=0)
    return sdf

def get_path(extension):
    return f'data/{FILE_NAME}.{extension}'

def get_size(extension):
    stat = os.stat(get_path(extension))
    return stat.st_size/1024**2

def init_parquet():
    path = get_path('parquet')
    if not os.path.isfile(path):
        data = read_df(FILE_NAME, encoding='latin-1', low_memory=False) 
        data.to_parquet(path)

init_parquet()
build_header()
build_body()
