import time
import streamlit as st

def execute_process(container, delay):
    for seconds in range(delay):
        container.write(f'⏳ Se passaram {seconds} de {delay}s.')
        time.sleep(1)
    container.write(f'Operação finalizada em {seconds}s. Recarrengando...')
    time.sleep(1)

def build_header():
    text ='<h1>DESABILITANDO UM BOTÃO</h1>'+\
    '''<p>O streamlit não possui uma forma de desabilitar diretamente o botão.</p>
    <p>Apesar do botão possuir um parâmetro para indicar se ele está abilitado, não é possível alterar seu valor dinamicamente.
    A estratégia usada é de substituir o botão abilitado por um desabilitado.</p>
    <p>Observe ainda o uso do <code>st.empty()</code> para gerar um <i>Place Holder</i> no qual o botão original é substituído.
    Caso não fosse utilizado este <i>Place Holder</i>, o botão seria adicionado na tela ao invés de substituir o anterior.
    </p>'''
    st.write(text, unsafe_allow_html=True)

def build_body():
    c, _ = st.columns([.5,.5])
    delay = c.slider('Delay (s)', min_value=0, max_value=5, value=4)
    col1, col2, _ = st.columns([.2,.3,.5])
    #O streamlit não dá suporte a desabilitar diretamente o botão em um 'onclick' como outras ferramentas.
    #Por isso, a estratégia usada é substituir o botão original por um desabilitado e recarregar a página após o processamento.
    button_placeholder = col1.empty()
    if button_placeholder.button('Executar'):
        #O container 'col2.empty()' é utilizado para que se substitua o seu conteúdo.
        #Se usar o container diretamente, os conteúdos são adicionados ao invés de serem substituídos.
        button_placeholder.button('Executando...', disabled=True)
        execute_process(col2.empty(), delay)
        st.experimental_rerun()

build_header()
build_body()