import time
from pandas_profiling import ProfileReport
import streamlit as st

def execute_process(container):
    for seconds in range(delay):
        container.write(f'⏳ Se passaram {seconds} de {delay}s.')
        time.sleep(1)
    container.write(f'Operação finalizada em {seconds}s. Recarrengando...')
    time.sleep(1)

c, _ = st.columns([.5,.5])
delay = c.slider('Tempo da operação (s)', min_value=0, max_value=5, value=4)

col1, col2, _ = st.columns([.2,.3,.5])
#O streamlit não dá suporte a desabilitar diretamente o botão em um 'onclick' como outras ferramentas.
#Por isso, a estratégia usada é substituir o botão original por um desabilitado e recarregar a página após o processamento.
button_placeholder = col1.empty()
if button_placeholder.button('Executar'):
    #O container 'col2.empty()' é utilizado para que se substitua o seu conteúdo.
    #Se usar o container diretamente, os conteúdos são adicionados ao invés de serem substituídos.
    button_placeholder.button('Executando...', disabled=True)
    execute_process(col2.empty())
    st.experimental_rerun()
