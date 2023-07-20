# PISI3 - BSI - UFRPE
por: Gabriel Alves (gabriel.alves@ufrpe.br)

Este projeto é utilizado para as aulas da disciplina de Projeto Interdisciplinar para Sistemas de Informação 3 (pisi3), do 3° período do curso de Bacharelado em Sistemas de Informação (BSI) da sede, da Universidade Federal Rural de Pernambuco (UFRPE).

## Instalação:
<ol>
  <li>Instale o python</li>

  <li>Execute a linha abaixo para criar um ambiente virtual do python para o projeto. Observe que a pasta <code>venv</code> está no <code>.gitignore</code>.
    <code>python -m venv venv</code>
  </li>

  <li>Entre no terminal e execute:
    <code>python -m pip install --upgrade pip</code>
  </li>

  <li>Baixe este projeto e descompacte o conteudo em uma pasta <code>teste_streamlit</code>. Entre na pasta.</li>

  <li>Execute:
    <code>pip install -r requirements.txt --upgrade</code>
  </li>

  <li>Execute:
    <code>streamlit run Home.py</code>
  </li>
</ol>

## Informações Adicionais:

Este projeto tem 3 branches com códigos distintos:
<ol>
  <li>main: código com várias funcionalidades diferentes: sidebar, pandas, arquivos parquet, ydata-profile, clusterização etc.</li>
  <li>basics: código inicial com o uso básico do streamlit. Possui uma caixa de texto e exibe uma mensagem ao preenchê-la.</li>
  <li>theme: código para mostrar como funciona o uso de temas no streamlit.</li>
</ol>
Disponível em: https://gaaj-pisi3-bsi-ufrpe.streamlit.app/
