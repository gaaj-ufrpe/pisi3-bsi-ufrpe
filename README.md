# PISI3 - BSI - UFRPE

por: Gabriel Alves (<gabriel.alves@ufrpe.br>)

Este projeto é utilizado para as aulas da disciplina de Projeto Interdisciplinar para Sistemas de Informação 3 (PISI3), do 3° período do curso de Bacharelado em Sistemas de Informação (BSI) da Sede, da Universidade Federal Rural de Pernambuco (UFRPE).

## Instalação

* Instale o VSCode.
* Efetue o clone do projeto: `CTRL+SHIFT+P > Git:Clone > Clone from GitHub > https://github.com/gaaj-ufrpe/pisi3-bsi-ufrpe`
* Instale o python.
* Acesse a aba "Terminal" disponível na parte inferior do VSCode.
* Execute a linha abaixo para criar um ambiente virtual do python para o projeto. Observe que a pasta `venv` está no `.gitignore`.
    `python -m venv venv`
* Atualize o pip:
    `python -m pip install --upgrade pip`  
* Instale as libs necessárias para o projeto:
    `pip install -r requirements.txt --upgrade`
* Rode o sistema:
    `streamlit run Home.py`

## Informações Adicionais

Este projeto tem 3 branches com códigos distintos:

* main: código com várias funcionalidades diferentes: sidebar, pandas, arquivos parquet, ydata-profile, clusterização etc.
* basics: código inicial com o uso básico do streamlit. Possui uma caixa de texto e exibe uma mensagem ao preenchê-la.
* theme: código para mostrar como funciona o uso de temas no streamlit.

Repositório: <https://github.com/gaaj-ufrpe/pisi3-bsi-ufrpe>

Disponível em: <https://gaaj-pisi3-bsi-ufrpe.streamlit.app>

Classroom: <https://classroom.google.com/c/NjExNTAzOTU4MDQy?cjc=7qgaz7u>

Acesse: <https://bsi.ufrpe.br>

Contato: <gabriel.alves@ufrpe.br>
