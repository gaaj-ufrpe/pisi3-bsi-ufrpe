import pandas as pd
import streamlit as st
from enum import Enum
from pages.util.plot_pages_util import read_titanic_df
from pages.util.transformers import ScalerType, EncoderType
from pages.util.classifier import ClassifierType

def build_page():
    build_header()
    build_body()

def build_header():
    st.write('<h1>Classificação com a base do Titanic</h1>', unsafe_allow_html=True)
    # Documentação:
    # https://scikit-learn.org/stable/modules/svm.html
    # https://datagy.io/sklearn-one-hot-encode/

def build_body():
    build_controls()
    df = load_df()
    classificador_sel = st.session_state['classificador']
    classe_sel = st.session_state['classe']
    caracteristicas_sel = st.session_state['caracteristicas']
    caracteristicas_encode = [x for x in ['classe', 'sexo', 'sobreviveu'] if x in caracteristicas_sel]
    caracteristicas_scale = [x for x in ['idade','tarifa'] if x in caracteristicas_sel]
    encoder = EncoderType.get(st.session_state['encoder'])
    scaler = ScalerType.get(st.session_state['scaler'])
    df = df[caracteristicas_sel+[classe_sel]]
    classifier_type = ClassifierType.get(classificador_sel)
    classificador = classifier_type.\
        build(df, classe_sel, caracteristicas_encode, caracteristicas_scale, encoder, scaler)
    classificador.classify()

def build_controls():
    #observe o uso do parâmetro key nos controles. esta chave é usada pelo streamlit
    #para colocar o valor selecionado no dicionário (mapa) da sessão. com isso, é possível
    #recuperar o valor selecionado em qualquer página ou parte do sistema, acessando o 
    #elemento st.session_state.
    c1, c2 = st.columns([.3,.7])
    class_cols = ['classe','sobreviveu']
    class_col = c1.selectbox('Target', options=class_cols,  index=len(class_cols)-1, key='classe')
    features_opts = ['idade','tarifa','sexo','classe','sobreviveu']
    features = features_opts.copy()
    features.remove(class_col)
    features = c2.multiselect('Características *(Features)*', options=features,  default=features, key='caracteristicas')
    if len(features) < 2:
        st.error('É preciso selecionar pelo menos 2 características.')
        return
    c1, c2, c3 = st.columns(3)
    c1.selectbox('Classificador', options=ClassifierType.values(), index=0, key='classificador')
    c2.selectbox('Encoder *(Var. Discretas - Classe, Sexo e Sobreviveu)*', options=EncoderType.values(), index=0, key='encoder')
    c3.selectbox('Scaler *(Var. Contínuas - Idade e Tarifa)*', options=ScalerType.values(), index=0, key='scaler')

def load_df()->pd.DataFrame:
    df_raw = ingest_df()
    return preprocess_df(df_raw)

def ingest_df()->pd.DataFrame:
    return read_titanic_df()

def preprocess_df(df:pd.DataFrame)->pd.DataFrame:
    cols = ['idade','tarifa','sexo','classe','sobreviveu']
    df = df[cols]
    df = df[cols].copy()
    df['sexo'] = df['sexo'].fillna('null')
    df.fillna(-1,inplace=True)
    return df

build_page()