import pandas as pd
import streamlit as st
from enum import Enum
from pages.util.plot_pages_util import read_titanic_df
from pages.util.transformers import Scalers, Encoders
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, classification_report

class ClassificationResults:
    def __init__(self, score, accuracy, confusion_matrix):
       self.score = score
       self.accuracy = accuracy
       self.confusion_matrix = confusion_matrix

class Classifiers(Enum):
    KNN = 'K-Nearest Neighbors', 
    RANDOM_FOREST = 'Random Forest',
    SVM_LINEAR_SCV = 'SVM - LinearSVC',
    SVM_SCV_LINEAR_KERNEL = 'SVM - SVC com Kernel Linear',
    SVM_SCV_RBF_KERNEL = 'SVM - SVC com Kernel RBF'

    def __init__(self, description):
       self.description = description

    @classmethod
    def values(self):
        return [x.description for x in Classifiers]

    @classmethod
    def get(self, description):
        result =  [x for x in Classifiers if x.description == description]
        return None if len(result) == 0 else result[0]

    def create_classifier(self) -> ClassifierMixin:
        if self == Classifiers.KNN:
            return KNeighborsClassifier()
        elif self == Classifiers.RANDOM_FOREST:
            return RandomForestClassifier()
        elif self == Classifiers.SVM_LINEAR_SCV:
            return LinearSVC()
        elif self == Classifiers.SVM_SCV_LINEAR_KERNEL:
            return SVC(kernel='linear')
        elif self == Classifiers.SVM_SCV_RBF_KERNEL:
            return SVC(kernel='rbf')
        else:
            raise TypeError('Tipo de classificador não suportado.')
    
    def classify(self, df:pd.DataFrame, caracteristicas:list[str], classe:str):
        df = df.copy().dropna()
        X = df[caracteristicas]
        X = transform_df(X)
        y = df[classe]
        SEED = 42
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = SEED)
        classificador = self.create_classifier()
        classificador.fit(X_train, y_train)
        y_train_pred = classificador.predict(X_train)
        y_test_pred = classificador.predict(X_test)
        st.write('''<b>ATENÇÃO:</b> O classificador deve ser treinado com o conjunto de treino e testado com o conjunto de teste. 
                 Difícil não?! <b>Treino é treino, <del>jogo é jo...</del> teste é teste!</b>
                 ''', unsafe_allow_html=True)
        c1, _, c2 = st.columns([.49,.02,.49])
        with c1:
            self.report('TREINO', '<div style="color: red; font-size: 1.5em">Vc <b style="font-size: 1.5em">testou</b> com o conjunto de <b style="font-size: 1.5em">treino</b>. Passe no RH!!!</div>', 
                        y_train, y_train_pred, caracteristicas, classe)
        with c2:
            self.report('TESTE', '<div style="font-family: cursive; font-size: 1.1em">Parabéns, vc <b style="font-size: 1.5em">testou</b> com o conjunto de <b style="font-size: 1.5em">teste</b>. Tá vendo que não é difícil?!</div>', 
                        y_test, y_test_pred, caracteristicas, classe)

    def report(self, title, desc, y_true, y_pred, caracteristicas, classe):
        report = classification_report(y_true, y_pred, output_dict=True)
        accuracy, support, df_report = self.build_report_df(report)
        title = f'{self.description} ({title})'
        st.write(f'<h2>{title}</h2>', unsafe_allow_html=True)
        st.write(f'''
                    <b>Accuracy: {accuracy:.4%}</b><br/>
                    Suppport: {support:.0f}
                 ''', unsafe_allow_html=True)
        st.write(f'')
        df_report = df_report.style.format({'precision': '{:.2%}', 'recall': '{:.2%}', 'f1-score': '{:.2%}', 'support': '{:.0f}'})
        st.dataframe(df_report)
        st.write(desc, unsafe_allow_html=True)

    def build_report_df(self, report):
        df_dict = {
            'classe': [],
            'precision': [],
            'recall': [],
            'f1-score': [],
            'support': [],
        }
        accuracy = 0
        support = 0
        for k, v in report.items():
            if k == 'accuracy':
                accuracy = v
            else:
                df_dict['classe'].append(k)
                df_dict['precision'].append(v['precision'])
                df_dict['recall'].append(v['recall'])
                df_dict['f1-score'].append(v['f1-score'])
                s = int(v['support'])
                df_dict['support'].append(s)
                support = s
        df_report = pd.DataFrame(data=df_dict)
        return accuracy,support,df_report

    def score(self, classificador, X_train, X_test, y_train, y_test):
        return classificador.score(X_train, y_train), classificador.score(X_test, y_test)

    def accuracy(self, y_train, y_train_pred, y_test, y_test_pred):
        return accuracy_score(y_train, y_train_pred), accuracy_score(y_test, y_test_pred)

    def __str__(self) -> str:
        return self.description

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
    classify(load_df())

def build_controls():
    #observe o uso do parâmetro key nos controles. esta chave é usada pelo streamlit
    #para colocar o valor selecionado no dicionário (mapa) da sessão. com isso, é possível
    #recuperar o valor selecionado em qualquer página ou parte do sistema, acessando o 
    #elemento st.session_state.
    c1, c2 = st.columns([.3,.7])
    class_cols = ['classe','sobreviveu']
    class_col = c1.selectbox('Classe', options=class_cols,  index=len(class_cols)-1, key='classe')
    features_opts = ['idade','tarifa','sexo','classe','sobreviveu']
    features = features_opts.copy()
    features.remove(class_col)
    features = c2.multiselect('Características', options=features,  default=features, key='caracteristicas')
    if len(features) < 2:
        st.error('É preciso selecionar pelo menos 2 características.')
        return None
    c1, c2, c3 = st.columns(3)
    c1.selectbox('Classificador', options=Classifiers.values(), index=0, key='classificador')
    c2.selectbox('Encoder', options=Encoders.values(), index=0, key='encoder')
    c3.selectbox('Scaler', options=Scalers.values(), index=0, key='scaler')

def load_df()->pd.DataFrame:
    df_raw = ingest_df()
    return preprocess_df(df_raw)

def ingest_df()->pd.DataFrame:
    return read_titanic_df()

def preprocess_df(df:pd.DataFrame)->pd.DataFrame:
    cols = ['idade','tarifa','sexo','classe_val','sobreviveu_val']
    df = df[cols]
    df = df[cols].copy()
    df['sexo'] = df['sexo'].fillna('null')
    df.fillna(-1,inplace=True)
    df.rename(columns={'classe_val':'classe',
                       'sobreviveu_val':'sobreviveu'}, inplace=True)
    return df
    
def transform_df(df:pd.DataFrame)->pd.DataFrame:
    df = scale(df)
    df = encode(df)
    return df

def get_caracteristicas(df:pd.DataFrame, caracteristicas:list[str]):
    caracteristicas_sel = st.session_state['caracteristicas']
    return [x for x in caracteristicas if x in caracteristicas_sel]

def scale(df:pd.DataFrame):
    caracteristicas = get_caracteristicas(df, ['idade','tarifa'])
    if len(caracteristicas) == 0:
        return df
    scaler = st.session_state['scaler']
    scaler = Scalers.get(scaler)
    for c in caracteristicas:
        df = scaler.fit_transform(df, c)
    return df

def encode(df:pd.DataFrame):
    caracteristicas = get_caracteristicas(df, ['sexo'])
    if len(caracteristicas) == 0:
        return df
    encoder = st.session_state['encoder']
    encoder = Encoders.get(encoder)
    return encoder.fit_transform(df, caracteristicas)

def classify(df:pd.DataFrame) -> pd.DataFrame:
    classificador = st.session_state['classificador']
    classificador = Classifiers.get(classificador)
    #Considerando o encapsulamento (base da OO), os princípios da coesão e baixo acoplamento e o SOLID,
    #a recuperação da classe e das características deveria estar aqui, ou em outro lugar
    #como no próprio método de classificação do enum? (e olhe que nem estamos usando OO :P)
    classe = st.session_state['classe']
    caracteristicas = st.session_state['caracteristicas']
    classificador.classify(df, caracteristicas, classe)

build_page()