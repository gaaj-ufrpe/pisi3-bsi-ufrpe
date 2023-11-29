import pandas as pd
import streamlit as st
from enum import Enum
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pages.util.transformers import EncoderType, ScalerType, Transformer
import seaborn as sns
import matplotlib.pyplot as plt

class ClassifierType(Enum):
    KNN = 'K-Nearest Neighbors', lambda: KNeighborsClassifier(), 
    RANDOM_FOREST = 'Random Forest', lambda: RandomForestClassifier(random_state=42),
    SVM_LINEAR_SCV = 'SVM - LinearSVC', lambda: LinearSVC(),
    SVM_SCV_LINEAR_KERNEL = 'SVM - SVC com Kernel Linear', lambda: SVC(kernel='linear'),
    SVM_SCV_RBF_KERNEL = 'SVM - SVC com Kernel RBF', lambda: SVC(kernel='rbf')

    def __init__(self, description:str, builder:callable):
       self.description = description
       self.builder = builder

    @classmethod
    def values(cls):
        return [x.description for x in ClassifierType]

    @classmethod
    def get(cls, description):
        result =  [x for x in ClassifierType if x.description == description]
        return None if len(result) == 0 else result[0]
    
    def build(self, df:pd.DataFrame, classe:str,
              encode:list[str], scale:list[str],
              encoder_type: EncoderType, scaler_type: ScalerType,
              seed=42, test_size=.20):
        return Classifier(self.description, self.builder, df, classe, 
                          encode, scale, encoder_type, scaler_type, seed, test_size)
    
    def __str__(self) -> str:
        return self.description

class Classifier:
    def __init__(self, description: str, builder:callable, df:pd.DataFrame, target:str, 
                 encode:list[str], scale:list[str], encoder_type: EncoderType, scaler_type: ScalerType, 
                 seed=42, test_size=.20):
        self.description = description
        self.seed = seed
        df, X, y = self.__preprocess(df, target)
        self.df = df
        lbl_enc = LabelEncoder()
        y = lbl_enc.fit_transform(y)
        self.target_classes = lbl_enc.classes_
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
        #O fit do encoder e do scaler tem que ser feito apenas sobre os dados do conjunto de treino
        #Caso seja feito com todo o conjunto de dados, o encoder e o scaler serão 'contaminados' com os dados usados no treino.
        self.transformer = Transformer(self.X_train, encode, scale, encoder_type, scaler_type)
        self.X_train = self.transformer.transform(self.X_train)
        self.X_test = self.transformer.transform(self.X_test)
        self.sklearn_classifier = builder()
        self.sklearn_classifier.fit(self.X_train, self.y_train)

    def __preprocess(self, df:pd.DataFrame, target:str):
        df = df.copy().dropna()
        return df, df[df.columns.difference([target])],df[target]
    
    def classify(self):
        y_train_pred = self.sklearn_classifier.predict(self.X_train)
        y_test_pred = self.sklearn_classifier.predict(self.X_test)
        st.write('''<b>ATENÇÃO:</b> O classificador deve ser treinado com o conjunto de treino e testado com o conjunto de teste. 
                    Difícil não?! <b>Treino é treino, <del>jogo é jo...</del> teste é teste!</b>
                    ''', unsafe_allow_html=True)
        st.write('''<b>ATENÇÃO 2:</b> Para a etapa de treinamento, deve-se balancear as classes, especialmente no caso
                 de dados muito desbalanceados, como no caso de eventos raros (ex.: fraude em cartão, doenças raras etc.).
                    ''', unsafe_allow_html=True)
            
        st.write(f'<h2>{self.description}</h2>', unsafe_allow_html=True)
        c1, _, c2 = st.columns([.49,.02,.49])
        with c1:
            self.__report('TREINO', '<div style="color: red; font-size: 1.5em">Treino é treino!</div>', 
                        self.y_train, y_train_pred)
        with c2:
            self.__report('TESTE', '<div style="font-family: cursive; font-size: 1.1em">Jogo é jogo!</div>', 
                        self.y_test, y_test_pred)
        st.write('')
        with st.expander('Dados Brutos'):
            st.dataframe(self.df)
        with st.expander('Dados Processados'):
            c1, _, c2 = st.columns([.49,.02,.49])
            c1.write('<h3>Treino</h3>', unsafe_allow_html=True)
            c1.dataframe(self.X_train)
            c2.write('<h3>Teste</h3>', unsafe_allow_html=True)
            c2.dataframe(self.X_test)

    def __report(self, title, desc, y_true, y_pred):
        st.write(f'<h3>{title}</h3>', unsafe_allow_html=True)
        self.__build_confusion_matrix(y_true, y_pred)
        self.__build_results(desc, y_true, y_pred)

    def __build_results(self, desc, y_true, y_pred):
        report = classification_report(y_true, y_pred, output_dict=True)
        accuracy, support, df_report = self.__build_report_df(report)
        st.write(f'''
                    <b>Accuracy: {accuracy:.4%}</b><br/>
                    Suppport: {support:.0f}
                    ''', unsafe_allow_html=True)
        st.write('')
        df_report = df_report.style.format({'precision': '{:.2%}', 'recall': '{:.2%}', 'f1-score': '{:.2%}', 'support': '{:.0f}'})
        st.dataframe(df_report)
        st.write(desc, unsafe_allow_html=True)

    def __build_confusion_matrix(self, y_true, y_pred):
        c, _ = st.columns([.7,.3])
        with c:
            matrix = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(matrix, annot=True, fmt='d', ax=ax, cmap='Blues', cbar=False)
            ax.set_xlabel('Valores Previstos')
            ax.set_ylabel('Valores Reais')
            ax.xaxis.set_ticklabels(self.target_classes)
            ax.yaxis.set_ticklabels(self.target_classes)
            st.write('Matriz de Confusão')
            st.pyplot(fig)

    def __build_report_df(self, report):
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

    # def score(self, classificador, X_train, X_test, y_train, y_test):
    #     return classificador.score(X_train, y_train), classificador.score(X_test, y_test)

    # def accuracy(self, y_train, y_train_pred, y_test, y_test_pred):
    #     return accuracy_score(y_train, y_train_pred), accuracy_score(y_test, y_test_pred)

    def __str__(self) -> str:
        return self.description
