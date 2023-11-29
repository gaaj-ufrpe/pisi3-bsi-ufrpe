import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def build_page():
    build_header()
    build_body()

def build_header():
    st.write('<h1>Matriz de Confusão</h1>', unsafe_allow_html=True)
    st.write('''
            <br>
             Métricas:
             <br>
             Acurácia (A) = (TP + TN) / (TP+TN+FP+FN)
             <br>
             Precisão (P) = TP/(TP+FP)
             <br>
             Recall (R) = TP/(TP+FN) 
             <br>
             F1-Score (F1) = 2*P*R/(P+R) 
             <br>
             A acurácia não é boa para amostras desbalanceadas. 
             A precisão despresa os falsos negativos, enquanto o recall despresa os falsos positivos.
             Logo, precisão deve ser usada quando a ocorrência de falsos negativos não for um problema (teste de gravidez caseiro),
             enquanto o recall deve ser usado quando o falso positivo não for um problema (teste de covid caseiro). 
             Note que nestes casos, o teste de gravidez poderia ser refeito sem problema para o usuário, especialmente considerando um teste de sangue, enquanto 
             o teste de covid dando um falso positivo, a pessoa não sairia de casa. Já um falso negativo, a pessoa sairia de casa espalhando o vírus.
             O F1-Score é a média harmônica da precisão e recall.
            <br>
            <br>
             Artigo: 
             <a href="https://medium.com/@ejunior029/principais-m%C3%A9tricas-de-classifica%C3%A7%C3%A3o-de-modelos-em-machine-learning-94eeb4b40ea9">
             Principais métricas de avaliação de modelos em Machine Learning
             </a>
            <br>
            ''', unsafe_allow_html=True)
    
def build_body():
    y_real = ['positive','positive','positive','negative','negative','positive','positive','positive','positive','negative']
    y_predict1 = ['positive','positive','positive','positive','positive','positive','positive','positive','positive','positive']
    y_predict2 = ['negative','positive','positive','positive','positive','positive','positive','positive','positive','positive']
    y_predict3 = ['positive','positive','positive','positive','positive','positive','positive','positive','positive','negative']
    y_predict4 = ['negative','positive','positive','positive','positive','positive','positive','positive','positive','negative']
    predicts = {
        'Previsto Classificador 1': y_predict1,
        'Previsto Classificador 2': y_predict2,
        'Previsto Classificador 3': y_predict3,
        'Previsto Classificador 4': y_predict4,
    }
    for name, y_pred in predicts.items():
        with st.expander(name):
            c1,_,c2,_,c3 = st.columns([.3,.05,.35,.05,.25])
            print_data(y_real, y_pred, c1)
            vals = ['positive','negative']
            matrix = confusion_matrix(y_real, y_pred, labels=vals)
            print_confusion_matrix(matrix, vals, c2)
            print_metrics(y_real, y_pred, matrix, vals, c3)

def print_data(y_real, y_pred, container):
    container.write('Previsão do Classificador')
    container.dataframe(pd.DataFrame(data={'Real':y_real, 'Previsto':y_pred}), use_container_width=True)

def print_confusion_matrix(matrix, vals, container):
    fig, ax = plt.subplots()
    sns.heatmap(matrix, annot=True, fmt='d', ax=ax, cmap='Blues', cbar=False)
    ax.set_title('Matriz de Confusão')
    ax.set_xlabel('Valores Previstos')
    ax.set_ylabel('Valores Reais')
    ax.xaxis.set_ticklabels(vals)
    ax.yaxis.set_ticklabels(vals)
    container.write('Matriz de Confusão')
    container.pyplot(fig)

def print_metrics(y_real, y_pred, matrix, vals, container):
    container.write(f"A = {accuracy_score(y_real, y_pred)}")
    container.write(f"P = {precision_score(y_real, y_pred, labels=vals, pos_label='positive')}")
    container.write(f"R = {recall_score(y_real, y_pred, labels=vals, pos_label='positive')}")
    container.write(f"F1 = {f1_score(y_real, y_pred, labels=vals, pos_label='positive')}")
    results = classification_report(y_real, y_pred, labels=vals, zero_division=0)
    st.text_area(label='Classification Report', height=240, disabled=True, value=results)

build_page()