import streamlit as st
from utils import df_names, read_df

def build_header():
    text ='''<h1>AGRUPAMENTO</h1>
    <p>A ordem do agrupamento é importante para o resultado. 
    Faça os testes alterando esta ordem para ver a diferença no dataframe gerado.
    Observe que ao agrupar, as colunas utilizadas no <code>df.groupby</code> são transformadas em índices.
    Para torná-las colunas novamente, é preciso chamara o método <code>reset_index</code> sobre o dataframe gerado.</p>
    <p>No pandas, utiliza-se a notação <code>df[cols].groupby(by=group_cols).agg(tot_fun)</code>, onde:<br>
    <ul>
        <li><b>df</b> é o dataframe;</li>
        <li><b>cols</b> são as colunas a serem selecionadas (colunas a serem agrupadas e as totalizadas);</li>
        <li><b>group_cols</b> são as colunas a serem agrupadas;</li>
        <li><b>tot_fun</b> a função de agrupamento (vide lib numpy).</li>
    </ul></p>'''
    st.write(text, unsafe_allow_html=True)
    
def build_body():
    col1, col2 = st.columns([.3,.7])
    df_name = col1.selectbox('Dataset', df_names())
    df = read_df(df_name)
    cols = list(df.columns)
    group_cols = col2.multiselect('Agrupar', options=cols, default=cols[0])
    tot_opts = [x for x in cols if x not in group_cols]
    tot_fun = col1.selectbox('Função', options=['count','nunique','sum','mean'])
    tot_cols = col2.multiselect('Totalizar', options=tot_opts, default=tot_opts[0])
    select_cols = tot_cols+group_cols
    df_grouped = df[select_cols].groupby(by=group_cols).agg(tot_fun)
    st.write('Data frame:')
    st.dataframe(df_grouped, use_container_width=True)

build_header()
build_body()
