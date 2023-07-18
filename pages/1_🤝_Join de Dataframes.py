import streamlit as st
import pandas as pd

mergeing_dfs = []

def build_header():
    text ='''<h1>JOIN</h1>
    <p>Assim como o SQL possui diversas opções para efetuar o join de tabelas, o Pandas também dá suporte a este tipo de operação.
    Estas funções são equivalente às operações realizadas com conjuntos (união, interseção etc.).</p>
    <p>
    <p>Para uma documentação completa, acesse <a href="https://pandas.pydata.org/docs/user_guide/merging.html">https://pandas.pydata.org/docs/user_guide/merging.html</a></p>
    '''
    st.write(text, unsafe_allow_html=True)

def build_body():
    init_dfs()
    with st.expander('Concat'):
        build_concat()
    with st.expander('merge'):
        build_merge()

def print_dfs(dfs):
    st.write('<h5>Data Frames</h5>', unsafe_allow_html=True)
    cols = st.columns(len(dfs))
    for idx, (c, df) in enumerate(zip(cols,dfs)):
        c.write(f'Data Frame {idx+1}')
        c.dataframe(df, use_container_width=True)

def build_concat_lines():
    st.write('<h4>Adicionando Linhas</h4>', unsafe_allow_html=True)
    print_dfs(mergeing_dfs)
    c1, c2 = st.columns([.5,.5])
    c1.write('Código: <code>pd.concat([df1,df2,df3])</code>', unsafe_allow_html=True)
    c1.write('''Descrição: 
    <p>A função concat do pandas concatena dois ou mais dataframes.
    Apesar de possuir diversos parâmetros, seu uso mais simples precisa que os dataframes
    possuam a mesma quantidade de colunas e que elas tenham o mesmo nome, adicionando as linhas dos dataframes uma após a outra.</p>
    <p>Apesar de poder ser usado para fazer o "merge", outras funções específicas para esta finalidade podem ser mais indicadas.</p>
    ''', unsafe_allow_html=True)
    c2.dataframe(pd.concat(mergeing_dfs))

def build_concat_columns():
    new_dfs = []
    for idx, df in enumerate(mergeing_dfs):
        ndf = df.copy().reset_index(drop=True)
        ndf.columns = [f'A_df{idx}',f'B_df{idx}',f'C_df{idx}',f'D_df{idx}']
        new_dfs.append(ndf)
    st.write('<h4>Adicionando Colunas</h4>', unsafe_allow_html=True)
    print_dfs(new_dfs)
    c1, c2 = st.columns([.5,.5])
    c1.write('''Código: 
    <code>\\
    df1 = df1.copy().reset_index(drop=True)\\
    df2 = df1.copy().reset_index(drop=True)\\
    df3 = df1.copy().reset_index(drop=True)\\
    df1.columns = ['A_df1','B_df1','C_df1','D_df1']\\
    df2.columns = ['A_df2','B_df2','C_df2','D_df2']\\
    df3.columns = ['A_df3','B_df3','C_df3','D_df3']\\
    pd.concat([df1,df2,df3], axis=1)\\
    </code>
    ''', unsafe_allow_html=True)
    c1.write('''Descrição: 
    <p>Utilizando a função concat passando o <code>axis=1</code>, indica-se que a concatenação deve ocorrer pelo eixo das colunas.
    O padrão é o <code>axis=0</code>, indicando a concatenação de linhas.</p>
    <p>Observe no código gerado que foi feita uma cópia do dataframe original e o índice foi resetado e removido.
    A cópia precisa ser feita, pois caso contrário as mudanças como no nome das colunas seriam refletidas não apenas neste exemplo,
    mas em todos os demais lugares que usassem o dataframe.</p>
    <p>Já o reset e drop dos índices precisam ser feitos para que a concatenação das colunas ocorra corretamente. Os merges e merges
    das tabelas sempre ocorrem pelos índices no pandas (mesmo que estes sejam criados implicitamente dentro da função). Como os valores dos índices de cada dataframe
    são diferentes, a concatenação sem a mudança dos índices iria gerar linhas diferentes para cada data frame. Faça testes para observar o comportamento.</p>
    ''', unsafe_allow_html=True)
    c2.dataframe(pd.concat(new_dfs,axis=1))

def build_concat():
    build_concat_lines()
    build_concat_columns()

def build_merge_left(new_dfs):
    st.write('<h4>Left merge</h4>', unsafe_allow_html=True)
    st.write('''Descrição: 
    <p>Observe que o dataframe resultante permanece com TODAS as linhas do dataframe à esquerda (left) mesmo que o valor da coluna "KEY" não esteja presente
    no dataframe passado como parâmetro.</p>
    ''', unsafe_allow_html=True)
    print_dfs(new_dfs)
    df1, df2, df3 = new_dfs
    c1, c2, c3 = st.columns(3)
    c1.write(f'Código: <code>df1.merge(df2, on="KEY", how="left")</code>', unsafe_allow_html=True)
    c1.dataframe(df1.merge(df2, on="KEY", how="left"))
    c2.write(f'Código: <code>df1.merge(df3, on="KEY", how="left")</code>', unsafe_allow_html=True)
    c2.dataframe(df1.merge(df3, on="KEY", how="left"))
    c3.write(f'Código: <code>df2.merge(df3, on="KEY", how="left")</code>', unsafe_allow_html=True)
    c3.dataframe(df2.merge(df3, on="KEY", how="left"))

def build_merge_inner(new_dfs):
    st.write('<h4>Inner merge</h4>', unsafe_allow_html=True)
    st.write('''Descrição: 
    <p>Observe que o dataframe resultante permanece apenas com as linhas cuja coluna "KEY" possui valores iguais nos dataframes para os quais foi feito o merge.</p>
    ''', unsafe_allow_html=True)
    print_dfs(new_dfs)
    df1, df2, df3 = new_dfs
    c1, c2, c3 = st.columns(3)
    c1.write(f'Código: <code>df1.merge(df2, on="KEY", how="inner")</code>', unsafe_allow_html=True)
    c1.dataframe(df1.merge(df2, on="KEY", how="inner"))
    c2.write(f'Código: <code>df1.merge(df3, on="KEY", how="inner")</code>', unsafe_allow_html=True)
    c2.dataframe(df1.merge(df3, on="KEY", how="inner"))
    c3.write(f'Código: <code>df2.merge(df3, on="KEY", how="inner")</code>', unsafe_allow_html=True)
    c3.dataframe(df2.merge(df3, on="KEY", how="inner"))

def build_merge():
    st.write('''<p>O método <code>merge</code> do dataframe é equivalente ao método <code>pd.merge</code>, porém o primeiro é chamado diretamente no dataframe,
    enquanto o segundo é chamado pela biblioteca do pandas, passando os dois dataframes. Este método cria os índices internamente, diferentemente do método join. 
    Por isso, este método é mais simples e conveniente de usar na maior parte dos casos.</p>
    ''', unsafe_allow_html=True)
    new_dfs = [df.copy().reset_index(drop=True) for df in mergeing_dfs]
    for idx, df in enumerate(new_dfs):
        df.columns = map(lambda x: f'{x}_df{idx+1}', df.columns)
        if idx == 0:
            df['KEY'] = ['K0','K0','K1','K1']
        elif idx == 1:
           df['KEY'] = ['K0','K0','K0','K0']
        else:
           df['KEY'] = ['K1','K1','K1','K1']
    build_merge_left(new_dfs)
    build_merge_inner(new_dfs)


def init_dfs():
    df1 = pd.DataFrame(
    {
        "A": ["A0", "A1", "A2", "A3"],
        "B": ["B0", "B1", "B2", "B3"],
        "C": ["C0", "C1", "C2", "C3"],
        "D": ["D0", "D1", "D2", "D3"],
    },
    index=[0, 1, 2, 3],
    )
    mergeing_dfs.append(df1)
    df2 = pd.DataFrame(
        {
            "A": ["A4", "A5", "A6", "A7"],
            "B": ["B4", "B5", "B6", "B7"],
            "C": ["C4", "C5", "C6", "C7"],
            "D": ["D4", "D5", "D6", "D7"],
        },
        index=[4, 5, 6, 7],
    )
    mergeing_dfs.append(df2)
    df3 = pd.DataFrame(
        {
            "A": ["A8", "A9", "A10", "A11"],
            "B": ["B8", "B9", "B10", "B11"],
            "C": ["C8", "C9", "C10", "C11"],
            "D": ["D8", "D9", "D10", "D11"],
        },
        index=[8, 9, 10, 11],
    )
    mergeing_dfs.append(df3)


build_header()
build_body()