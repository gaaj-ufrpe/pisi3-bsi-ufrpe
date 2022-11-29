import pandas as pd
from pandas_profiling import ProfileReport
from pandas_profiling.utils.cache import cache_file

#Data:
#https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
#http://dados.recife.pe.gov.br/dataset/acidentes-de-transito-com-e-sem-vitimas
#
print('Digite o nome do arquivo:')
data_file = input()
try:
    df = pd.read_csv(f'data/{data_file}.csv', sep=',')
except:
    df = pd.read_csv(f'data/{data_file}.csv', sep=';')

profile = ProfileReport(df, title=f"{data_file} Dataset")
profile.to_file(f"{data_file}.html")