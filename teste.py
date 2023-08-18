import pandas as pd

ibovespa = pd.read_csv('./src/data/BD.csv', sep=',')
ibovespa = ibovespa.iloc[:, :2]
ibovespa = ibovespa.rename(columns={'Último':'Fechamento'})

print(ibovespa)