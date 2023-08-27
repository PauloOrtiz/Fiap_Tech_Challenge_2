import pandas as pd

ibovespa = pd.read_csv('./src/data/BD.csv', sep=',')
ibovespa = ibovespa.iloc[:, :2]
ibovespa = ibovespa.rename(columns={'Último':'Fechamento'})
ibovespa['Data'] = pd.to_datetime(ibovespa['Data'],format='%d.%m.%Y')
ibovespa['Fechamento'] = ibovespa['Fechamento'].astype(str)
ibovespa['Fechamento'] = ibovespa['Fechamento'].str.replace('.', '')
ibovespa['Fechamento'] = ibovespa['Fechamento'].astype(int)
ibovespa = ibovespa[::-1]

ibovespa.to_csv("src/data/ibovespa.csv", index= False)


'''
ibovespa = pd.read_csv('./src/data/BD.csv', sep=',')
ibovespa = ibovespa.iloc[:, :2]
ibovespa = ibovespa.rename(columns={'Último':'Fechamento'})
ibovespa['Data'] = pd.to_datetime(ibovespa['Data'],format='%d.%m.%Y')
ibovespa = ibovespa[::-1]
ibovespa['Fechamento'] = ibovespa['Fechamento'].astype(str)

# Removendo o ponto (.) usado como separador de milhares
ibovespa['Fechamento'] = ibovespa['Fechamento'].str.replace('.', '')

# Convertendo a coluna 'Fechamento' de volta para float
ibovespa['Fechamento'] = ibovespa['Fechamento'].astype(int)
'''