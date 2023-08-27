import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

ibovespa = pd.read_csv('./src/data/ibovespa.csv', sep=',')
ibovespa['Data'] = pd.to_datetime(ibovespa['Data'],format='%Y-%m-%d')
ibovespa['Fechamento'] = pd.to_numeric(ibovespa['Fechamento'], errors='coerce')
ibovespa['Data'] = pd.to_datetime(ibovespa['Data'])
ibovespa.set_index('Data', inplace=True)

data_diff = ibovespa.diff().dropna()

# Suponha que seus dados estejam em um DataFrame chamado df e a coluna de interesse seja 'Fechamento'


# Dividir os dados em treino e teste
train_size = int(len(data_diff) * 0.8)
train, test = data_diff[0:train_size], data_diff[train_size:]

best_score, best_cfg = float("inf"), None

# Defina os intervalos de parâmetros
p_values = range(0, 3)
d_values = [0]
q_values = range(0, 3)

for p in p_values:
    for d in d_values:
        for q in q_values:
            order = (p,d,q)
            try:
                model = ARIMA(train, order=order)
                model_fit = model.fit()
                predictions = model_fit.forecast(steps=len(test))
                error = mean_squared_error(test, predictions)
                if error < best_score:
                    best_score, best_cfg = error, order
                print('ARIMA%s MSE=%.3f' % (order, error))
            except:
                continue

print('Melhor ARIMA%s MSE=%.3f' % (best_cfg, best_score))
