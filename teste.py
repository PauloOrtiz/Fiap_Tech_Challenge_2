from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import pandas as pd
import itertools

series = pd.read_csv('./src/data/ibovespa2021.csv', sep=',')
series['Data'] = pd.to_datetime(series['Data'],format='%Y-%m-%d')
series['Fechamento'] = pd.to_numeric(series['Fechamento'], errors='coerce')

# Definindo os parâmetros P, D, Q para a parte sazonal
p, d, q = 0, 1, 0  # Fixando os valores para ARIMA
P, D, Q = range(0, 2), range(0, 2), range(0, 2)
seasonal_pdq = list(itertools.product(P, D, Q, [12]))  # supondo sazonalidade anual

results = []

for param_seasonal in seasonal_pdq:
    try:
        tmp_mdl = SARIMAX(series['Fechamento'],
                          order=(p, d, q),
                          seasonal_order=param_seasonal,
                          enforce_stationarity=True,
                          enforce_invertibility=True)
        res = tmp_mdl.fit()
        results.append([(p, d, q), param_seasonal, res.aic, res.bic])
    except:
        continue

results_df = pd.DataFrame(results, columns=['pdq', 'seasonal_pdq', 'AIC', 'BIC'])

# Ordenando pelo AIC
results_df = results_df.sort_values(by='AIC', ascending=True)

print(results_df)

results_df.to_csv("./src/data/modelo_sarima.csv", index=False)
