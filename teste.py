from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import pandas as pd
import itertools

series = pd.read_csv('./src/data/ibovespa2021.csv', sep=',')
series['Data'] = pd.to_datetime(series['Data'],format='%Y-%m-%d')
series['Fechamento'] = pd.to_numeric(series['Fechamento'], errors='coerce')

# Definindo os parâmetros p, d, q, P, D, Q
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]  # supondo sazonalidade anual

best_aic = np.inf
best_pdq = None
best_seasonal_pdq = None
tmp_model = None
best_mdl = None

results = []

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            tmp_mdl = SARIMAX(series['Fechamento'],
                              order = param,
                              seasonal_order = param_seasonal,
                              enforce_stationarity=True,
                              enforce_invertibility=True)
            res = tmp_mdl.fit()
            if res.aic < best_aic:
                best_aic = res.aic
                best_pdq = param
                best_seasonal_pdq = param_seasonal
                best_mdl = tmp_mdl
            res = tmp_mdl.fit()
            results.append([param, param_seasonal, res.aic, res.bic])
        except:
            continue


results_df = pd.DataFrame(results, columns=['pdq', 'seasonal_pdq', 'AIC', 'BIC'])

# Ordenando pelo AIC
results_df = results_df.sort_values(by='AIC', ascending=True)

print(results_df)

results_df.to_csv("./src/data/modelo_sarima.csv")