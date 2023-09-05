from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import pandas as pd
import itertools
import streamlit as st

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


train_size = int(0.80 * len(ibovespa))
train_data = ibovespa['Fechamento'].iloc[:train_size]
test_data = ibovespa['Fechamento'].iloc[train_size:]

model = SARIMAX(ibovespa['Fechamento'], order=(0,1,0), seasonal_order=(1,0,1,12))
results = model.fit()

forecast = results.get_forecast(steps=len(test_data))
mean_forecast = forecast.predicted_mean

y_true = test_data
y_pred = mean_forecast

# Verificações
zero_values = y_true[y_true == 0]
if len(zero_values) > 0:
    st.write(f"Há {len(zero_values)} valores zero em 'y_true'")
else:
    st.write("Não há valores zero em 'y_true'")

nan_inf_values = y_pred[np.isnan(y_pred) | np.isinf(y_pred)]
if len(nan_inf_values) > 0:
    st.write(f"Há {len(nan_inf_values)} valores NaN ou infinitos em 'y_pred'")
else:
    st.write("Não há valores NaN ou infinitos em 'y_pred'")

# Cálculo das métricas
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)

# Cálculo detalhado do MAPE
errors = (y_true - y_pred) / y_true
errors = errors.replace({np.inf: np.nan, -np.inf: np.nan})  # substitua infinitos por NaN
def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

y_true_array = np.array(y_true)
y_pred_array = np.array(y_pred)

# Verifique se os shapes são iguais
if y_true_array.shape != y_pred_array.shape:
    st.write("Erro: y_true e y_pred têm shapes diferentes!")
    st.stop()

# Cálculo detalhado do MAPE
errors = y_true_array - y_pred_array
relative_errors = np.where(y_true_array != 0, errors / y_true_array, 0)  # Calcula o erro relativo apenas onde y_true não é zero

# Calcular o MAPE
mape = np.mean(np.abs(relative_errors)) * 100

st.write(f"MAE: {mae:.2f}")
st.write(f"MSE: {mse:.2f}")
st.write(f"RMSE: {rmse:.2f}")
st.write(f"MAPE: {mape:.2f}%")