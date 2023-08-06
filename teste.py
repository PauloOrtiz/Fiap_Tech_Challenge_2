import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

# Carregar os dados
data = pd.read_csv('./src/data/BD.csv', sep=',', parse_dates=['Data'], index_col='Data')

# Inverter a série temporal (se necessário)
data = data[::-1]

# Visualizar os primeiros registros
print(data.head())


# Dividir os dados em conjuntos de treinamento e teste
train_data = data[:int(len(data)*0.7)]
test_data = data[int(len(data)*0.7):]

print(f"Treinamento: {len(train_data)}")
print(f"Teste: {len(test_data)}")


# Ajustar o modelo ARIMA
model = ARIMA(train_data, order=(5,1,0))
model_fit = model.fit(disp=0)

# Fazer previsões
start_index = len(train_data)
end_index = len(train_data) + len(test_data) - 1
forecast = model_fit.predict(start=start_index, end=end_index)

# Calcular o RMSE
rmse = sqrt(mean_squared_error(test_data, forecast))
print(f"RMSE: {rmse}")

# Plotar os valores reais e as previsões
plt.plot(test_data, label='Valores Reais')
plt.plot(forecast, color='red', label='Previsões')
plt.legend()
plt.show()
