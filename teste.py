import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go


ibovespa = pd.read_csv('./src/data/ibovespa.csv', sep=',')
ibovespa['Data'] = pd.to_datetime(ibovespa['Data'],format='%Y-%m-%d')
ibovespa['Fechamento'] = pd.to_numeric(ibovespa['Fechamento'], errors='coerce')
ibovespa['Data'] = pd.to_datetime(ibovespa['Data'])
ibovespa.set_index('Data', inplace=True)



train_size = int(len(ibovespa) * 0.8)
train, test = ibovespa[0:train_size], ibovespa[train_size:]

model = ARIMA(train, order=(2, 2, 1))
model_fit = model.fit()
forecast = model_fit.forecast(steps=len(test))

print(model_fit)
print(forecast)
print(train)


fig = go.Figure()


fig.add_trace(go.Scatter(x=ibovespa.index, y=ibovespa['Fechamento'], mode='lines', name='Real'))


fig.add_trace(go.Scatter(x=test.index, y=forecast, mode='lines', name='Previsão'))

fig.show()


model = ARIMA(train, order=(2, 2, 2))
model_fit = model.fit()
forecast = model_fit.forecast(steps=len(test))

fig = go.Figure()


fig.add_trace(go.Scatter(x=ibovespa.index, y=ibovespa['Fechamento'], mode='lines', name='Real'))


fig.add_trace(go.Scatter(x=test.index, y=forecast, mode='lines', name='Previsão'))

fig.show()

model = ARIMA(train, order=(2, 0, 2))
model_fit = model.fit()
forecast = model_fit.forecast(steps=len(test))

fig = go.Figure()


fig.add_trace(go.Scatter(x=ibovespa.index, y=ibovespa['Fechamento'], mode='lines', name='Real'))


fig.add_trace(go.Scatter(x=test.index, y=forecast, mode='lines', name='Previsão'))

fig.show()