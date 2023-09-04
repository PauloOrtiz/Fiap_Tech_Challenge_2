import streamlit as st
from PIL import Image
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller

st.set_page_config(page_title="Premissas da Modelagem", page_icon=":house:")

image = Image.open("./src/img/Model.jpg")
st.image(image)



ibovespa = pd.read_csv('./src/data/ibovespa2021.csv', sep=',')
ibovespa['Data'] = pd.to_datetime(ibovespa['Data'],format='%Y-%m-%d')
ibovespa['Fechamento'] = pd.to_numeric(ibovespa['Fechamento'], errors='coerce')




st.markdown("""
    <style>
    body {
        color: #ffffff;
        background-color: #4B8BBE;
    }
    h1 {
        color: #CD8D00;
        text-align: center;
    }
    h2 {
        color: #306998;
    }
    h3 {
        color: #E3A15D;
    }
    p{
        text-indent: 40px;
    }
    </style>
    """, unsafe_allow_html=True)



    # Título da seção
   
st.markdown("""
# Estacionariedade e a Importância do Teste de Dickey-Fuller       
            
                
## Teste de Dickey-Fuller
            
Ao trabalhar com séries temporais, um dos conceitos mais cruciais é a estacionariedade. Uma série temporal é dita estacionária se suas propriedades estatísticas, como média, variância e autocorrelação, são constantes ao longo do tempo. A maioria dos modelos de séries temporais, como ARIMA, assume que a série é estacionária. Uma série não estacionária pode ser problemática porque pode ser difícil modelar e prever.

O Teste de Dickey-Fuller Aumentado (ADF) é um dos métodos mais comuns para verificar a estacionariedade de uma série temporal. A hipótese nula do teste é que a série temporal é não estacionária. Se o valor-p do teste for menor ou igual a um nível de significância (geralmente 0,05), rejeitamos a hipótese nula e concluímos que a série é estacionária.

Vamos agora realizar o Teste de Dickey-Fuller em nossa série temporal do Ibovespa e interpretar os resultados:
""",unsafe_allow_html=True)

# Função para realizar o teste
def perform_adf_test(series):
    result = adfuller(series)
    return result

# Executando o teste
ibovespa_series = ibovespa['Fechamento']
result = perform_adf_test(ibovespa_series)

# Mostrando os resultados no Streamlit
st.markdown("**Resultado do Teste Dickey-Fuller:**")
st.markdown(f'**Estatística de teste:** {result[0]}')
st.markdown(f'**Valor-p:** {result[1]}')
st.markdown(f'**Número de defasagens usadas:** {result[2]}')
st.markdown(f'**Número de observações:** {result[3]}')
st.markdown("**Valores críticos:**")
for key, value in result[4].items():
    st.markdown(f'   **{key}:** {value}')

if result[1] <= 0.05:
    st.markdown("**Conclusão:** A Série é estacionária (rejeita-se a hipótese nula)")
else:
    st.markdown("**Conclusão:** A Série não é estacionária (não rejeita-se a hipótese nula)")

st.markdown("""
### Com base no Teste Dickey-Fuller, podemos observar o seguinte:

- A **Estatística de teste** é de aproximadamente -2.39, que é maior do que todos os valores críticos (1%, 5% e 10%). 
- O **Valor-p** é de 0.142, que é significativamente maior que 0.05. 

Em testes estatísticos, um valor-p menor que 0.05 é frequentemente usado para rejeitar a hipótese nula. No contexto do Teste Dickey-Fuller, a hipótese nula é de que a série temporal possui uma raiz unitária e, portanto, não é estacionária.

Como o valor-p é maior que 0.05 e a estatística de teste é maior que os valores críticos, **não rejeitamos a hipótese nula**. Isso sugere que a série temporal não é estacionária.

Isso é importante porque muitos modelos de séries temporais, como ARIMA, assumem que a série é estacionária. Se a série não for estacionária, pode ser necessário aplicar transformações, como diferenciação, para torná-la estacionária antes de modelar.
""")

st.markdown("""
## Transformando a Série em Estacionária através da Diferenciação

Ao aplicar a diferenciação, estamos basicamente subtraindo a observação atual da observação anterior. Isso pode ajudar a remover tendências e padrões sazonais, tornando a série mais estacionária.

Vamos visualizar a série após aplicar uma primeira diferenciação para entender melhor como ela se transformou.
""")

ibovespa_diff1 = ibovespa.diff().dropna()    
# Gráfico usando Plotly
if 'Data' not in ibovespa_diff1.columns:
    st.write("A coluna 'Data' não existe em ibovespa_diff1!")
# Definindo 'Data' como índice para ibovespa antes da diferenciação
ibovespa.set_index('Data', inplace=True)
ibovespa_diff1 = ibovespa.diff().dropna()

# Verificando se a coluna 'Data' é do tipo datetime
if not pd.api.types.is_datetime64_any_dtype(ibovespa_diff1.index):
    st.write("O índice de ibovespa_diff1 não é do tipo datetime!")
# Se não for datetime, você pode converter o índice para datetime aqui, se necessário

# Plotando o gráfico
fig = go.Figure()
fig.add_trace(go.Scatter(x=ibovespa_diff1.index, y=ibovespa_diff1['Fechamento'], mode='lines', name='1º Diferenciação'))

# Configurando o layout do gráfico
unique_years = ibovespa_diff1.index.year.unique()
fig.update_layout(
    title={
    'text': "Série com 1º Diferenciação",
    'y':0.95,
    'x':0.5,
    'xanchor': 'center',
    'yanchor': 'top',
    'font': {
        'size': 20,
        'color': '#306998'
    }},
    xaxis_title='Anos',
    yaxis_title="Valor",
    xaxis=dict(
        tickvals=pd.to_datetime([f'{year}-01-01' for year in unique_years]),  # Escolhe um ponto para cada ano único
        ticktext=unique_years,  # Mostra apenas o ano
        tickangle=-45,  # Inclina os rótulos para melhor visualização
        title_font=dict(size=18, color='#CD8D00'),
        tickfont=dict(size=14, color='#333')
    ),
    template="plotly_dark",
    yaxis=dict(
        title_font=dict(size=18, color='#CD8D00'),
        tickfont=dict(size=14, color='#333')
    )
    xaxis_title='Anos',
    yaxis_title="Pontuação",
)

st.plotly_chart(fig)

st.markdown("""
## Por que realizar o Teste Dickey-Fuller após a diferenciação?

No entanto, após a diferenciação, é crucial retestar a série para confirmar se ela se tornou estacionária. O Teste Dickey-Fuller é uma ferramenta poderosa para essa verificação.
""")

# Função para realizar o teste
def perform_adf_test(series):
    result = adfuller(series)
    st.markdown("**Resultado do Teste Dickey-Fuller:**")
    st.markdown(f'**Estatística de teste:** {result[0]}')
    st.markdown(f'**Valor-p:** {result[1]}')
    st.markdown(f'**Número de defasagens usadas:** {result[2]}')
    st.markdown(f'**Número de observações:** {result[3]}')
    st.markdown("**Valores críticos:**")
    for key, value in result[4].items():
        st.markdown(f'   **{key}:** {value}')

    if result[1] <= 0.05:
        st.markdown("**Conclusão:** A Série é estacionária (rejeita-se a hipótese nula)")
    else:
        st.markdown("**Conclusão:** A Série não é estacionária (não rejeita-se a hipótese nula)")

# Chamando a função
perform_adf_test(ibovespa_diff1['Fechamento'])

# Conclusão
st.markdown("""
### Conclusão:

Após a diferenciação, a série tornou-se estacionária, conforme confirmado pelo Teste Dickey-Fuller. Como o valor-p tende a 0.0, rejeitamos a hipótese nula, indicando que a série é estacionária. Isso é um bom sinal, pois agora podemos prosseguir com a modelagem ARIMA, sabendo que a suposição de estacionariedade foi atendida.
""")

st.write("""
## Em Direção à Modelagem
Agora que temos uma visão clara da trajetória do Ibovespa ao longo dos anos, é hora de mergulhar mais fundo. Vamos usar técnicas avançadas de Machine Learning para prever o futuro deste índice icônico. Junte-se a nós nesta emocionante jornada de descoberta!
""",unsafe_allow_html=True )
