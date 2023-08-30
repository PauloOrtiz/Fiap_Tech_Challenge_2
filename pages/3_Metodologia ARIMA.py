import streamlit as st
from PIL import Image
import pandas as pd


st.set_page_config(page_title="Arima", page_icon=":house:")

image = Image.open("./src/img/ibovespa.jpg")
st.image(image)





st.markdown("""
<style>
body {
    color: #ffffff;
    background-color: #4B8BBE;
}
h1 {
    color: #FFD43B;
}
h2 {
    color: #306998;
}
</style>
""", unsafe_allow_html=True)

st.title("Testando o Modelo ARIMA")

st.write("""
## ARIMA: Uma Jornada no Tempo
ARIMA, que significa AutoRegressive Integrated Moving Average, é um modelo clássico na análise e previsão de séries temporais. Ele é como um viajante do tempo, capaz de olhar para o passado para prever o futuro.

O ARIMA usa três componentes principais para modelar uma série temporal: termos autoregressivos (AR), termos de média móvel (MA) e a ordem de diferenciação (I). Juntos, esses componentes permitem ao ARIMA capturar uma variedade de diferentes estruturas de séries temporais.

## Como o ARIMA funciona?
O ARIMA funciona encontrando padrões nos dados passados e usando esses padrões para prever o futuro. Os termos autoregressivos permitem que o modelo capture a influência dos valores passados na série temporal. Os termos de média móvel permitem que o modelo capture os erros de previsão passados. A diferenciação permite que o modelo capture tendências na série temporal.

## A Fórmula do ARIMA
A fórmula do ARIMA é um pouco complexa, mas aqui está uma versão simplificada:
""")

with st.expander("Ver a fórmula do ARIMA"):
    st.latex(r"""
    Y_t = c + \phi_1 Y_{t-1} + \phi_2 Y_{t-2} + ... + \phi_p Y_{t-p} + \theta_1 e_{t-1} + \theta_2 e_{t-2} + ... + \theta_q e_{t-q} + e_t
    """)

st.write("""
## Usando ARIMA em Python
Em Python, podemos usar a biblioteca `statsmodels` para trabalhar com o modelo ARIMA. A função `ARIMA` desta biblioteca nos permite ajustar um modelo ARIMA aos nossos dados e usar esse modelo para fazer previsões.

Agora, vamos testar o modelo ARIMA em nossos dados do Ibovespa!
""")