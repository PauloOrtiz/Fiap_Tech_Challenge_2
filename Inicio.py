import streamlit as st
from PIL import Image

st.set_page_config(page_title="Main Page", page_icon=":house:")

image = Image.open("./src/img/b3.jpg")
st.image(image)





st.markdown("""
<style>
body {
    color: #ffffff;
    background-color: #4B8BBE;
}
h1 {
    color: #CD8D00;
    text-aling: center;
}
h2 {
    color: #306998;
}
</style>
""", unsafe_allow_html=True)



st.write("""
# Previsão do Fechamento do Ibovespa
## Bem-vindo à nossa aventura no mundo financeiro!
Você já se perguntou como seria se pudéssemos prever o futuro do mercado de ações? Embora não possamos garantir previsões perfeitas, podemos usar a ciência de dados e o aprendizado de máquina para tentar prever tendências e movimentos futuros.

## O Ibovespa
Nesta jornada, nosso protagonista é o Ibovespa, o principal indicador de desempenho das ações negociadas na B3, a bolsa de valores brasileira. Ele é como um termômetro do mercado de ações brasileiro, refletindo o desempenho médio das ações mais negociadas e mais representativas do mercado. Quando o Ibovespa sobe, isso geralmente significa que a maioria das ações está subindo. Quando cai, a maioria das ações está caindo.

## A missão
Nossa missão é realizar uma análise preditiva do fechamento do Ibovespa. Vamos explorar dados históricos, mergulhar profundamente na análise desses dados e treinar modelos de aprendizado de máquina para prever o fechamento do Ibovespa. Nosso objetivo é alcançar uma precisão de 70% ou superior em nossas previsões. Esta é uma tarefa desafiadora, mas estamos prontos para enfrentá-la!

## Os dados 
Nossa aventura começa com a coleta de dados. Conseguimos obter dados históricos do Ibovespa atraves da bilbioteca yfinance. Esses dados são como um diário do Ibovespa, registrando seus altos e baixos ao longo de um período significativo de 02 de janeiro de 2001 a 1º de agosto de 2023. Esses dados serão a base para nossas previsões. Imagine-os como o mapa do tesouro que nos guiará em nossa jornada.

## Navegação
Esta jornada é dividida em várias etapas, cada uma delas explorando um aspecto diferente do nosso estudo. Use o menu à esquerda para navegar pelas diferentes seções. Cada passo da jornada é crucial para alcançar nosso objetivo final: prever o fechamento do Ibovespa.

Agora, vamos embarcar nesta emocionante jornada de aprendizado de máquina juntos!
""", unsafe_allow_html=True)