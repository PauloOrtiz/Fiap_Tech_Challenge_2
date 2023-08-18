import streamlit as st
from PIL import Image

st.set_page_config(page_title="Main Page", page_icon=":house:")

image = Image.open("./src/img/ibovespa.jpg")
st.image(image)





st.markdown("""
<style>
body {
    color: #ffffff;
    background-color: #4B8BBE;
}
h1 {
    color: #CD8D00;
}
h2 {
    color: #306998;
}
</style>
""", unsafe_allow_html=True)

st.title("Previsão do Fechamento do Ibovespa")

st.write("""
## Bem-vindo à nossa aventura no mundo financeiro!
<p style="text-indent: 40px;"> Você já se perguntou como seria se pudéssemos prever o futuro do mercado de ações? Embora não possamos garantir previsões perfeitas, podemos usar a ciência de dados e o aprendizado de máquina para tentar prever tendências e movimentos futuros.<p> 

## O Ibovespa
<p style="text-indent: 40px;">Nesta jornada, nosso protagonista é o Ibovespa, o principal indicador de desempenho das ações negociadas na B3, a bolsa de valores brasileira. Ele é como um termômetro do mercado de ações brasileiro, refletindo o desempenho médio das ações mais negociadas e mais representativas do mercado. Quando o Ibovespa sobe, isso geralmente significa que a maioria das ações está subindo. Quando cai, a maioria das ações está caindo.<p> 

## A missão
<p style="text-indent: 40px;">Nossa missão é realizar uma análise preditiva do fechamento do Ibovespa. Vamos explorar dados históricos, mergulhar profundamente na análise desses dados e treinar modelos de aprendizado de máquina para prever o fechamento do Ibovespa. Nosso objetivo é alcançar uma precisão de 70% ou superior em nossas previsões. Esta é uma tarefa desafiadora, mas estamos prontos para enfrentá-la!<p>

## Os dados 
<p style="text-indent: 40px;">Nossa aventura começa com a coleta de dados. Conseguimos obter dados históricos do Ibovespa do site Investing. Esses dados são como um diário do Ibovespa, registrando seus altos e baixos ao longo de um período significativo de 27 de dezembro de 2000 a 1º de agosto de 2023. Esses dados serão a base para nossas previsões. Imagine-os como o mapa do tesouro que nos guiará em nossa jornada.<p>

## Navegação
<p style="text-indent: 40px;">Esta jornada é dividida em várias etapas, cada uma delas explorando um aspecto diferente do nosso estudo. Use o menu à esquerda para navegar pelas diferentes seções. Cada passo da jornada é crucial para alcançar nosso objetivo final: prever o fechamento do Ibovespa.<p>

<p style="text-indent: 40px;">Agora, vamos embarcar nesta emocionante jornada de aprendizado de máquina juntos!<p>
""", unsafe_allow_html=True)