import streamlit as st
from PIL import Image

st.set_page_config(page_title="Analise", page_icon=":house:")

image = Image.open("./src/img/ibovespa.jpg")
st.image(image)




tab1, tab2 = st.tabs(["ML/Normalizados","Prinpais Modelo ST"])


with tab1:
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

    st.title("Machine Learning e Normalização")

    st.write("""
    ## Machine Learning: Nosso Poderoso Aliado
    O aprendizado de máquina é uma ferramenta poderosa que nos permite fazer previsões a partir de dados. Neste estudo, vamos usar o aprendizado de máquina para prever o fechamento do Ibovespa. Para isso, vamos utilizar a biblioteca Scikit-Learn, uma das bibliotecas mais populares e completas para aprendizado de máquina em Python.

    O aprendizado de máquina funciona encontrando padrões nos dados. No nosso caso, vamos treinar nossos modelos de aprendizado de máquina nos dados históricos do Ibovespa para que eles possam aprender os padrões de subida e descida do índice.

    ## Normalização: Preparando os Dados
    Antes de treinarmos nossos modelos, precisamos preparar nossos dados. Uma etapa importante desse processo é a normalização.

    A normalização é uma técnica que transforma todos os valores numéricos dos dados para uma escala comum, sem distorcer as diferenças nos intervalos de valores ou perder informações. Isso é especialmente importante para alguns algoritmos de aprendizado de máquina que não funcionam bem quando os atributos numéricos têm escalas diferentes.

    No caso do nosso conjunto de dados do Ibovespa, a decisão de normalizar os dados dependerá do algoritmo de aprendizado de máquina que planejamos usar. Alguns algoritmos, como ARIMA ou LSTM, que são comumente usados para previsão de séries temporais, podem se beneficiar da normalização, pois esses modelos podem ser sensíveis à escala dos dados. A normalização pode ajudar a estabilizar a variação ao longo do tempo e a reduzir o efeito de outliers. No entanto, é sempre uma boa ideia experimentar com e sem normalização para ver qual abordagem produz os melhores resultados para o nosso conjunto de dados específico.
    """)

with tab2:
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

    st.title("Modelos de Aprendizado de Máquina para Previsão do Ibovespa")

    st.write("""
    ## Modelos de Aprendizado de Máquina
    Existem muitos modelos de aprendizado de máquina que podemos usar para prever o fechamento do Ibovespa. Aqui estão os principais modelos, listados em ordem de relevância:

    1. **ARIMA (AutoRegressive Integrated Moving Average)**: ARIMA é um modelo estatístico que usa diferenças e defasagens de valores em dados de séries temporais para encontrar padrões específicos. ARIMA é adequado para séries temporais univariadas que demonstram padrões de tendência e sazonalidade. Ele não requer normalização dos dados e é uma escolha comum para muitos problemas de previsão de séries temporais.

    2. **SARIMA (Seasonal AutoRegressive Integrated Moving Average)**: SARIMA é uma extensão do ARIMA que é capaz de modelar sazonalidade. Assim como o ARIMA, o SARIMA não requer normalização dos dados. É uma boa escolha para séries temporais que demonstram sazonalidade, além de tendências.

    3. **LSTM (Long Short-Term Memory)**: LSTM é um tipo de rede neural recorrente que é capaz de aprender e lembrar de longas sequências de informações, o que é útil para tarefas de previsão de séries temporais. As redes LSTM requerem que os dados sejam normalizados, geralmente para uma escala entre 0-1. Elas são uma boa escolha para séries temporais complexas que demonstram padrões não lineares.

    4. **Prophet**: Prophet é um modelo de previsão de séries temporais desenvolvido pelo Facebook. Ele é projetado para lidar com séries temporais que têm tendências fortes e sazonalidade, e também leva em conta feriados e eventos especiais. O Prophet não requer normalização dos dados.

    5. **Modelos de Regressão com Variáveis ​​de Tempo Retardado**: Estes são modelos de regressão que usam valores passados da série temporal como variáveis preditoras. Isso é conhecido como "lagged variables". Esses modelos podem se beneficiar da normalização se os dados estiverem em escalas diferentes.

    Vamos testar esses modelos para prever o fechamento do Ibovespa. Você pode ver os resultados desses testes na página 'Testes' no menu à esquerda.
    """)