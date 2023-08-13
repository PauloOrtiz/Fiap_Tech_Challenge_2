import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Analise", page_icon=":house:")

image = Image.open("./src/img/ibovespa.jpg")
st.image(image)

df_ibovespa = pd.read_csv("./src/data/BD.csv", sep=",", decimal = ".")
df_ibovespa["Último"] = df_ibovespa["Último"].round(2)
df_ibovespa["anomes"] = pd.to_datetime(df_ibovespa["Data"], format='%d.%m.%Y').dt.strftime('%Y-%m')
df_ibovespa_agg = df_ibovespa.groupby("anomes")["Último"].mean().round(2).reset_index()

tab1, tab2, tab3, tab4 = st.tabs(["EDA", "??","ML/Normalizados","Prinpais Modelo ST"])

with tab1:

    st.markdown("""
    Para melhor compreender melhor a variação do indice IBOVESPA é necessário compreender o historico e as suas principais variações.
    Por isso, fizemos uma analise entre os anos 2000 a 2023 para entender como o Ibovespa oscilou durante esse período, em que é possível ter um panorama de longo prazo do índice da Bolsa.

    A crise de 2008, conhecida como a crise do sub-prime, trouxe diversos dias de baixa à B3 graças a hipótese de um colapso hipotecário nos EUA, que arrastou várias instituições financeiras americanas para a corda bamba.

    Em 2016 a instabilidade política nas eleições presidenciais nos Estados Unidados entre o candidatp Donald Trump e Hillary Clinton e a alta variação no preço do petrolio brasileiro fez com que o mercado tivesse quedas expressivas na cotação.

    O ano de 2020 foi marcado por um grande aumento da aversão a risco nos mercados globais impulsionadas pela preocupação com relação ao surto de coronavírus na China, essa preocupação se elevou fortemente em fevereiro, à medida que a Covid-19 se espalhou por vários países.
    """)

    # Criar figura e subplots
    fig = go.Figure()

    # Adicionar traço de linha para o DataFrame
    fig.add_trace(go.Scatter(x=df_ibovespa_agg['anomes'], y=df_ibovespa_agg['Último'], mode='markers+lines', line=dict(color='blue'), marker=dict(color='blue')))

    # Adicionar traço de preenchimento abaixo da linha
    fig.add_trace(go.Scatter(x=df_ibovespa_agg['anomes'], y=df_ibovespa_agg['Último'], fill='tozeroy', fillcolor='rgba(0, 0, 255, 0.3)', mode='none'))

    # Personalizar layout
    fig.update_layout(
        title='Variação do Ibovespa ao Longo dos Anos',
        xaxis_title='Ano',
        yaxis_title='Valor',
        xaxis=dict(
            tickformat='%Y',  # Formato do eixo X para exibir apenas o ano
            showgrid=True,
            gridwidth=0.5,
            gridcolor='black',
            showline=True,
            linewidth=1,
            linecolor='black',
            zeroline=False,
            showticklabels=True,
            tickfont=dict(size=10),
            tickangle=0,
            showspikes=False,
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=0.5,
            gridcolor='black',
            showline=True,
            linewidth=1,
            linecolor='black',
            zeroline=False,
            showticklabels=True,
            tickfont=dict(size=10),
            showspikes=False,
        ),
        showlegend=False,
        plot_bgcolor='white',
        margin=dict(l=80, r=20, t=60, b=60),
    )
    # Mostrar o gráfico
    st.plotly_chart(fig)

with tab2:
    pass

with tab3:
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

    ## Análise Preditiva e Séries Temporais
    Em suma, a Análise Preditiva é uma abordagem de análise de dados que utiliza técnicas estatísticas, Aprendizado de Máquina e mineração de dados para fazer previsões sobre eventos futuros ou comportamentos com base em padrões e informações históricas.
    
    Como vimos anteriormente, o método de Séries Temporais utiliza principalmente a aprendizagem supervisionada e tem como aplicação mais comum a Análise Preditiva, visto que temos um conjunto de dados sequenciais ordenados no tempo, em dias, semanas, meses, anos etc., onde cada ponto dos dados é influenciado por observações passadas e usado para prever valores futuros que a série possivelmente irá assumir. 
    
    A nossa análise em questão traz os dados de fechamento diários do Ibovespa, conhecido como IBOV, o índice principal da Bolsa de Valores de São Paulo (B3) que representa a performance média das ações das empresas mais negociadas no mercado brasileiro. Assim, o índice serve como um termômetro para investidores, analistas e instituições financeiras que o acompanham a fim de avaliar a saúde econômica do Brasil e tomar decisões de investimento.
             

    ## Normalização: Preparando os Dados
    Antes de treinarmos nossos modelos, precisamos preparar nossos dados. Uma etapa importante desse processo é a normalização.

    A normalização é uma técnica que transforma todos os valores numéricos dos dados para uma escala comum, sem distorcer as diferenças nos intervalos de valores ou perder informações. Isso é especialmente importante para alguns algoritmos de aprendizado de máquina que não funcionam bem quando os atributos numéricos têm escalas diferentes.

    No caso do nosso conjunto de dados do Ibovespa, a decisão de normalizar os dados dependerá do algoritmo de aprendizado de máquina que planejamos usar. Alguns algoritmos, como ARIMA ou LSTM, que são comumente usados para previsão de séries temporais, podem se beneficiar da normalização, pois esses modelos podem ser sensíveis à escala dos dados. A normalização pode ajudar a estabilizar a variação ao longo do tempo e a reduzir o efeito de outliers. No entanto, é sempre uma boa ideia experimentar com e sem normalização para ver qual abordagem produz os melhores resultados para o nosso conjunto de dados específico.
    """)

with tab4:
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