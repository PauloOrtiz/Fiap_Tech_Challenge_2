import streamlit as st
from PIL import Image
import pandas as pd
import statsmodels.api as sm
import numpy as np
from statsmodels.tsa.arima.model import ARIMA       
import plotly.graph_objects as go



st.set_page_config(page_title="Arima", page_icon=":house:")

image = Image.open("./src/img/Arima.png")
st.image(image)

ibovespa = pd.read_csv('./src/data/ibovespa.csv', sep=',')
ibovespa['Data'] = pd.to_datetime(ibovespa['Data'],format='%Y-%m-%d')
ibovespa.set_index('Data', inplace=True) 
ibovespa['Fechamento'] = pd.to_numeric(ibovespa['Fechamento'], errors='coerce')


tab1, tab2, tab3 = st.tabs(["ARIMA", "Treino e Teste", "Modelos"])

with tab1:
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

    st.title("O Modelo ARIMA")

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
        
        st.markdown("""
            1. **$Y_t$**: Valor atual da série temporal.
            2. **\( c \)**: Constante representando o valor médio da série sem efeitos AR e MA.
            3. **Termos AR**: Relacionam a observação atual com suas anteriores. O número de termos é determinado por \( p \).
            4. **Termos MA**: Relacionam o erro da observação atual com erros passados. O número de termos é determinado por \( q \).
            5. **\( e_t \)**: Erro no tempo \( t \), diferença entre valor observado e previsto.

            Em essência, ARIMA usa termos autoregressivos e de média móvel para prever a série temporal. A quantidade de termos é definida pelos valores de \( p \) e \( q \).
            """, unsafe_allow_html=True)

    st.write("""
    ## Usando ARIMA em Python
    Em Python, podemos usar a biblioteca `statsmodels` para trabalhar com o modelo ARIMA. A função `ARIMA` desta biblioteca nos permite ajustar um modelo ARIMA aos nossos dados e usar esse modelo para fazer previsões.

    Agora, vamos testar o modelo ARIMA em nossos dados do Ibovespa!
    """)
with tab2:


    ibovespa_mensal = ibovespa['Fechamento'].resample("30D").mean()
    ibovespa_mensal = ibovespa_mensal.to_frame(name="Fechamento")
    model = ARIMA(ibovespa_mensal, order=(1, 2, 1))
    res = model.fit()
    st.write(res.summary())
    # Carregando seus dados
    

with tab3:
    st.markdown(
        """
        # Escolhendo Modelos

        ## Modelo Arima 2 2 0

        # Previsões e Erros

        ## Tabelas 

        ## Acuracia 
        """
    )


    

