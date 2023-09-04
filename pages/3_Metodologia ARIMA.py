import streamlit as st
from PIL import Image
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf
import numpy as np
from statsmodels.tsa.arima.model import ARIMA 
import statsmodels.api as sm      
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt



st.set_page_config(page_title="Arima", page_icon=":house:")

tabs_font_css = """
<style>
.st-b3{
   display: flow;
}
</style>
"""
st.write(tabs_font_css, unsafe_allow_html=True)

image = Image.open("./src/img/Arima.png")
st.image(image)

ibovespa = pd.read_csv('./src/data/ibovespa2021.csv', sep=',')
ibovespa['Data'] = pd.to_datetime(ibovespa['Data'],format='%Y-%m-%d')
ibovespa.set_index('Data', inplace=True) 
ibovespa['Fechamento'] = pd.to_numeric(ibovespa['Fechamento'], errors='coerce')
AIC_BIC = pd.read_csv('./src/data/df_AIC_BIC.csv', sep=';', index_col=None)


tab1, tab2, tab3, tab4, tab5 = st.tabs(["ARIMA", "Escolha do modelo", "Modelo para previsão","Acurácia", "Diagnostico do Modelo"])

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
            2. **$c$**: Constante representando o valor médio da série sem efeitos AR e MA.
            3. **Termos $phi$**: Relacionam a observação atual com suas anteriores. O número de termos é determinado por $p$.
            4. **Termos $theta$**: Relacionam o erro da observação atual com erros passados. O número de termos é determinado por $q$.
            5. **$e_t$**: Erro no tempo $t$, diferença entre valor observado e previsto.

            Em essência, ARIMA usa termos autoregressivos e de média móvel para prever a série temporal. A quantidade de termos é definida pelos valores de \( p \) e \( q \).
            """, unsafe_allow_html=True)

    st.write("""
    ## Usando ARIMA em Python
    Em Python, podemos usar a biblioteca `statsmodels` para trabalhar com o modelo ARIMA. A função `ARIMA` desta biblioteca nos permite ajustar um modelo ARIMA aos nossos dados e usar esse modelo para fazer previsões.

    Agora, vamos testar o modelo ARIMA em nossos dados do Ibovespa!
    """)
with tab2:
    
    ibovespa_diff1 = ibovespa.diff().dropna() 

    lag_acf = acf(ibovespa_diff1['Fechamento'], nlags=40)
    lag_pacf = pacf(ibovespa_diff1['Fechamento'], nlags=40, method='ols')
    conf_int = 1.96/np.sqrt(len(ibovespa_diff1['Fechamento']))

    st.markdown("""
    # Introdução

    Ao trabalhar com modelos ARIMA, uma etapa crucial é entender a autocorrelação presente nos dados. A autocorrelação refere-se à correlação de uma série temporal com uma versão atrasada de si mesma. Para entender e visualizar essa autocorrelação, utilizamos as funções de autocorrelação (ACF) e autocorrelação parcial (PACF).

    A análise foi realizada em um conjunto de dados que abrange de janeiro de 2021 a 1 de agosto de 2023. Para tornar a série estacionária, aplicamos uma diferenciação. Vamos agora observar os gráficos resultantes da ACF e PACF após essa diferenciação.
    """,unsafe_allow_html=True)

    st.markdown("""
    ## Função de Autocorrelação (ACF)

    A Função de Autocorrelação, ou ACF, nos dá valores de autocorrelação de uma série temporal com seus lags. Em outras palavras, ela nos mostra como uma observação está correlacionada com suas observações anteriores.
    """,unsafe_allow_html=True)

    fig_acf = go.Figure()
    fig_acf.add_trace(go.Scatter(y=lag_acf, mode='lines+markers'))
    fig_acf.add_shape(type="line", x0=0, x1=40, y0=conf_int, y1=conf_int, line=dict(color="red", width=0.5))
    fig_acf.add_shape(type="line", x0=0, x1=40, y0=-conf_int, y1=-conf_int, line=dict(color="red", width=0.5))
    fig_acf.update_layout(
        xaxis_title='Lag',
        yaxis_title="Autocorrelação",
        xaxis=dict(
            title_font=dict(size=18, color='#CD8D00'),
            tickfont=dict(size=14, color='#333')
        ),
        yaxis=dict(
            title_font=dict(size=18, color='#CD8D00'),
            tickfont=dict(size=14, color='#333')
        ),
        title={
            'text': 'ACF (Autocorrelation Function)',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {
                'size': 20,
                'color': '#306998'
            }
        },
    )
    st.plotly_chart(fig_acf)

    st.markdown("""
    ## Função de Autocorrelação Parcial (PACF)

    A Função de Autocorrelação Parcial, ou PACF, nos dá a correlação parcial de uma série temporal com seus lags. Ela nos mostra a correlação entre observações que estão separadas por um número específico de períodos, desconsiderando as correlações de períodos intermediários.
    """, unsafe_allow_html=True)

    fig_pacf = go.Figure()
    fig_pacf.add_trace(go.Scatter(x=list(range(len(lag_pacf))), y=lag_pacf, mode='lines+markers', name='PACF'))
    fig_pacf.add_shape(type="line", x0=0, x1=40, y0=conf_int, y1=conf_int, line=dict(color="red", width=0.5))
    fig_pacf.add_shape(type="line", x0=0, x1=40, y0=-conf_int, y1=-conf_int, line=dict(color="red", width=0.5))
    fig_pacf.update_layout(
        xaxis_title='Lag',
        yaxis_title="Autocorrelação",
        xaxis=dict(
            title_font=dict(size=18, color='#CD8D00'),
            tickfont=dict(size=14, color='#333')
        ),
        yaxis=dict(
            title_font=dict(size=18, color='#CD8D00'),
            tickfont=dict(size=14, color='#333')
        ),
        title={
            'text': 'PACF (Partial Autocorrelation Function)',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {
                'size': 20,
                'color': '#306998'
            }
        },
    )

    

    st.plotly_chart(fig_pacf)

    st.markdown("""
    ### Conclusão

    Ao observar os gráficos de ACF e PACF, notamos que ambos representam consistentemente o intervalo de confiança em todos os lags. Isso sugere que, após a diferenciação, a série temporal não apresenta autocorrelações significativas em seus lags. Esse é um bom indicativo de que a diferenciação foi eficaz em remover padrões e tendências da série, tornando-a estacionária e, assim, mais adequada para modelagem ARIMA.
    """,unsafe_allow_html=True)

    # AIC
    st.markdown("""
                
    ## Estatísticas para a escolha do modelo: 
                            
    ### Critério de Informação de Akaike (AIC)

    O critério de informação de Akaike (AIC) é uma métrica estatística que avalia a qualidade de modelos estatísticos com base na quantidade de informação que eles capturam dos dados. Foi desenvolvido por Hirotugu Akaike e é amplamente utilizado para seleção de modelos, como em contextos de regressão.

    A fórmula para calcular o AIC é:

    **[ AIC = 2k - 2$ln(L)$]**

    Onde:

    - $k$ é o número de parâmetros no modelo.
    - $\ln(L)$ é o logaritmo natural da função de verossimilhança do modelo, indicando quão bem o modelo se ajusta aos dados.

    O objetivo do AIC é equilibrar a capacidade do modelo de se ajustar aos dados (quantificada pela verossimilhança) e a penalização pela complexidade do modelo (quantificada pelo número de parâmetros). Modelos com valores menores de AIC são preferidos, pois sugerem um bom equilíbrio entre ajuste e parcimônia.
    """)

    # BIC
    st.markdown("""
    ### Critério de Informação Bayesiano (BIC)

    O Critério de Informação Bayesiano (BIC) é outra métrica estatística que avalia modelos estatísticos, considerando tanto a adequação aos dados quanto a complexidade do modelo. É uma alternativa ao AIC e é frequentemente usado em contextos similares.

    A fórmula para calcular o BIC é:

    **[ BIC = -2$ln(L)$ + k * $ln(n)$]**

    Onde:

    - $\ln(L)$ é o logaritmo natural da função de verossimilhança do modelo.
    - $k$ é o número de parâmetros no modelo.
    - $n$ é o tamanho da amostra, ou seja, o número de observações nos dados.

    O BIC penaliza a complexidade do modelo de forma mais rigorosa que o AIC. Isso significa que o BIC tende a favorecer modelos mais simples. Assim, ao comparar vários modelos, o modelo com o menor valor de BIC é geralmente considerado o melhor, pois equilibra ajuste aos dados e simplicidade.
    """)

    st.markdown("""## Análise de Parâmetros ARIMA e Métricas de Qualidade""")

    table_html = AIC_BIC.to_html(index=False)

    # Exibindo o HTML no Streamlit
    st.write(table_html, unsafe_allow_html=True)

    st.write("""
    ### Conclusão:
    A tabela acima apresenta diferentes combinações de parâmetros para o modelo ARIMA e suas respectivas métricas AIC e BIC. 

    Observando os dados, a combinação com parâmetros AR(p)=1, Integração(d)=0 e MA(q)=0 apresenta o menor valor de AIC, sugerindo que esta pode ser a melhor configuração para o modelo ARIMA, considerando os dados disponíveis. No entanto, é importante também considerar outros fatores e testes ao finalizar a seleção do modelo.
    """)


with tab3:
    train_size = int(0.80 * len(ibovespa))
    train_df = ibovespa.iloc[:train_size]
    test_df = ibovespa.iloc[train_size:]

    model = ARIMA(train_df['Fechamento'], order=(0, 1, 0))
    model_fit = model.fit()

    steps = 60

    forecast = model_fit.get_forecast(steps=steps)
    mean_forecast = forecast.predicted_mean

    forecast_index = pd.date_range(start=test_df.index[-1], periods=steps + 1, freq='D')[1:]
        
    fig = go.Figure()

    
    fig.add_trace(go.Scatter(x=train_df.index, y=train_df['Fechamento'], mode='lines', name='Treino'))
    fig.add_trace(go.Scatter(x= test_df.index, y= test_df ['Fechamento'], mode='lines', name='Teste'))

    
    fig.add_trace(go.Scatter(x=forecast_index, y=mean_forecast, mode='lines', name='Previsão', line=dict(color='red')))

    
    fig.update_layout(title='Previsão do Ibovespa com Modelo ARIMA',
                    xaxis_title='Data',
                    yaxis_title='Valor de Fechamento do Ibovespa')

    
    st.plotly_chart(fig)


with tab4:
    pass

with tab5:
    model = sm.tsa.ARIMA(ibovespa['Fechamento'], order=(1, 1, 0))
    results = model.fit()

    forecast = results.get_prediction(start=-25)
    mean_forecast = forecast.predicted_mean
    confidence_intervals = forecast.conf_int()
    lower_limits = confidence_intervals.loc[:, 'lower Fechamento']
    upper_limits = confidence_intervals.loc[:, 'upper Fechamento']

    ibovespa_t = ibovespa.loc['2023-05-01':]

    fig = make_subplots(rows=1, cols=1)
    # Gráfico de linha para os valores observados
    trace_observed = go.Scatter(x=ibovespa_t.index, y=ibovespa_t['Fechamento'], mode='lines', name='Observado')
    fig.add_trace(trace_observed)

    # Gráfico de linha para a previsão
    trace_forecast = go.Scatter(x=mean_forecast.index, y=mean_forecast.values, mode='lines', name='Previsão', line=dict(color='red'))
    fig.add_trace(trace_forecast)

    # Sombreamento da faixa de confiança
    trace_confidence = go.Scatter(x=lower_limits.index, y=lower_limits, fill='tonexty', fillcolor='rgba(255, 0, 0, 0.2)',name='Lower Limit', line=dict(color='rgba(255, 0, 0, 0)'))
    fig.add_trace(trace_confidence)
    trace_confidence = go.Scatter(x=upper_limits.index, y=upper_limits, fill='tonexty', fillcolor='rgba(255, 0, 0, 0.2)',name='Upper Limit', line=dict(color='rgba(255, 0, 0, 0)'))
    fig.add_trace(trace_confidence)

    # Layout
    fig.update_layout(
    title={
    'text': "Previsão do Ibovespa com Intervalo de Confiança",
    'y':0.95,
    'x':0.5,
    'xanchor': 'center',
    'yanchor': 'top',
    'font': {
        'size': 20,
        'color': '#306998'
    }},xaxis_title='Data',
    yaxis_title="Valor de Fechamento do Ibovespa",
    xaxis=dict(
        tickangle=-45,  # Inclina os rótulos para melhor visualização
        title_font=dict(size=18, color='#CD8D00'),
        tickfont=dict(size=14, color='#333')
    ),
    template="plotly_dark",
    yaxis=dict(
        title_font=dict(size=18, color='#CD8D00'),
        tickfont=dict(size=14, color='#333')
    ))

    # Exibir gráfico Plotly
    st.plotly_chart(fig)

    model = ARIMA(ibovespa, order=(0,1,0))
    results = model.fit()
    results.plot_diagnostics(figsize=(15, 12))
    plt.savefig("diagnostics.png")
    st.image("diagnostics.png")

    
        

