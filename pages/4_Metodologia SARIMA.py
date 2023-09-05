import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sarima", page_icon=":house:")

tabs_font_css = """
<style>
.st-b3{
   display: flow;
}
</style>
"""
st.write(tabs_font_css, unsafe_allow_html=True)

image = Image.open("./src/img/Sarima.png")
st.image(image)

ibovespa = pd.read_csv('./src/data/ibovespa2021.csv', sep=',')
ibovespa['Data'] = pd.to_datetime(ibovespa['Data'],format='%Y-%m-%d')
ibovespa.set_index('Data', inplace=True) 
ibovespa['Fechamento'] = pd.to_numeric(ibovespa['Fechamento'], errors='coerce')


tab1, tab2, tab3, tab4, tab6 = st.tabs(["SARIMA", "Escolha do modelo", "Modelo para previsão","Acurácia", "Diagnóstico do Modelo"])

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

with tab1:
    st.markdown("""
    # SARIMA: Uma Jornada Além do ARIMA

    Quando nos aventuramos no mundo das séries temporais, rapidamente nos deparamos com o modelo ARIMA, uma ferramenta poderosa e flexível. No entanto, o universo da previsão é vasto e, às vezes, precisamos de algo mais robusto, algo que considere sazonalidades e variáveis exógenas. É aqui que o SARIMA entra em cena.

    ## O que é SARIMA?

    O SARIMA é uma extensão do ARIMA. O 'S' refere-se à sazonalidade. Em outras palavras, enquanto o ARIMA é excelente para prever séries temporais baseadas em seus próprios valores passados, o SARIMA leva em consideração padrões sazonais e outras variáveis externas que podem influenciar a série.

    ## A Fórmula Mágica

    A beleza do SARIMA reside em sua capacidade de combinar diferentes componentes:

    - **AR (AutoRegressivo)**: Baseia-se na relação entre uma observação e um número de observações defasadas.
    - **I (Integrado)**: Usa a diferenciação das observações para tornar a série temporal estacionária.
    - **MA (Média Móvel)**: Baseia-se na relação entre uma observação e um erro residual proveniente de uma média móvel aplicada a observações defasadas.
    - **Sazonalidade**: Captura padrões que se repetem em intervalos fixos.
    

    ## Por que SARIMA para o Ibovespa?

    O Ibovespa, como principal indicador do mercado de ações brasileiro, é influenciado por uma miríade de fatores. Além de suas próprias flutuações históricas, ele é afetado por eventos econômicos, políticos e globais. A sazonalidade, como os padrões de fim de ano ou eventos recorrentes, também pode desempenhar um papel. O SARIMA, com sua capacidade de considerar todas essas nuances, é uma escolha natural para uma análise aprofundada do Ibovespa.

    ## Embarque Nesta Jornada

    Convido você a se juntar a nós nesta exploração detalhada do Ibovespa através das lentes do SARIMA. Vamos descobrir os segredos escondidos nos dados, desvendar padrões e, esperançosamente, lançar luz sobre o futuro deste índice crucial.
    """, unsafe_allow_html=True)
    
with tab2:
    st.markdown("""
    # A Arte da Escolha: Determinando o Melhor Modelo SARIMA

    A previsão de séries temporais é tanto uma arte quanto uma ciência. Enquanto a ciência nos fornece as ferramentas e técnicas, a arte está em escolher o modelo certo. No universo SARIMA, essa escolha é crucial e pode ser um pouco complexa.

    ## O Desafio da Escolha

    O SARIMA não é apenas um modelo, mas uma família de modelos. Cada combinação de parâmetros AR (p), I (d), MA (q), sazonal (P, D, Q) e o período de sazonalidade (s) pode resultar em um modelo diferente. A questão é: qual combinação é a melhor para nossos dados?

    ## Testando e Validando

    A resposta está em testar e validar. Aqui estão os passos que podemos seguir:

    - **Estacionariedade**: Certifique-se de que a série temporal é estacionária. Se não for, diferencie-a até que seja.
    - **ACF e PACF**: Use as funções de autocorrelação (ACF) e autocorrelação parcial (PACF) para ter uma ideia inicial dos parâmetros.
    - **Grid Search**: Teste diferentes combinações de parâmetros e escolha a que oferece o melhor desempenho, geralmente medido pelo AIC (Critério de Informação de Akaike) ou BIC (Critério de Informação Bayesiano).
    
    
    ## Vamos Começar!

    A seguir, vamos mergulhar nos dados e começar nossa busca pelo modelo SARIMA ideal para o Ibovespa. Acompanhe cada etapa, observe as métricas e junte-se a nós nesta jornada empolgante de descoberta.
   
    """)
    
    
  
    st.markdown("""                
        ## Ajustando o Modelo SARIMA
        ### Análisando os Critérios de informação
        
        
        Abaixo, apresentamos os resultados de diferentes combinações de parâmetros sazonais para o modelo SARIMA, mantendo a parte ARIMA fixa em (0,1,0). 
        Os critérios de informação AIC e BIC são usados para avaliar a qualidade de cada modelo. Em geral, modelos com valores AIC e BIC mais baixos são preferíveis.
    """)

    
    sarima = pd.read_csv('./src/data/modelo_sarima.csv', sep=',', index_col=None)
    table_html = sarima.to_html(index=False)

    # Exibindo o HTML no Streamlit
    st.write(table_html, unsafe_allow_html=True)
    
    st.markdown("""
    ### Observações:
    - O modelo com os parâmetros sazonais `(0, 0, 0, 12)` apresenta o menor valor de AIC, sugerindo que pode ser o modelo mais adequado dentre os testados.
    - É importante lembrar que, além dos critérios de informação, devemos considerar outros fatores, como a interpretabilidade do modelo e a validação em um conjunto de testes, ao escolher o modelo final.
    """)
    
    
with tab3: 
    st.title("Análise do Modelo SARIMA Escolhido")
    st.markdown("""
        ## Introdução
        A escolha de um modelo de séries temporais é uma tarefa que combina ciência e arte. Embora critérios como o AIC nos forneçam uma métrica objetiva para avaliar diferentes modelos, a decisão final muitas vezes leva em consideração outros fatores, como a capacidade do modelo de capturar padrões sazonais, tendências e outros componentes da série.
        
        Nesta seção, vamos explorar em detalhes o modelo SARIMA escolhido para nossa série temporal: SARIMA(0, 1, 0)x(1, 0, 1, 12).
    """)
    st.markdown("""
    ## Sumário Estatístico do Modelo
    Abaixo, apresentamos um resumo das estatísticas do modelo escolhido. Este sumário nos fornece informações valiosas sobre os parâmetros do modelo, sua significância estatística e outras métricas que nos ajudam a entender o desempenho do modelo.
    """)       
    
    model1 = SARIMAX(ibovespa['Fechamento'],order=(0,1,0), seasonal_order=(1, 0, 1, 12)).fit(ds=-1)
    st.write(model1.summary())
    
    
    st.markdown("""
    ## Conclusão
    A análise do sumário estatístico revela que o modelo SARIMA escolhido se ajusta bem aos dados. Os parâmetros do modelo são estatisticamente significativos, o que indica que eles contribuem de forma valiosa para as previsões.
    
    Além disso, métricas como o Ljung-Box e o Jarque-Bera nos fornecem confiança na adequação do modelo e na normalidade dos resíduos, respectivamente.
    
    Neste caso, o valor do Ljung-Box é de 1.53, indicando que não há autocorrelação significativa nos resíduos, porém, o teste Jarque-Bera mostrou que os resíduos não seguem uma distribuição normal. Apesar disso, seguimos para analisar a acurácia e diagnóstico detalhado do modelo.
    
    Em resumo, este modelo representa uma ferramenta robusta e confiável para prever a série temporal em questão, e estamos confiantes em suas previsões para o futuro.
    """)    
    
with tab4: 
    
    train_size = int(0.80 * len(ibovespa))
    train_data = ibovespa['Fechamento'].iloc[:train_size]
    test_data = ibovespa['Fechamento'].iloc[train_size:]

    model = SARIMAX(ibovespa['Fechamento'], order=(0,1,0), seasonal_order=(1,0,1,12))
    results = model.fit()

    forecast = results.get_forecast(steps=len(test_data))
    mean_forecast = forecast.predicted_mean

    y_true = test_data
    y_pred = mean_forecast

    # Verificações
    zero_values = y_true[y_true == 0]
    if len(zero_values) > 0:
        st.write(f"Há {len(zero_values)} valores zero em 'y_true'")
    else:
        st.write("Não há valores zero em 'y_true'")

    nan_inf_values = y_pred[np.isnan(y_pred) | np.isinf(y_pred)]
    if len(nan_inf_values) > 0:
        st.write(f"Há {len(nan_inf_values)} valores NaN ou infinitos em 'y_pred'")
    else:
        st.write("Não há valores NaN ou infinitos em 'y_pred'")

    # Cálculo das métricas
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    # Cálculo detalhado do MAPE
    errors = (y_true - y_pred) / y_true
    errors = errors.replace({np.inf: np.nan, -np.inf: np.nan})  # substitua infinitos por NaN
    def calculate_mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    errors = y_true - y_pred
    relative_errors = np.where(y_true != 0, errors / y_true, 0)  # Calcula o erro relativo apenas onde y_true não é zero

    # Calcular o MAPE
    mape = np.mean(np.abs(relative_errors)) * 100

    st.write(f"MAE: {mae:.2f}")
    st.write(f"MSE: {mse:.2f}")
    st.write(f"RMSE: {rmse:.2f}")
    st.write(f"MAPE: {mape:.2f}%")




with tab6:
    # Ajuste do modelo SARIMA
    model_s1 = SARIMAX(ibovespa['Fechamento'], order=(0, 1, 0), seasonal_order=(1, 0, 1, 12)).fit(dis=-1)

    st.title("Análise dos resíduos")

    st.markdown("""
    Para realizar o diagnóstico do modelo SARIMA, utilizamos os mesmos artifícios gráficos de quando avaliamos o modelo ARIMA. Abaixo plotamos os gráficos que nos ajudarão a validar as previsões e o modelo que melhor se ajustou aos nossos dados.
    """)

    model_s1.plot_diagnostics(figsize=(15, 12))
    plt.savefig("diagnostics.png")
    st.image("diagnostics.png")

    st.markdown("""
    - **Gráfico 1**:  Como podemos observar, os resíduos apresentam comportamento aleatório;
    - **Gráfico 2**: Histograma dos Resíduos - Em relação à distribuição dos resíduos, pode-se observar que se assemelham à distribuição Normal;
    - **Gráfico 3**: Normal Q-Q - Os pontos estão sobrepostos a linha vermelha, o que indica que o resíduos seguem uma distribuição Normal;
    - **Gráfico 4**: Mede o  ACF dos resíduos onde pelo menos 95% dos lags não podem ser significativos e estão dentro da faixa significância.
    """,unsafe_allow_html=True)
    