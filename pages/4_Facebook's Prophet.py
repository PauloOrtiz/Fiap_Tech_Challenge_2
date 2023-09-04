import streamlit as st
from PIL import Image
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np



st.set_page_config(page_title="Prophet", page_icon=":house:")

image = Image.open("./src/img/Prophet.png")
st.image(image)

ibovespa = pd.read_csv('./src/data/ibovespa2021.csv', sep=',')
ibovespa['Data'] = pd.to_datetime(ibovespa['Data'],format='%Y-%m-%d')
ibovespa = ibovespa.rename(columns={'Data':'ds'})
ibovespa['Fechamento'] = pd.to_numeric(ibovespa['Fechamento'], errors='coerce')
acuracia = pd.read_csv('./src/data/ACURACIA_prophet.csv', sep=';')


tab1, tab2, tab3, tab4 = st.tabs(["Prophet", "Previsões e Decomposição", "Acurácia", "Validação do Modelo"])

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

    st.markdown(
        """
        # Prophet: O Oráculo Moderno

        Em um mundo onde os dados são abundantes e as tendências são voláteis, surge o Prophet, uma ferramenta de previsão para séries temporais desenvolvida pelo Facebook. Como um oráculo moderno, o Prophet é capaz de olhar para os padrões do passado e prever o que o futuro pode trazer.

        O Prophet é especialmente projetado para lidar com séries temporais que possuem padrões sazonais fortes e vários pontos de mudança. Se o ARIMA é um viajante do tempo, o Prophet é um vidente, capaz de entender e prever tendências, sazonalidades e feriados.
        
        ## Por que o Prophet é Especial?

        O Prophet se destaca em sua capacidade de lidar com dados faltantes, tendências que mudam ao longo do tempo e até mesmo feriados! Ele foi projetado para ser flexível e intuitivo, tornando a previsão de séries temporais uma tarefa mais simples, mesmo para aqueles que não são especialistas em estatística.

        ## A Magia por Trás do Prophet

        O Prophet utiliza um modelo aditivo, onde tendências não lineares são ajustadas à sazonalidade anual, semanal e aos efeitos dos feriados. Ele automaticamente detecta pontos de mudança nas tendências e permite que os usuários especifiquem suas próprias sazonalidades e feriados.

        ## A Fórmula do Prophet

        Embora o Prophet seja poderoso, sua fórmula é elegante em sua simplicidade. Aqui está uma visão geral:
        """
    )
    with st.expander("Ver a fórmula do Prophet"):
        st.latex(r"""
            y(t) = g(t) + s(t) + h(t) + \epsilon_t
        """)
        st.markdown("""
            Onde:
            - $y(t)$ é a previsão.
            - $g(t)$ representa a tendência.
            - $s(t)$ captura a sazonalidade.
            - $h(t)$ representa os efeitos dos feriados.
            - $epsilon_t$ é o erro.
            """)
        
with tab2:
    
    st.title("Previsão de Séries Temporais com Prophet")

    st.write("""
    ## Previsão com Prophet

    Abaixo, você pode ver a previsão feita pelo Prophet. Os dados de treinamento são mostrados em azul, enquanto a previsão é mostrada em vermelho.
    """)


    ibovespa = ibovespa.rename(columns={"Fechamento": "y"})
    df = pd.DataFrame(ibovespa)
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=0)
    forecast = model.predict(future)
    
   
    train_size = int(0.80 * len(ibovespa))
    train_df = ibovespa.iloc[:train_size]
    test_df = ibovespa.iloc[train_size:]
        
    trace1 = go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Previsão')
    trace2 = go.Scatter(x=train_df['ds'], y=train_df['y'], mode='lines', name='Treinamento')
    trace3 = go.Scatter(x=test_df['ds'], y=test_df['y'], mode='lines', name='Teste')

    layout = go.Layout(title='Previsão com Prophet (Treinamento e Teste)', xaxis=dict(title='Data'), yaxis=dict(title='Pontos do Ibovespa'))
    fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
    fig.update_layout(
    title={
    'text': "Previsão com Prophet (Treinamento e Teste)",
    'y':0.95,
    'x':0.5,
    'xanchor': 'center',
    'yanchor': 'top',
    'font': {
        'size': 20,
        'color': '#306998'
    }},xaxis_title='Anos',
    yaxis_title="Pontuação`",
    xaxis=dict(
        tickangle=-45,  # Inclina os rótulos para melhor visualização
        title_font=dict(size=18, color='#CD8D00'),
        tickfont=dict(size=14, color='#333')
    ),
    template="plotly_dark",
    yaxis=dict(
        title_font=dict(size=18, color='#CD8D00'),
        tickfont=dict(size=14, color='#333')
    )
    
    )

    st.plotly_chart(fig)

    st.write("""
    ## Decomposição da Série Temporal

    Uma das características mais úteis do Prophet é sua capacidade de decompor automaticamente uma série temporal em seus componentes. 
    Isso inclui a tendência geral, sazonalidade anual e efeitos de feriados (se fornecidos).
    Abaixo, você pode ver a decomposição da tendência e sazonalidade da série.
    """)


    fig = make_subplots(rows=2, cols=1, subplot_titles=('Tendência', 'Sazonalidade'))

    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['trend'], name='Tendência'), row=1, col=1)
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yearly'], name='Sazonalidade'), row=2, col=1)

   
    fig.update_layout(
    title={
    'text': "Decomposição da Série Temporal",
    'y':0.95,
    'x':0.5,
    'xanchor': 'center',
    'yanchor': 'top',
    'font': {
        'size': 20,
        'color': '#306998'
    }},
    )
    
    st.plotly_chart(fig)
         
    st.write("""
    ## Conclusão

    Através do Prophet, conseguimos obter uma previsão robusta para os pontos do Ibovespa. 
    A decomposição da série nos ajudou a entender melhor os componentes subjacentes da série temporal.
    Esta análise pode ser estendida para prever mais pontos no futuro, ajustar hiperparâmetros ou incorporar mais informações, como feriados, para melhorar a previsão.
    """)

with tab3:


    st.title("Avaliação da Acurácia do Modelo")

    st.write("""
    ## Observados vs Preditos

    Uma das maneiras mais diretas de avaliar a performance de um modelo de previsão é comparar os valores previstos com os valores reais observados. 
    Isso nos dá uma visualização clara de onde o modelo está acertando e onde está errando. 
    A proximidade entre as duas linhas (observada e prevista) indica a precisão do modelo. 
    Uma sobreposição perfeita indicaria um modelo perfeito, o que raramente é o caso na prática.
    """)


    y_pred = forecast['yhat'][train_size:]
    y_true = test_df['y']
    x_indices = np.arange(len(y_true))

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x_indices, y=y_true, mode='lines', name='Valores Observados', line=dict(color='blue')))

    # Adicionar os valores previstos
    fig.add_trace(go.Scatter(x=x_indices, y=y_pred, mode='lines', name='Valores Preditos', line=dict(color='red')))

    # Definir os títulos e rótulos dos eixos
    fig.update_layout(
    title={
    'text': "Valores Observados vs Valores Preditos",
    'y':0.95,
    'x':0.5,
    'xanchor': 'center',
    'yanchor': 'top',
    'font': {
        'size': 20,
        'color': '#306998'
    }},xaxis_title='Número de Observações da Série',
    yaxis_title="Pontuação`",
    xaxis=dict(
        tickangle=-45,  # Inclina os rótulos para melhor visualização
        title_font=dict(size=18, color='#CD8D00'),
        tickfont=dict(size=14, color='#333')
    ),
    template="plotly_dark",
    yaxis=dict(
        title_font=dict(size=18, color='#CD8D00'),
        tickfont=dict(size=14, color='#333')
    )
    
    )
    

    # Exibir a figura no Streamlit
    st.plotly_chart(fig)

    st.write("""
    ## Métricas de Acurácia

    Para quantificar a precisão do modelo, usamos métricas de acurácia. Vamos entender o que cada métrica significa:

    - **MSE (Mean Squared Error)**: É a média dos quadrados dos erros. Quanto maior o número, pior é o modelo. É útil porque penaliza grandes erros.
    - **MAE (Mean Absolute Error)**: É a média dos erros absolutos. Dá uma ideia direta de quão errado são as previsões.
    - **MAPE (Mean Absolute Percentage Error)**: É a média dos erros percentuais absolutos. Dá uma ideia da precisão em termos percentuais.

    """)
    table_html = acuracia.to_html(index=False)

    st.write(table_html, unsafe_allow_html=True)

    st.write("""
    ## Conclusão

    Observando as métricas de acurácia, podemos concluir que o modelo tem um desempenho razoável. 
    O MAPE de 7,788% indica que, em média, o modelo erra em cerca de 7,78% nas suas previsões. 
    Dependendo do contexto e da aplicação, esse nível de precisão pode ser considerado bom. 
    No entanto, sempre há espaço para melhorias, e é importante considerar outras técnicas ou ajustes no modelo para melhorar ainda mais a acurácia.

    """)


    st.markdown(
        """
        
        """
    )

with tab4:

    st.markdown("""
    
    # Analisando o comportamento dos Resíduos do Modelo

    Com base na análise dos resíduos, podemos fazer uma avaliação geral do desempenho do modelo. Se os resíduos estiverem próximos da linha de referência zero e não houver padrões evidentes, isso sugere que o modelo está bem ajustado aos dados. Caso contrário, pode ser necessário reavaliar e ajustar o modelo.
    """)

    residuals = y_true - y_pred
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_df['ds'], y=residuals, mode='lines', name='Residuals'))
    fig.add_shape(
    type="line",
    x0=min(test_df['ds']),
    x1=max(test_df['ds']),
    y0=0,
    y1=0,
    line=dict(color="red", width=2, dash="dash"),
    )
    fig.update_layout(
    title={
    'text': "Resíduos do Modelo",
    'y':0.95,
    'x':0.5,
    'xanchor': 'center',
    'yanchor': 'top',
    'font': {
        'size': 20,
        'color': '#306998'
    }},xaxis_title='Data',
    yaxis_title="Resíduos",
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
    st.plotly_chart(fig)

    st.markdown("""
        A linha horizontal vermelha tracejada em y=0 é uma linha de referência que indica onde os resíduos deveriam estar idealmente se o modelo fosse perfeito. Desvios significativos dos resíduos em relação a esta linha podem indicar problemas no modelo. Conforme observamos, em geral, os erros se comportam de maneira relativamente próxima ao eixo y=0. 
        Outro fator a ser observado, é que os erros apresentam aleatoriedade, ou seja, não há como identificar padrões de sazonalidade ou tendência em sua série. 
        Além disso, se a dispersão dos resíduos se ampliar ou estreitar à medida que você se move ao longo do eixo x, isso pode ser um sinal de heteroscedasticidade, o que significa que a variância dos erros não é constante. Esse comportamento também, em geral, não foi identificado na série através do gráfico
    """,unsafe_allow_html=True)

    