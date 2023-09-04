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
         
    

with tab3:

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

    table_html = acuracia.to_html(index=False)

    st.write(table_html, unsafe_allow_html=True)

    


    st.markdown(
        """
        
        """
    )

with tab4:
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

    df = pd.DataFrame({'Resíduos': residuals})
    trace = go.Histogram(
    x=df['Resíduos'],
    nbinsx=20,
    marker=dict(color='blue', line=dict(color='black', width=1))
    )

    layout = go.Layout(
        title='Distribuição dos Resíduos',
        xaxis=dict(title='Resíduos'),
        yaxis=dict(title='Frequência')
    )

    fig = go.Figure(data=[trace], layout=layout)
    st.plotly_chart(fig)
