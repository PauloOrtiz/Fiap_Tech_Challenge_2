import streamlit as st
from PIL import Image
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet



st.set_page_config(page_title="Prophet", page_icon=":house:")

image = Image.open("./src/img/Prophet.png")
st.image(image)

ibovespa = pd.read_csv('./src/data/ibovespa2021.csv', sep=',')
ibovespa['Data'] = pd.to_datetime(ibovespa['Data'],format='%Y-%m-%d')
ibovespa = ibovespa.rename(columns={'Data':'ds'})
ibovespa['Fechamento'] = pd.to_numeric(ibovespa['Fechamento'], errors='coerce')


tab1, tab2, tab3 = st.tabs(["Prophet", "Decomposição da série", "Modelos e Previsões"])

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
    unique_years = ibovespa.ds.year.unique()
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

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
    xaxis_title='Anos',
    yaxis_title="Pontuação",
    xaxis=dict(
        tickvals=pd.to_datetime([f'{year}-01-01' for year in unique_years]),  # Escolhe um ponto para cada ano único
        ticktext=unique_years,  # Mostra apenas o ano
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



with tab3:
    st.markdown(
        """
        # Escolhendo Modelos

        ## Modelo Prophet

        # Previsões e Erros

        ## Tabelas 

        ## Acuracia 
        """
    )