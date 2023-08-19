import streamlit as st
from PIL import Image
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Analise Exploratoria", page_icon=":house:")

image = Image.open("./src/img/ibovespa.jpg")
st.image(image)



ibovespa = pd.read_csv('./src/data/BD.csv', sep=',')
ibovespa = ibovespa.iloc[:, :2]
ibovespa = ibovespa.rename(columns={'Último':'Fechamento'})
ibovespa['Data'] = pd.to_datetime(ibovespa['Data'],format='%d.%m.%Y')
ibovespa = ibovespa[::-1]

tab1, tab2, tab3 = st.tabs(["Introdução", "Descritivo","Tendência"])

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
    # A Jornada do Ibovespa

    Ao olhar para o vasto mundo dos mercados financeiros, um índice se destaca no Brasil: o Ibovespa. Ele é o principal indicador do desempenho médio das cotações das ações negociadas na B3, a bolsa de valores brasileira. Mas, o que realmente aconteceu com o Ibovespa ao longo dos anos? Vamos embarcar em uma viagem no tempo e descobrir!

    ## O Início (2000-2005)
    No início do novo milênio, o Brasil enfrentou desafios econômicos, mas também oportunidades. O país estava se adaptando à globalização e às mudanças tecnológicas. O Ibovespa refletiu essas dinâmicas, mostrando flutuações à medida que o mercado reagia a eventos nacionais e internacionais.

    ## O Boom das Commodities (2006-2011)
    Com a crescente demanda global por commodities, o Brasil, rico em recursos naturais, se beneficiou enormemente. O Ibovespa atingiu novos patamares à medida que as empresas ligadas a commodities, como a Vale e a Petrobras, viram suas ações subirem.

    ## Desafios e Resiliência (2012-2017)
    A economia global enfrentou desafios, desde a crise financeira global até a desaceleração do crescimento em mercados emergentes. O Ibovespa não foi imune a esses desafios, mas mostrou resiliência, refletindo a capacidade do mercado brasileiro de se adaptar e superar adversidades.


    ## O Presente (2018-2023)
    Nos anos mais recentes, o Ibovespa experimentou novas dinâmicas, com avanços tecnológicos, mudanças políticas e eventos globais moldando seu curso. O que o futuro reserva? Só o tempo dirá.

    """, unsafe_allow_html=True)

    fig = go.Figure()

    # Adicionando a série temporal ao gráfico
    fig.add_trace(go.Scatter(x=ibovespa['Data'], y=ibovespa['Fechamento'], mode='lines+markers', name='Fechamento'))

    # Configurando o layout do gráfico
    fig.update_layout(
    title={
        'text': 'Série Temporal dos valores diários de fechamento do índice Bovespa',
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {
            'size': 20,
            'color': '#306998'
        }
    },
    xaxis_title='Anos',
    yaxis_title='Fechamentos diários do índice',
    xaxis=dict(
        tickvals=ibovespa['Data'][::365],  # Escolhe um ponto a cada 365 dias (aproximadamente 1 ano)
        ticktext=ibovespa['Data'][::365].dt.year,  # Mostra apenas o ano
        tickangle=-45,  # Inclina os rótulos para melhor visualização
        title_font=dict(size=18, color='#CD8D00'),
        tickfont=dict(size=14, color='#333')
    ),
    yaxis=dict(
        title_font=dict(size=18, color='#CD8D00'),
        tickfont=dict(size=14, color='#333')
    ),

    )


    # Mostrando o gráfico
    st.plotly_chart(fig)

    st.write("""
    ## Em Direção à Modelagem
    Agora que temos uma visão clara da trajetória do Ibovespa ao longo dos anos, é hora de mergulhar mais fundo. Vamos usar técnicas avançadas de Machine Learning para prever o futuro deste índice icônico. Junte-se a nós nesta emocionante jornada de descoberta!
    """,unsafe_allow_html=True )

with tab2:
    ibovespa['Fechamento'] = ibovespa['Fechamento'].astype(str)

    # Removendo o ponto (.) usado como separador de milhares
    ibovespa['Fechamento'] = ibovespa['Fechamento'].str.replace('.', '')

    # Convertendo a coluna 'Fechamento' de volta para float
    ibovespa['Fechamento'] = ibovespa['Fechamento'].astype(float)

    fechamento = ibovespa['Fechamento']
    
    media = fechamento.mean()
    mediana = fechamento.median()
    desvio_padrao = fechamento.std()
    minimo = fechamento.min()
    maximo = fechamento.max()

    st.title('Análise Descritiva da Ibovespa')

    st.markdown("""
    ## Introdução
    A **análise descritiva** é uma das primeiras e mais importantes etapas na exploração de dados. Ela nos permite compreender a estrutura e as características fundamentais dos dados, fornecendo uma base sólida para análises subsequentes.

    Neste contexto, vamos explorar o comportamento da **Ibovespa**, o principal indicador do mercado de ações brasileiro. Através de métricas simples, como média, mediana, desvio padrão, e valores extremos, conseguiremos ter uma visão clara da trajetória deste índice ao longo do tempo.

    Ao final desta análise, esperamos ter uma compreensão mais profunda das tendências, volatilidades e características distintivas da Ibovespa, preparando o terreno para análises mais avançadas e modelagens preditivas.
    """)

    st.markdown("""
        ## Média:
        A média do fechamento diário nos dá uma ideia do valor central da Ibovespa ao longo do período analisado. Se a média for significativamente diferente da mediana, isso pode indicar a presença de outliers ou uma distribuição assimétrica.
    """)
    st.markdown(f"**Média:** {media:.2f}")
    st.markdown("""
        ## Mediana:
        A mediana nos mostra o valor do meio quando todos os fechamentos diários são organizados em ordem. É menos suscetível a outliers do que a média e pode oferecer uma visão mais "realista" do valor típico.
    """)
    st.markdown(f"**Mediana:** {mediana:.2f}")
    st.markdown("""
        ## Desvio Padrão:
        O desvio padrão mede a dispersão ou volatilidade da Ibovespa. Um desvio padrão alto indica que os fechamentos diários variaram muito em relação à média, enquanto um desvio padrão baixo sugere que os fechamentos foram consistentemente próximos da média.
    """)
    st.markdown(f"**Desvio Padrão:** {desvio_padrao:.2f}")
    st.markdown("""
        ## Mínimo e Máximo:
        Os valores mínimo e máximo nos dão uma ideia da amplitude dos fechamentos diários. Eles podem nos ajudar a identificar eventos extremos, como crises financeiras ou booms econômicos.
    """)
    st.markdown(f"**Mínimo:** {minimo:.2f}")
    st.markdown(f"**Máximo:** {maximo:.2f}")

    st.markdown("""
    ### Conclusão da Análise Descritiva

    A partir da análise descritiva realizada, observamos os seguintes pontos-chave sobre o índice Ibovespa:

    - **Média**: O valor médio do índice ao longo do período analisado é de 53,115.97, indicando o nível geral em torno do qual o índice oscilou.
    - **Mediana**: O valor mediano de 53,766.50 sugere que, em metade das ocasiões, o índice estava acima deste valor e, na outra metade, abaixo. Isso também indica que a distribuição dos valores não é fortemente inclinada, pois a média e a mediana estão próximas.
    - **Desvio Padrão**: Um desvio padrão de 33,539.88 mostra a volatilidade do índice. Quanto maior o desvio padrão, maior a variação em relação à média.
    - **Mínimo e Máximo**: O índice atingiu um mínimo de 116.00 e um máximo de 130,776.00 durante o período analisado, mostrando a vasta gama de variação que o Ibovespa experimentou.

    Estes insights nos fornecem uma compreensão clara do comportamento histórico do índice Ibovespa. Com esta base, estamos bem posicionados para realizar análises mais detalhadas, identificar tendências e, eventualmente, avançar para a modelagem preditiva.
    """)
    
with tab3:
    st.title('Tendência da Ibovespa')

    st.markdown("""
    ## Por que usar médias móveis?

    A análise de séries temporais, como os preços diários de fechamento da Ibovespa, pode ser desafiadora devido à volatilidade inerente desses dados. Flutuações diárias podem ser influenciadas por uma miríade de fatores, desde notícias econômicas globais até eventos políticos locais.

    A **média móvel** é uma ferramenta estatística que suaviza essas flutuações, tornando mais fácil identificar tendências de longo prazo. Ao calcular a média de preços em uma janela de tempo específica que se move ao longo do tempo, obtemos uma linha suavizada que pode ajudar a:

    1. **Identificar Tendências**: Uma média móvel ascendente sugere uma tendência de alta, enquanto uma descendente indica uma tendência de baixa.
    2. **Reduzir o Ruído**: Ao suavizar flutuações de curto prazo, a média móvel ajuda a focar em movimentos de longo prazo.
    3. **Determinar Pontos de Entrada e Saída**: Em análise técnica, quando uma média móvel de curto prazo cruza acima de uma média móvel de longo prazo, isso pode indicar um bom momento para comprar. O oposto pode sugerir um momento de venda.
    """)


    # Calcular médias móveis
    ibovespa['MM50'] = ibovespa['Fechamento'].rolling(window=50).mean()
    ibovespa['MM200'] = ibovespa['Fechamento'].rolling(window=200).mean()

    # Criar o gráfico
    fig = go.Figure()

    # Adicionar as séries ao gráfico
    fig.add_trace(go.Scatter(x=ibovespa['Data'], y=ibovespa['Fechamento'], mode='lines', name='Ibovespa'))
    fig.add_trace(go.Scatter(x=ibovespa['Data'], y=ibovespa['MM50'], mode='lines', name='Média Móvel 50 Dias', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=ibovespa['Data'], y=ibovespa['MM200'], mode='lines', name='Média Móvel 200 Dias', line=dict(color='red')))

    # Estilização
    fig.update_layout(
        xaxis_title='Anos',
        yaxis_title="Pontuação",
        xaxis=dict(
            tickvals=ibovespa['Data'][::365],
            ticktext=ibovespa['Data'][::365].dt.year,
            tickangle=-45,
            title_font=dict(size=18, color='#CD8D00'),
            tickfont=dict(size=14, color='#333')
        ),
        yaxis=dict(
            title_font=dict(size=18, color='#CD8D00'),
            tickfont=dict(size=14, color='#333')
        ),
        title={
            'text': 'Série Temporal aplicando a media movel',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {
                'size': 20,
                'color': '#306998'
            }
        },
        legend=dict(
            title ="Lengeda",
            x=0.5,
            y=-0.3,
            xanchor='center',
            yanchor='top'),
        
    )

    st.plotly_chart(fig)
