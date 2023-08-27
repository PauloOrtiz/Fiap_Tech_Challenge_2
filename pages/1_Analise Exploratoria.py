import streamlit as st
from PIL import Image
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Analise Exploratoria", page_icon=":house:")

image = Image.open("./src/img/ibovespa.jpg")
st.image(image)




ibovespa = pd.read_csv('./src/data/ibovespa.csv', sep=',')
ibovespa['Data'] = pd.to_datetime(ibovespa['Data'],format='%Y-%m-%d')
ibovespa['Fechamento'] = pd.to_numeric(ibovespa['Fechamento'], errors='coerce')

tab1, tab2, tab3 = st.tabs(["Introdução", "Descritivo","Médias Móveis"])

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
    yaxis_title='Pontos',
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

    st.write(ibovespa.dtypes)
  

with tab2:
    

    fechamento = ibovespa['Fechamento']
    
    media = fechamento.mean()
    mediana = fechamento.median()
    desvio_padrao = fechamento.std()
    minimo = fechamento.min()
    maximo = fechamento.max()

    

    st.markdown("""
        <h1>Análise Descritiva da Ibovespa: Uma Visão Detalhada</h1>
        
        <h2>Introdução</h2>
        <p>A análise descritiva é o alicerce da exploração de dados. Ela desvenda a natureza e a estrutura dos dados, pavimentando o caminho para investigações mais profundas. Neste relatório, mergulhamos no universo da Ibovespa, o barômetro do mercado de ações brasileiro, para entender sua dinâmica ao longo do tempo.</p>
        
        <h2>Panorama Geral</h2>
        <p>A Ibovespa é mais do que apenas um número; é o reflexo da economia brasileira. Através de métricas chave, como média, mediana, desvio padrão e extremos, podemos decifrar sua trajetória e as histórias que ela conta.</p>
        
        <h2>Métricas Centrais</h2>
        <p><strong>Média:</strong> Representando o valor central, a média do fechamento diário é de <strong>58,721.11</strong>. Uma discrepância significativa entre média e mediana pode sinalizar outliers ou uma distribuição inclinada.</p>
        <p><strong>Mediana:</strong> O ponto médio da nossa série, <strong>56,381.00</strong>, oferece uma perspectiva equilibrada, minimizando o impacto de valores extremos.</p>
        
        <h2>Volatilidade e Variação</h2>
        <p><strong>Desvio Padrão:</strong> Com um valor de <strong>30,957.87</strong>, esta métrica revela a volatilidade da Ibovespa. Uma maior dispersão indica períodos de incerteza, enquanto uma menor sugere estabilidade.</p>
        <p><strong>Extremos:</strong></p>
        <ul>
            <li><strong>Mínimo:</strong> <strong>8,371.00</strong> - Momentos de retração ou crises.</li>
            <li><strong>Máximo:</strong> <strong>130,776.00</strong> - Picos de otimismo e crescimento.</li>
        </ul>
        
        <h2>Reflexões Finais</h2>
        <p>A Ibovespa, através de suas flutuações, narra a saga econômica do Brasil. Nossas observações:</p>
        <ul>
            <li>A proximidade entre média e mediana sugere uma distribuição equilibrada, com poucos períodos de extremos desproporcionais.</li>
            <li>A vasta diferença entre os valores mínimo e máximo destaca a resiliência e a capacidade de recuperação do mercado brasileiro.</li>
            <li>O desvio padrão considerável aponta para períodos de volatilidade, refletindo as diversas fases econômicas que o país atravessou.</li>
        </ul>
        
        <h2>Conclusão</h2>
        <p>Esta análise descritiva fornece uma visão panorâmica da Ibovespa, capturando sua essência e evolução. Com este entendimento, estamos prontos para aprofundar nossa análise, identificar padrões subjacentes e, possivelmente, prever futuras tendências.</p>
    """, unsafe_allow_html=True)
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
    ibovespa['MM30'] = ibovespa['Fechamento'].rolling(window=30).mean()
    ibovespa['MM180'] = ibovespa['Fechamento'].rolling(window=180).mean()

    # Criar o gráfico
    fig = go.Figure()

    # Adicionar as séries ao gráfico
    fig.add_trace(go.Scatter(x=ibovespa['Data'], y=ibovespa['Fechamento'], mode='lines', name='Ibovespa'))
    fig.add_trace(go.Scatter(x=ibovespa['Data'], y=ibovespa['MM30'], mode='lines', name='Média Móvel 30 Dias', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=ibovespa['Data'], y=ibovespa['MM180'], mode='lines', name='Média Móvel 180 Dias', line=dict(color='red')))

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
