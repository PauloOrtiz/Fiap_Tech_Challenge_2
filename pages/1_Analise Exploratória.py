import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf



st.set_page_config(page_title="Analise Exploratoria", page_icon=":house:")

image = Image.open("./src/img/ibovespa.jpg")
st.image(image)
tabs_font_css = """
<style>
.st-b3{
   display: flow;
}
</style>
"""

st.write(tabs_font_css, unsafe_allow_html=True)



ibovespa = pd.read_csv('./src/data/ibovespa.csv', sep=',')
ibovespa['Data'] = pd.to_datetime(ibovespa['Data'],format='%Y-%m-%d')
ibovespa['Fechamento'] = pd.to_numeric(ibovespa['Fechamento'], errors='coerce')

ibovespa2021 = pd.read_csv('./src/data/ibovespa2021.csv', sep=',')
ibovespa2021['Data'] = pd.to_datetime(ibovespa['Data'],format='%Y-%m-%d')
ibovespa2021['Fechamento'] = pd.to_numeric(ibovespa['Fechamento'], errors='coerce')

tab1,tab2,tab3,tab4,tab5 = st.tabs(["Histórico","Estatísticas Descritivas","Médias Móveis e Desvios","Autocorrelação","Decomposição"])

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
    

    st.title("Jornada Econômica do Brasil: Uma Visão Geral")

    st.header("O Início (2001-2005)")
    st.write("""
    O início do milênio foi marcado por desafios significativos para a economia brasileira. Em 2001, o país enfrentou uma crise energética que causou instabilidade na bolsa, com setores elétrico e metalúrgico sofrendo grandes perdas. O impacto dos ataques de 11 de setembro em Nova York também reverberou no Brasil, com muitas ações atingindo suas cotações mínimas. Em 2002, o cenário político trouxe incertezas com a eleição de Lula, mas até 2004, a economia mostrou sinais de recuperação, crescendo 5,2%.
    """)

    st.header("Crise Imobiliária dos EUA (2006-2008)")
    st.write("""
    A economia brasileira mostrou-se resiliente em 2006, com crescimento sustentado. No entanto, a crise imobiliária dos EUA em 2008 teve repercussões globais. A quebra do Lehman Brothers desencadeou uma retirada massiva de investimentos, afetando também o Brasil. O dólar disparou, mas medidas de estímulo ajudaram a mitigar os impactos, levando a um crescimento robusto de 7,5% em 2010.
    """)

    st.header("Boom das Commodities (2009-2011)")
    st.write("""
    A retomada econômica em 2009 foi marcada pelo crescimento consistente, especialmente no segundo semestre. Em 2010, o Brasil brilhou no cenário global, com destaque no grupo BRICS. A exploração do pré-sal pela Petrobrás e o aumento nas exportações de commodities, como minério de ferro e soja, fortaleceram a economia. Em 2011, apesar dos desafios globais, o Brasil mostrou resiliência, com o PIB crescendo 2,7%.
    """)

    st.header("Desafios e Resiliência (2012-2019)")
    st.write("""
    Este período foi desafiador para a bolsa brasileira, com uma queda acumulada de 39% entre 2012 e 2015. A crise política, a Operação Lava Jato e a recessão econômica impactaram negativamente. Em 2016, o Brasil enfrentou sua pior recessão, mas mostrou sinais de recuperação nos anos seguintes. A greve dos caminhoneiros em 2018 e a volatilidade do dólar trouxeram mais incertezas, mas a economia manteve-se resiliente.
    """)

    st.header("Covid-19 (2020-2022)")
    st.write("""
    A pandemia de COVID-19 em 2020 trouxe desafios sem precedentes. A economia brasileira, como muitas outras, sofreu impactos significativos. Setores foram afetados de maneira desigual, e o PIB registrou uma queda de 3,2%. A pandemia continuou a influenciar a economia nos anos subsequentes, com efeitos profundos em diversos setores.
    """)

    st.header("O Presente (2023)")
    st.write("""
    Em 2023, o Brasil mostra sinais de recuperação, com perspectivas econômicas otimistas. A queda das cotações de commodities e a valorização do real sugerem uma inflação controlada. No entanto, o cenário ainda é de cautela e expectativa. O Ibovespa, reflexo da economia brasileira, aguarda os próximos capítulos da história econômica global.
    """)

    st.write("O que o futuro reserva para a economia brasileira? Só o tempo dirá.")

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

    

    
  

with tab2:
    
    

    

    st.markdown("""
        # Análise Descritiva da Ibovespa: Uma Visão Detalhada
        
        ## Introdução
        A análise descritiva é o alicerce da exploração de dados. Ela desvenda a natureza e a estrutura dos dados, pavimentando o caminho para investigações mais profundas. Neste relatório, mergulhamos no universo da Ibovespa, o barômetro do mercado de ações brasileiro, para entender sua dinâmica ao longo do tempo.
        
        ## Panorama Geral
        A Ibovespa é mais do que apenas um número; é o reflexo da economia brasileira. Através de métricas chave, como média, mediana, desvio padrão e extremos, podemos decifrar sua trajetória e as histórias que ela conta.
        
        ## Métricas Centrais</h2>
        **Média:** Representando o valor central, a média do fechamento diário é de **58.721**. Uma discrepância significativa entre média e mediana pode sinalizar outliers ou uma distribuição inclinada.
        **Mediana:** O ponto médio da nossa série, **56.381**, oferece uma perspectiva equilibrada, minimizando o impacto de valores extremos.
        
        ## Volatilidade e Variação
        **Desvio Padrão:** Com um valor de **30.957**, esta métrica revela a volatilidade da Ibovespa. Uma maior dispersão indica períodos de incerteza, enquanto uma menor sugere estabilidade.
        **Extremos:**
        <ul>
            <li><b>Mínimo:</b> <b>8.371</b> - Momentos de retração ou crises.</li>
            <li><b>Máximo:</b> <b>130.776</b> - Picos de otimismo e crescimento.</li>
        </ul>
        
        ## Reflexões Finais
        A Ibovespa, através de suas flutuações, narra a saga econômica do Brasil. Nossas observações:
        <ul>
            <li>A proximidade entre média e mediana sugere uma distribuição equilibrada, com poucos períodos de extremos desproporcionais.</li>
            <li>A vasta diferença entre os valores mínimo e máximo destaca a resiliência e a capacidade de recuperação do mercado brasileiro.</li>
            <li>O desvio padrão considerável aponta para períodos de volatilidade, refletindo as diversas fases econômicas que o país atravessou.</li>
        </ul>
        
        ### Conclusão
        Esta análise descritiva fornece uma visão panorâmica da Ibovespa, capturando sua essência e evolução. Com este entendimento, estamos prontos para aprofundar nossa análise, identificar padrões subjacentes e, possivelmente, prever futuras tendências.
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
            'text': 'Série Temporal aplicando a Média Móvel',
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
            title ="Legenda",
            x=0.5,
            y=-0.3,
            xanchor='center',
            yanchor='top'),
        
    )

    st.plotly_chart(fig)
    
    st.title('Volatilidade da Ibovespa')

    st.markdown("""
    ## Por que usar desvio padrão móvel?

    A volatilidade é uma característica inerente aos mercados financeiros. O desvio padrão móvel é uma ferramenta que nos permite visualizar essa volatilidade ao longo do tempo. Ao calcular o desvio padrão dos preços em uma janela de tempo específica que se move ao longo do tempo, obtemos uma linha que indica a volatilidade:

    1. **Identificar Volatilidade**: Um desvio padrão móvel crescente sugere aumento da volatilidade, enquanto um decrescente indica estabilização.
    2. **Entender Riscos**: Períodos de alta volatilidade podem ser considerados mais arriscados.
    3. **Tomada de Decisão**: Investidores podem usar a volatilidade para ajustar suas estratégias de investimento.
    """)

    # Calcular desvio padrão móvel
    window = 30  # Janela de 30 dias
    ibovespa['DesvioPadrao'] = ibovespa['Fechamento'].rolling(window=window).std()

    # Criar o gráfico
    fig2 = go.Figure()

    # Adicionar as séries ao gráfico
    fig2.add_trace(go.Scatter(x=ibovespa['Data'], y=ibovespa['Fechamento'], mode='lines', name='Ibovespa'))
    fig2.add_trace(go.Scatter(x=ibovespa['Data'], y=ibovespa['DesvioPadrao'], mode='lines', name='Desvio Padrão Móvel 30 Dias', line=dict(color='green')))

    # Estilização
    fig2.update_layout(
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
            'text': 'Série Temporal com Desvio Padrão Móvel',
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
            title ="Legenda",
            x=0.5,
            y=-0.3,
            xanchor='center',
            yanchor='top'),
    )

    st.plotly_chart(fig2)
    
    st.title('')

    st.markdown("""
    ## Análise de Bollinger Bands para Ibovespa
    
    ### O que são Bollinger Bands?

    Bollinger Bands, ou Bandas de Bollinger, é uma ferramenta de análise técnica criada por John Bollinger. Elas são compostas por uma média móvel no centro e duas bandas de preço, acima e abaixo da média móvel, que representam os desvios padrão.

    ### Por que usar Bollinger Bands?

    1. **Identificar períodos de alta e baixa volatilidade**: As bandas se expandem durante períodos de alta volatilidade e se contraem durante períodos de baixa volatilidade.
    2. **Identificar potenciais pontos de compra e venda**: Quando o preço toca ou ultrapassa uma das bandas, pode ser um sinal de sobrecompra ou sobrevenda, respectivamente.
    3. **Tendências de mercado**: Quando os preços se movem fora das bandas, é um forte indicativo de continuação da tendência atual.

    ### Como interpretar?

    - **Toque ou ultrapassagem da banda superior**: Pode indicar que o ativo está sobrecomprado e pode estar pronto para uma reversão ou queda.
    - **Toque ou ultrapassagem da banda inferior**: Pode indicar que o ativo está sobrevendido e pode estar pronto para uma reversão ou alta.
    - **Preços se movendo acima da média móvel**: Indica uma tendência de alta.
    - **Preços se movendo abaixo da média móvel**: Indica uma tendência de baixa.

    
    """)
    
    window = 30
    ibovespa['MM30'] = ibovespa['Fechamento'].rolling(window=window).mean()
    ibovespa['Desvio'] = ibovespa['Fechamento'].rolling(window=window).std()

    # Calculando as Bollinger Bands
    k = 2
    ibovespa['Banda Superior'] = ibovespa['MM30'] + (ibovespa['Desvio'] * k)
    ibovespa['Banda Inferior'] = ibovespa['MM30'] - (ibovespa['Desvio'] * k)

    # Plotando o gráfico
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=ibovespa['Data'], y=ibovespa['Fechamento'], mode='lines', name='Ibovespa'))
    fig.add_trace(go.Scatter(x=ibovespa['Data'], y=ibovespa['MM30'], mode='lines', name='Média Móvel 30 Dias', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=ibovespa['Data'], y=ibovespa['Banda Superior'], mode='lines', name='Banda Superior', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=ibovespa['Data'], y=ibovespa['Banda Inferior'], mode='lines', name='Banda Inferior', line=dict(color='green')))

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
            'text': 'Série do Ibovespa com as bandas de Bollinger',
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
            title ="Legenda",
            x=0.5,
            y=-0.3,
            xanchor='center',
            yanchor='top'),
    )
    
    st.plotly_chart(fig)

with tab4: 
    lag_acf = acf(ibovespa2021['Fechamento'], nlags=40)
    lag_pacf = pacf(ibovespa2021['Fechamento'], nlags=40, method='ols')
    conf_int = 1.96/np.sqrt(len(ibovespa2021['Fechamento']))

    st.markdown("""
        ## Redução da Série Temporal
        Com o objetivo de simplificar a análise e facilitar os ajustes nos modelos diante da série de eventos caóticos que afetaram o fechamento diário do índice Bovespa, como as significativas quedas ocorridas durante a crise imobiliária dos EUA em 2008 e a pandemia em 2020, seguidas por períodos de recuperação, decidimos restringir a série temporal a um intervalo que vai de 04 de janeiro de 2021 a 01 de agosto de 2023. Isso proporcionará uma base mais estável e recente para nossa análise.
        
    """)
       
    st.markdown("""
    # Autocorrelação: ACF e PACF
    Ao mergulharmos no vasto oceano dos dados, é essencial entender as ondas e correntes que os movem. No mundo das séries temporais, essas "ondas" são muitas vezes as autocorrelações. Elas nos dão uma visão sobre como os valores em diferentes pontos no tempo estão relacionados entre si.

    ## O que é Autocorrelação?
    Imagine um lago tranquilo. Se você jogar uma pedra nele, as ondas se propagarão em círculos. A autocorrelação é semelhante a isso, mas em vez de ondas em um lago, estamos olhando para como um valor em uma série temporal se relaciona com outros valores em pontos de tempo anteriores.

    ## ACF (Autocorrelation Function)
    A Função de Autocorrelação, ou ACF, nos dá uma visão geral da autocorrelação em todos os atrasos. É como olhar para o lago de cima e ver todas as ondas que a pedra criou. O ACF nos mostra a correlação entre a série e sua versão defasada.
    """,unsafe_allow_html=True)
        
    fig_acf = go.Figure()
    fig_acf.add_trace(go.Scatter(y=lag_acf, mode='lines+markers'))
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
    ## PACF (Partial Autocorrelation Function)
    A Função de Autocorrelação Parcial, ou PACF, é um pouco mais específica. Ela nos mostra a autocorrelação em um atraso, controlando os atrasos anteriores. Usando nossa analogia do lago, é como focar em uma onda específica, ignorando todas as outras.
    """,unsafe_allow_html=True)

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
    ## Por que usar ACF e PACF?
    <ul>
        <li><b>Entender os Dados</b>: ACF e PACF nos ajudam a entender a estrutura temporal dos dados, revelando padrões e tendências.</p></li>
        <li><b>Modelagem</b>: Eles são ferramentas essenciais ao decidir os termos de um modelo ARIMA. Por exemplo, o ACF pode nos ajudar a identificar a ordem de média móvel (MA), enquanto o PACF pode nos ajudar com a ordem autoregressiva (AR).</li>
        <li><b>Detectar Estacionariedade</b>: A estacionariedade é uma propriedade crucial para muitos modelos de séries temporais. ACF e PACF podem nos ajudar a identificar se uma série é estacionária ou se precisa de diferenciação.</li>
    </ul>

    ### Conclusão
    
    O decaimento lento na ACF sugere que há dependência temporal nas observações da série ao longo de vários lags anteriores, o que pode ser indicativo de uma componente de tendência ou sazonalidade na série.

    O corte abrupto da PACF no terceiro lag sugere que, após considerar os efeitos dos três primeiros lags, a série não possui correlação parcial significativa com lags posteriores. Isso pode indicar que um modelo autorregressivo de ordem 2 (AR(2)) pode ser apropriado, pois os lags 1 e 2 têm correlações parciais significativas, enquanto as correlações parciais com lags posteriores são próximas de zero.

    A combinação de uma ACF com decaimento lento e uma PACF com decaimento exponencial pode ser indicativa de um modelo ARIMA, onde a ordem do termo autorregressivo (p) é 2 (com base na PACF) e a ordem de diferenciação (d) pode ser escolhida para lidar com a tendência ou sazonalidade na série.
    
    """, unsafe_allow_html=True)


with tab5:

    st.markdown("""
    # Decompondo a Série do Ibovespa

    <p>Ao embarcar na jornada de modelagem de séries temporais, é essencial entender a natureza dos dados com os quais estamos lidando. Uma série temporal é uma sequência de pontos de dados, medidos tipicamente em intervalos de tempo sucessivos. No nosso caso, estamos analisando o fechamento diário do índice Bovespa.</p>

    ## Por que a Decomposição?

    <p>A decomposição de uma série temporal é uma técnica estatística que transforma uma série temporal em múltiplos componentes diferentes. Cada um desses componentes representa uma parte específica da informação contida na série original. Ao fazer isso, podemos entender melhor a complexidade e a estrutura subjacente da série.</p>

    ### Componentes da Decomposição:
    
    <ol>
        <li><strong>Série Temporal Observada:</strong> Esta é a série original, o conjunto de dados que estamos analisando.</li>
        <li><strong>Tendência:</strong> A tendência mostra um padrão subjacente na série. Em outras palavras, é uma linha suavizada que captura a direção em que nossos dados estão se movendo.</li>
        <li><strong>Sazonalidade:</strong> A sazonalidade captura os padrões que se repetem em intervalos fixos de tempo. Por exemplo, um aumento nas vendas de sorvete durante os meses de verão é um exemplo de sazonalidade.</li>
        <li><strong>Resíduos:</strong> Após extrair a tendência e a sazonalidade da série original, o que resta é chamado de resíduo. Os resíduos são a diferença entre a série original e o que foi explicado pela tendência e sazonalidade.</li>
    </ol>    
    """, unsafe_allow_html=True)

    
    decomposicao = seasonal_decompose(ibovespa2021['Fechamento'], model='additive', period=12)

    # Criando subplots com espaçamento vertical
    fig = make_subplots(rows=4, cols=1, 
                        subplot_titles=("Série Temporal Observada", "Tendência", "Componente Sazonal", "Resíduos"),
                        vertical_spacing=0.2)  # Ajuste o valor conforme necessário

    # Adicionando os componentes da decomposição
    fig.add_trace(go.Scatter(y=decomposicao.observed, mode='lines', name='Série Temporal Observada'), row=1, col=1)
    fig.add_trace(go.Scatter(y=decomposicao.trend, mode='lines', name='Tendência'), row=2, col=1)
    fig.add_trace(go.Scatter(y=decomposicao.seasonal, mode='lines', name='Componente Sazonal'), row=3, col=1)
    fig.add_trace(go.Scatter(y=decomposicao.resid, mode='lines', name='Resíduos'), row=4, col=1)

    # Atualizando o layout
    fig.update_layout(title={
        'text': 'Decomposição da Série Temporal',
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {
            'size': 20,
            'color': '#306998'
        }
    }, showlegend=False)

    # Mostrando o gráfico no Streamlit
    st.plotly_chart(fig)

    st.markdown("""
    ## Por que é uma Boa Prática?

    <p>A decomposição nos permite ver claramente o que, de outra forma, poderia ser um padrão confuso. Ao entender a tendência, podemos fazer previsões de longo prazo. Ao entender a sazonalidade, podemos prever padrões de curto prazo. E ao analisar os resíduos, podemos entender o que não foi capturado pelos dois primeiros componentes e, assim, melhorar nossos modelos.</p>

    <p>Além disso, a decomposição pode ajudar a identificar se uma série temporal é aditiva ou multiplicativa. Uma série é considerada aditiva quando a magnitude da sazonalidade e da tendência não varia com o tempo. Por outro lado, em uma série multiplicativa, a magnitude da sazonalidade ou da tendência varia em proporção ao nível da série.</p>

    <p>Em resumo, antes de mergulhar em modelos avançados, é sempre uma boa prática decompor a série para entender seus componentes. Isso não apenas melhora nossa compreensão dos dados, mas também nos guia na escolha do modelo mais adequado para fazer previsões futuras.</p>
    """,  unsafe_allow_html=True)

