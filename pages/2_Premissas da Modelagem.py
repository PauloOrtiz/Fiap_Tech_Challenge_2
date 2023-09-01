import streamlit as st
from PIL import Image
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

st.set_page_config(page_title="Premissas da Modelagem", page_icon=":house:")

image = Image.open("./src/img/Model.jpg")
st.image(image)



ibovespa = pd.read_csv('./src/data/ibovespa.csv', sep=',')
ibovespa['Data'] = pd.to_datetime(ibovespa['Data'],format='%Y-%m-%d')
ibovespa['Fechamento'] = pd.to_numeric(ibovespa['Fechamento'], errors='coerce')


tab1, tab2 = st.tabs(["Modelagem", "Estacionariedade"])

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

    st.markdown("""
    # Iniciando a Modelagem

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

    
    decomposicao = seasonal_decompose(ibovespa['Fechamento'], model='additive', period=12)

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
    
with tab2:
    

    # Título da seção
   
    st.markdown("""
    # Estacionariedade e a Importância do Teste de Dickey-Fuller       
                
                 
    ## Teste de Dickey-Fuller
                
    Ao trabalhar com séries temporais, um dos conceitos mais cruciais é a estacionariedade. Uma série temporal é dita estacionária se suas propriedades estatísticas, como média, variância e autocorrelação, são constantes ao longo do tempo. A maioria dos modelos de séries temporais, como ARIMA, assume que a série é estacionária. Uma série não estacionária pode ser problemática porque pode ser difícil modelar e prever.

    O Teste de Dickey-Fuller Aumentado (ADF) é um dos métodos mais comuns para verificar a estacionariedade de uma série temporal. A hipótese nula do teste é que a série temporal é não estacionária. Se o valor-p do teste for menor ou igual a um nível de significância (geralmente 0,05), rejeitamos a hipótese nula e concluímos que a série é estacionária.

    Vamos agora realizar o Teste de Dickey-Fuller em nossa série temporal do Ibovespa e interpretar os resultados:
    """,unsafe_allow_html=True)

    # Função para realizar o teste
    def perform_adf_test(series):
        result = adfuller(series)
        return result

    # Executando o teste
    ibovespa_series = ibovespa['Fechamento']
    result = perform_adf_test(ibovespa_series)

    # Mostrando os resultados no Streamlit
    st.markdown("**Resultado do Teste Dickey-Fuller:**")
    st.markdown(f'**Estatística de teste:** {result[0]}')
    st.markdown(f'**Valor-p:** {result[1]}')
    st.markdown(f'**Número de defasagens usadas:** {result[2]}')
    st.markdown(f'**Número de observações:** {result[3]}')
    st.markdown("**Valores críticos:**")
    for key, value in result[4].items():
        st.markdown(f'   **{key}:** {value}')

    if result[1] <= 0.05:
        st.markdown("**Conclusão:** A Série é estacionária (rejeita-se a hipótese nula)")
    else:
        st.markdown("**Conclusão:** A Série não é estacionária (não rejeita-se a hipótese nula)")

    st.markdown("""
    ### Com base no Teste Dickey-Fuller, podemos observar o seguinte:

    - A **Estatística de teste** é de aproximadamente -0.862, que é maior do que todos os valores críticos (1%, 5% e 10%). 
    - O **Valor-p** é de 0.8003, que é significativamente maior que 0.05. 

    Em testes estatísticos, um valor-p menor que 0.05 é frequentemente usado para rejeitar a hipótese nula. No contexto do Teste Dickey-Fuller, a hipótese nula é de que a série temporal possui uma raiz unitária e, portanto, não é estacionária.

    Como o valor-p é maior que 0.05 e a estatística de teste é maior que os valores críticos, **não rejeitamos a hipótese nula**. Isso sugere que a série temporal não é estacionária.

    Isso é importante porque muitos modelos de séries temporais, como ARIMA, assumem que a série é estacionária. Se a série não for estacionária, pode ser necessário aplicar transformações, como diferenciação, para torná-la estacionária antes de modelar.
    """)

    st.markdown("""
    ## Transformando a Série em Estacionária através da Diferenciação

    A diferenciação é uma técnica comum usada para tornar uma série temporal estacionária. Uma série estacionária é aquela cujas propriedades estatísticas, como média e variância, permanecem constantes ao longo do tempo. Muitos modelos de séries temporais, como ARIMA, exigem que a série seja estacionária para fazer previsões precisas.

    Ao aplicar a diferenciação, estamos basicamente subtraindo a observação atual da observação anterior. Isso pode ajudar a remover tendências e padrões sazonais, tornando a série mais estacionária.

    Vamos visualizar a série após aplicar uma primeira diferenciação para entender melhor como ela se transformou.
    """)

    ibovespa_diff1 = ibovespa.diff().dropna()    
    # Gráfico usando Plotly
    if 'Data' not in ibovespa_diff1.columns:
        st.write("A coluna 'Data' não existe em ibovespa_diff1!")
    # Definindo 'Data' como índice para ibovespa antes da diferenciação
    ibovespa.set_index('Data', inplace=True)
    ibovespa_diff1 = ibovespa.diff().dropna()

    # Verificando se a coluna 'Data' é do tipo datetime
    if not pd.api.types.is_datetime64_any_dtype(ibovespa_diff1.index):
        st.write("O índice de ibovespa_diff1 não é do tipo datetime!")
    # Se não for datetime, você pode converter o índice para datetime aqui, se necessário

    # Plotando o gráfico
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ibovespa_diff1.index, y=ibovespa_diff1['Fechamento'], mode='lines', name='1º Diferenciação'))

    # Configurando o layout do gráfico
    unique_years = ibovespa_diff1.index.year.unique()
    fig.update_layout(
        title={
        'text': "Série com 1º Diferenciação",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {
            'size': 20,
            'color': '#306998'
        }},
        xaxis_title='Anos',
        yaxis_title="Valor",
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

    st.markdown("""
    ## Por que realizar o Teste Dickey-Fuller após a diferenciação?

    Quando trabalhamos com modelos de séries temporais, como o ARIMA, é essencial que a série seja estacionária. Uma série estacionária é aquela cujas propriedades estatísticas, como média e variância, não variam com o tempo. Modelos de séries temporais, como o ARIMA, têm a suposição implícita de estacionariedade. Se essa suposição não for satisfeita, os resultados do modelo podem ser imprecisos e enganosos.

    A diferenciação é uma técnica comum para transformar uma série não estacionária em estacionária. No entanto, após a diferenciação, é crucial retestar a série para confirmar se ela se tornou estacionária. O Teste Dickey-Fuller é uma ferramenta poderosa para essa verificação.
    """)

    # Função para realizar o teste
    def perform_adf_test(series):
        result = adfuller(series)
        st.markdown("**Resultado do Teste Dickey-Fuller:**")
        st.markdown(f'**Estatística de teste:** {result[0]}')
        st.markdown(f'**Valor-p:** {result[1]}')
        st.markdown(f'**Número de defasagens usadas:** {result[2]}')
        st.markdown(f'**Número de observações:** {result[3]}')
        st.markdown("**Valores críticos:**")
        for key, value in result[4].items():
            st.markdown(f'   **{key}:** {value}')

        if result[1] <= 0.05:
            st.markdown("**Conclusão:** A Série é estacionária (rejeita-se a hipótese nula)")
        else:
            st.markdown("**Conclusão:** A Série não é estacionária (não rejeita-se a hipótese nula)")

    # Chamando a função
    perform_adf_test(ibovespa_diff1['Fechamento'])

    # Conclusão
    st.markdown("""
    ### Conclusão:

    Após a diferenciação, a série tornou-se estacionária, conforme confirmado pelo Teste Dickey-Fuller. Com um valor-p de 0.0, rejeitamos a hipótese nula, indicando que a série é estacionária. Isso é um bom sinal, pois agora podemos prosseguir com a modelagem ARIMA, sabendo que a suposição de estacionariedade foi atendida.
    """)

    st.write("""
    ## Em Direção à Modelagem
    Agora que temos uma visão clara da trajetória do Ibovespa ao longo dos anos, é hora de mergulhar mais fundo. Vamos usar técnicas avançadas de Machine Learning para prever o futuro deste índice icônico. Junte-se a nós nesta emocionante jornada de descoberta!
    """,unsafe_allow_html=True )
