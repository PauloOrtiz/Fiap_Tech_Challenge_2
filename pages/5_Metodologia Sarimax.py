import streamlit as st
from PIL import Image


st.set_page_config(page_title="Sarimax", page_icon=":house:")

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


tab1, tab2, tab3, tab4, tab5 = st.tabs(["SARIMAX", "Escolha do modelo", "Modelo para previsão","Acurácia", "Diagnostico do Modelo"])

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
    # SARIMAX: Uma Jornada Além do ARIMA

    Quando nos aventuramos no mundo das séries temporais, rapidamente nos deparamos com o modelo ARIMA, uma ferramenta poderosa e flexível. No entanto, o universo da previsão é vasto e, às vezes, precisamos de algo mais robusto, algo que considere sazonalidades e variáveis exógenas. É aqui que o SARIMAX entra em cena.

    ## O que é SARIMAX?

    O SARIMAX é uma extensão do ARIMA. O 'S' refere-se à sazonalidade, e o 'X' refere-se às variáveis exógenas. Em outras palavras, enquanto o ARIMA é excelente para prever séries temporais baseadas em seus próprios valores passados, o SARIMAX leva em consideração padrões sazonais e outras variáveis externas que podem influenciar a série.

    ## A Fórmula Mágica

    A beleza do SARIMAX reside em sua capacidade de combinar diferentes componentes:

    - **AR (AutoRegressivo)**: Baseia-se na relação entre uma observação e um número de observações defasadas.
    - **I (Integrado)**: Usa a diferenciação das observações para tornar a série temporal estacionária.
    - **MA (Média Móvel)**: Baseia-se na relação entre uma observação e um erro residual proveniente de uma média móvel aplicada a observações defasadas.
    - **Sazonalidade**: Captura padrões que se repetem em intervalos fixos.
    - **Variáveis Exógenas**: Incorpora fatores externos que podem influenciar a série.

    ## Por que SARIMAX para o Ibovespa?

    O Ibovespa, como principal indicador do mercado de ações brasileiro, é influenciado por uma miríade de fatores. Além de suas próprias flutuações históricas, ele é afetado por eventos econômicos, políticos e globais. A sazonalidade, como os padrões de fim de ano ou eventos recorrentes, também pode desempenhar um papel. O SARIMAX, com sua capacidade de considerar todas essas nuances, é uma escolha natural para uma análise aprofundada do Ibovespa.

    ## Embarque Nesta Jornada

    Convido você a se juntar a nós nesta exploração detalhada do Ibovespa através das lentes do SARIMAX. Vamos descobrir os segredos escondidos nos dados, desvendar padrões e, esperançosamente, lançar luz sobre o futuro deste índice crucial.
    """, unsafe_allow_html=True)
    
with tab2:
    st.markdown("""
    # A Arte da Escolha: Determinando o Melhor Modelo SARIMA

    A previsão de séries temporais é tanto uma arte quanto uma ciência. Enquanto a ciência nos fornece as ferramentas e técnicas, a arte está em escolher o modelo certo. No universo SARIMA, essa escolha é crucial e pode ser um pouco complexa.

    ## O Desafio da Escolha

    O SARIMA não é apenas um modelo, mas uma família de modelos. Cada combinação de parâmetros AR (p), I (d), MA (q), sazonal (P, D, Q) e o período de sazonalidade (s) pode resultar em um modelo diferente. A questão é: qual combinação é a melhor para nossos dados?

    ## Testando e Validando

    A resposta está em testar e validar. Aqui estão os passos que podemos seguir:

    1. **Estacionariedade**: Certifique-se de que a série temporal é estacionária. Se não for, diferencie-a até que seja.
    2. **ACF e PACF**: Use as funções de autocorrelação (ACF) e autocorrelação parcial (PACF) para ter uma ideia inicial dos parâmetros.
    3. **Grid Search**: Teste diferentes combinações de parâmetros e escolha a que oferece o melhor desempenho, geralmente medido pelo AIC (Critério de Informação de Akaike) ou BIC (Critério de Informação Bayesiano).
    4. **Validação Cruzada**: Use a validação cruzada para garantir que o modelo escolhido tenha um bom desempenho em diferentes conjuntos de dados.
    
    ## Vamos Começar!

    A seguir, vamos mergulhar nos dados e começar nossa busca pelo modelo SARIMA ideal para o Ibovespa. Acompanhe cada etapa, observe as métricas e junte-se a nós nesta jornada empolgante de descoberta.
    """)