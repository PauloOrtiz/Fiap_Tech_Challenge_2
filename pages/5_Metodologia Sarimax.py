import streamlit as st
from PIL import Image


st.set_page_config(page_title="Sarimax", page_icon=":house:")

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