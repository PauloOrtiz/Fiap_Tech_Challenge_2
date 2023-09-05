import streamlit as st
from PIL import Image
import pandas as pd



st.set_page_config(page_title="Conclusão", page_icon=":house:")

conclusao = pd.read_csv('./src/data/Tabela_Metricas_Conclusao.csv', sep=';')

image = Image.open("./src/img/Conclusao.png")
st.image(image)

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
            
# Conclusão: A Revelação Final
            
Após uma jornada profunda pelo universo da análise de séries temporais, chegamos ao momento crucial. Atravessamos mares de dados, navegamos por modelos complexos e enfrentamos os desafios da previsão. Agora, é hora de revelar o campeão, o modelo que se destacou acima de todos os outros em nossa busca pela previsão mais precisa.

A precisão é a pedra angular de qualquer modelo de previsão. Ela determina a confiabilidade das previsões e, em muitos casos, pode ter implicações significativas em decisões de negócios, estratégias e planejamento.

## O Critério: A Acurácia

Avaliamos cada modelo com base em sua acurácia. Queríamos um modelo que não apenas capturasse as tendências e padrões nos dados, mas que também pudesse prever o futuro com a maior precisão possível. Cada modelo foi testado, ajustado e avaliado meticulosamente.            

## A Revelação

            
""")


st.write("""
Ao embarcar nesta jornada de análise e previsão, navegamos por mares de dados, enfrentamos tempestades de incertezas e exploramos diferentes horizontes modelados por ARIMA, SARIMA e Prophet. Cada modelo com sua própria essência, cada um com sua promessa de precisão. Mas qual deles realmente se destacou? Qual deles nos mostrou o caminho mais claro para o futuro?
""")

st.write("""
Ao olhar para o nosso farol de orientação - as métricas de acurácia - vemos uma história se desenrolar.
""")

table_html = conclusao.to_html(index=False)

st.write(table_html, unsafe_allow_html=True)

st.write("""
O modelo ARIMA(0,1,0) brilha intensamente com um MAPE de apenas 1,35%. Isso nos diz que, em sua jornada através do tempo, ele desviou de seu curso verdadeiro por apenas 1,35%. Em comparação, o SARIMA(0,1,0)(1,0,1,12) e o Prophet, embora valentes em suas tentativas, desviaram-se por 10,16% e 7,79% respectivamente.
""")

st.write("""
Assim, com o vento a nosso favor e as estrelas alinhadas, declaramos o ARIMA(0,1,0) como nosso fiel guia nesta jornada de previsão. Ele não apenas nos mostrou o caminho, mas o fez com a maior precisão entre todos os competidores.
""")

st.write("""
Agradecemos a todos os modelos por sua valiosa contribuição e a você, nosso estimado leitor, por nos acompanhar nesta incrível viagem. Que as previsões do ARIMA(0,1,0) iluminem nosso caminho adiante!
""")


st.markdown("""
## O Caminho à Frente
            
Embora tenhamos identificado o modelo mais preciso para nossos dados atuais, a jornada da análise de séries temporais é contínua. Os dados evoluem, as tendências mudam e novos desafios surgem. Estaremos sempre prontos para reavaliar, reajustar e buscar a excelência em nossas previsões.

Obrigado por nos acompanhar nesta jornada. Esperamos que as insights e previsões geradas sirvam bem ao seu propósito e ajudem a moldar um futuro brilhante.
            
""")

