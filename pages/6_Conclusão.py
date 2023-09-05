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

Sem mais delongas, vamos revelar o modelo vencedor. O modelo que, em nossa análise, provou ser o mais confiável e preciso em suas previsões. Observando as métricas de acurácia, podemos concluir que o modelo tem um desempenho razoável. O MAPE de 7,788% indica que, em média, o modelo erra em cerca de 7,78% nas suas previsões. Em comparação, o ARIMA acerta em 2% somente, tornando nosso modelo escolhido superior em termos de precisão.
            
""")

table_html = conclusao.to_html(index=False)

st.write(table_html, unsafe_allow_html=True)

st.markdown("""
## O Caminho à Frente
            
Embora tenhamos identificado o modelo mais preciso para nossos dados atuais, a jornada da análise de séries temporais é contínua. Os dados evoluem, as tendências mudam e novos desafios surgem. Estaremos sempre prontos para reavaliar, reajustar e buscar a excelência em nossas previsões.

Obrigado por nos acompanhar nesta jornada. Esperamos que as insights e previsões geradas sirvam bem ao seu propósito e ajudem a moldar um futuro brilhante.
            
""")

