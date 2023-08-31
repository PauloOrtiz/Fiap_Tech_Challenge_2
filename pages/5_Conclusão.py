import streamlit as st
from PIL import Image

st.set_page_config(page_title="Conclusão", page_icon=":house:")

image = Image.open("./src/img/ibovespa.jpg")
st.image(image)


st.title("")

st.markdown("""
            
# Conclusão: A Revelação Final
            
Após uma jornada profunda pelo universo da análise de séries temporais, chegamos ao momento crucial. Atravessamos mares de dados, navegamos por modelos complexos e enfrentamos os desafios da previsão. Agora, é hora de revelar o campeão, o modelo que se destacou acima de todos os outros em nossa busca pela previsão mais precisa.

A precisão é a pedra angular de qualquer modelo de previsão. Ela determina a confiabilidade das previsões e, em muitos casos, pode ter implicações significativas em decisões de negócios, estratégias e planejamento.

## O Critério: A Acurácia

Avaliamos cada modelo com base em sua acurácia. Queríamos um modelo que não apenas capturasse as tendências e padrões nos dados, mas que também pudesse prever o futuro com a maior precisão possível. Cada modelo foi testado, ajustado e avaliado meticulosamente.            

## A Revelação

Sem mais delongas, vamos revelar o modelo vencedor. O modelo que, em nossa análise, provou ser o mais confiável e preciso em suas previsões. O modelo que recomendamos para prever o futuro da série temporal em questão é...
            
""")

st.markdown("""
### Resultado em desenvolvimento - EM BREVE
""")


st.markdown("""
## O Caminho à Frente
            
Embora tenhamos identificado o modelo mais preciso para nossos dados atuais, a jornada da análise de séries temporais é contínua. Os dados evoluem, as tendências mudam e novos desafios surgem. Estaremos sempre prontos para reavaliar, reajustar e buscar a excelência em nossas previsões.

Obrigado por nos acompanhar nesta jornada. Esperamos que as insights e previsões geradas sirvam bem ao seu propósito e ajudem a moldar um futuro brilhante.
            
""")

