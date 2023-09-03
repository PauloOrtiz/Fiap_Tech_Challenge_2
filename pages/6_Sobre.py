import streamlit as st
from PIL import Image

st.set_page_config(page_title="Sobre", page_icon=":house:")

image = Image.open("./src/img/Fiap.png")
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
    # Sobre o Projeto
    
    Este projeto foi desenvolvido como parte do 2º Tech Challenge da turma de Data Science da FIAP. Foi uma oportunidade incrível para aprimorar nossas habilidades em análise de dados e colaboração em equipe, e gostaríamos de expressar nossa gratidão a todos que participaram deste desafio.

    ## Equipe do Projeto 
    
    - Amanda Bueno de Oliveira
    - João Guilherme Simões
    - Leonardo Fernandes de Moraes Alves
    - Luiz Antonio Simette de Mello Campos
    - Paulo Henrique Barbosa Ortiz de Souza
            

    ## Orientadores 
    
    Este projeto não teria sido possível sem a orientação de nossos professores:

    - Edgard Joseph Kiriyama
    - Matheus Pavani

    Eles nos forneceram a estrutura necessária para concluir este desafio e estamos profundamente gratos pelo tempo e esforço que dedicaram à nossa aprendizagem.

    ##  Agradecimentos

    Também gostaríamos de agradecer aos nossos colegas de classe e a todos os envolvidos na organização deste desafio.
        
    ## Referências 📚

     FOLHA DE S.PAULO. **Crise de energia derruba ações**. 2001. Disponível em: [link](https://www1.folha.uol.com.br/fsp/invest/in1405200121.htm). Acesso em: [data de acesso].
  
    - INFOMONEY. **Primeiro semestre de 2001 é caracterizado pela desvalorização cambial e crise energética**. Disponível em: [link](https://www.infomoney.com.br/mercados/primeiro-semestre-de-2001-e-caracterizado-pela-desvalorizacao-cambial-e-crise-energetica/). Acesso em: [data de acesso].

    - INFOMONEY. **11 de setembro de 2001: Wall Street antecipou a queda das torres gêmeas**. Disponível em: [link](https://www.infomoney.com.br/mercados/11-de-setembro-de-2001-wall-street-antecipou-a-queda-das-torres-gemeas/). Acesso em: [data de acesso].

    - SUNO RESEARCH. **Há 20 anos, bolsas mundiais mergulhavam no caos após ataque às torres gêmeas**. 2021. Disponível em: [link](https://suno.com.br/noticias/11-de-setembro-terror-mercados-bolsas/). Acesso em: [data de acesso].

    - INFOMONEY. **Governo Lula: estabilidade econômica de 2003 vai levar ao crescimento**. Disponível em: [link](https://www.infomoney.com.br/mercados/governo-lula-estabilidade-economica-de-2003-vai-levar-ao-crescimento/). Acesso em: [data de acesso].

    - AGÊNCIA DE NOTÍCIAS IBGE. **PIB cresceu 3,2% e foi de R$ 2,1 trilhões em 2005**. Disponível em: [link](https://agenciadenoticias.ibge.gov.br/agencia-sala-de-imprensa/2013-agencia-de-noticias/releases/13389-asi-pib-cresceu-32-e-foi-de-r-21-trilhoes-em-2005). Acesso em: [data de acesso].

    - BANCO CENTRAL DO BRASIL. **Relatório Anual 2006**. Disponível em: [link](https://www.bcb.gov.br/pec/boletim/banual2006/rel2006introdp.pdf). Acesso em: [data de acesso].

    - AGÊNCIA DE NOTÍCIAS IBGE. **PIB cresceu 4,0% e foi de R$ 2,37 trilhões em 2006**. Disponível em: [link](https://agenciadenoticias.ibge.gov.br/agencia-sala-de-imprensa/2013-agencia-de-noticias/releases/13565-asi-pib-cresceu-40-e-foi-de-r-237-trilhoes-em-2006). Acesso em: [data de acesso].

    - G1. **Entenda como crise de 2008 influenciou vida dos brasileiros**. Disponível em: [link](https://g1.globo.com/economia/seu-dinheiro/noticia/2011/09/entenda-como-crise-de-2008-influenciou-vida-dos-brasileiros.html). Acesso em: [data de acesso].

    - ITR UFRRJ. **A economia brasileira em 2007**. Disponível em: [link](https://itr.ufrrj.br/portal/wp-content/uploads/2017/10/t74.pdf). Acesso em: [data de acesso].

    - BANCO CENTRAL DO BRASIL. **Relatório Anual 2009**. Disponível em: [link](https://www.bcb.gov.br/pec/boletim/banual2009/rel2009cap1p.pdf). Acesso em: [data de acesso].

    - G1. **Economia brasileira cresce 7,5% em 2010, mostra IBGE**. Disponível em: [link](https://g1.globo.com/economia/noticia/2011/03/economia-brasileira-cresce-75-em-2010-mostra-ibge.html). Acesso em: [data de acesso].

        
    Gostaríamos de expressar nossa gratidão a todas estas fontes por disponibilizar esses dados publicamente.
    """, unsafe_allow_html=True)
