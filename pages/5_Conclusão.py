import streamlit as st
from PIL import Image

st.set_page_config(page_title="Conclusão", page_icon=":house:")

image = Image.open("./src/img/ibovespa.jpg")
st.image(image)




