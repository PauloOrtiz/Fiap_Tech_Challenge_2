import streamlit as st
from PIL import Image
import pandas as pd


st.set_page_config(page_title="Prophet", page_icon=":house:")

image = Image.open("./src/img/ibovespa.jpg")
st.image(image)