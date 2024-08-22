import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris

#Page configuration
st.set_page_config (
  page_title = 'Iris prediccion',
  layout = 'wide',
  initial_sidebar_state = 'expanded'
)

# Titulo de la app
st.title ('Prediccion clase de orquidea')

# Lectura de datos
iris = load_iris ()
df = pd.DataFrame (iris.data, columns = iris.feature_names)

st.write (df)
