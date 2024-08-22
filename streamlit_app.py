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

min_sepal_length = df['sepal length (cm)'].min()
max_sepal_length = df['sepal length (cm)'].max()
mean_sepal_length = df['sepal length (cm)'].mean()

min_sepal_width = df['sepal width (cm)'].min()
max_sepal_width = df['sepal width (cm)'].max()
mean_sepal_width = df['sepal width (cm)'].mean()

min_petal_length = df['petal length (cm)'].min()
max_petal_length = df['petal length (cm)'].max()
mean_petal_length = df['petal length (cm)'].mean()

min_petal_width = df['petal width (cm)'].min()
max_petal_width = df['petal width (cm)'].max()
mean_petal_width = df['petal width (cm)'].mean()

st.sidebar.header('Parametros de entrada')

sepal_length = st.sidebar.slider('Sepal Length (cm)', min_value=min_sepal_length, max_value=max_sepal_length, value=mean_sepal_length)
sepal_width = st.sidebar.slider('Sepal Width (cm)', min_value=min_sepal_width, max_value=max_sepal_width, value=mean_sepal_width)
petal_length = st.sidebar.slider('Petal Length (cm)', min_value=min_petal_length, max_value=max_petal_length, value=mean_petal_length)
petal_width = st.sidebar.slider('Petal Width (cm)', min_value=min_petal_width, max_value=max_petal_width, value=mean_petal_width)

from sklearn.model_selection import train_test_split

df ['target']=iris.target

X_train, X_test, y_train, y_test = train_test_split (
    df.drop (['target'], axis='columns'), iris.target,
    test_size = 0.2
)

from sklearn.ensemble import RandomForestClassifier

st.subheader('Predicciones')
model = RandomForestClassifier ()
model.fit (X_train, y_train)
st.write(model.score (X_test, y_test))

st.subheader('Analisis Exploratorio de Datos')
groupby_species_mean = df.groupby ('target').mean ()
st.write (groupby_species_mean)

y_predicted = model.predict (X_test)

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.hist(groupby_species_mean, bins=5)
ax.set_xlabel('Valor')
ax.set_ylabel('Frecuencia')
ax.set_title('Histograma')
st.pyplot(fig)

st.image("imagen.jpg", caption="Machine Learning")
