import streamlit as st
import joblib
import numpy as np

# Cargar el modelo y el escalador
model = joblib.load("decision_tree_model.pkl")
scaler = joblib.load("scaler.pkl")

# Configuración de la página
st.set_page_config(page_title="Predicción con Árbol de Decisión", layout="centered")

# Título
st.title("Predicción de Radius Mean")
st.write("Proporciona los valores de las características para realizar una predicción.")

# Crear entradas para las características
feature_labels = [f"Característica {i+1}" for i in range(29)]
input_values = []

for label in feature_labels:
    value = st.number_input(label, value=0.0, format="%.4f")
    input_values.append(value)

# Botón para realizar la predicción
if st.button("Realizar Predicción"):
    try:
        # Convertir los valores ingresados en un array numpy
        input_array = np.array(input_values).reshape(1, -1)
        # Escalar los datos
        scaled_input = scaler.transform(input_array)
        # Realizar la predicción
        prediction = model.predict(scaled_input)
        # Mostrar el resultado
        st.success(f"Predicción: {prediction[0]:.2f}")
    except Exception as e:
        st.error(f"Error en la predicción: {str(e)}")
