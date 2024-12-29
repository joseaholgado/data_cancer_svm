import streamlit as st
import joblib
import numpy as np

# Cargar el modelo y el escalador decision tree
# model = joblib.load("decision_tree_model.pkl")
# scaler = joblib.load("scaler.pkl")

# Cargar el modelo y el escalador decision tree
model = joblib.load("svm_model.pkl")
scaler = joblib.load("svm_scaler.pkl")

# Configuración de la página
st.set_page_config(page_title="Predicción de Radius Mean", layout="centered")

# Título de la aplicación para decision tree
# st.title("Predicción de Radius Mean")
# st.write("""
#    Este proyecto utiliza un modelo de Árbol de Decisión para predecir el radio promedio (`radius_mean`) 
#    basado en características celulares. Introduce los valores en los campos correspondientes para obtener una predicción.
# """)

# Título de la aplicación para svm
st.title("Predicción de Radius Mean")
st.write("""
    Este proyecto utiliza un modelo de SVM para predecir el radio promedio (`radius_mean`) 
    basado en características celulares. Introduce los valores en los campos correspondientes para obtener una predicción.
""")

# Crear entradas dinámicas para las características
st.header("Introduce las características")

# Nombres de las columnas
feature_names = [
    'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se',
    'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
    'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
    'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst',
    'fractal_dimension_worst'
]

default_values = [
    17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4,
    0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119,
    0.2654, 0.4601
]

input_values = []

for feature, default in zip(feature_names, default_values):
    value = st.number_input(f"{feature.replace('_', ' ').capitalize()}", value=default, format="%.4f")
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
        st.success(f"Predicción de Radius Mean: {prediction[0]:.2f}")
    except Exception as e:
        st.error(f"Error en la predicción: {str(e)}")
