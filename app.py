import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer

# Cargar el modelo y el DictVectorizer
with open('cryptocurrency_model.pkl', 'rb') as f:
    dv, model = pickle.load(f)

# Título de la aplicación
st.title("Predicción de árbol de decisiones de Crypotmonedas")

# Formulario para introducir datos de la criptomoneda
st.header("Introduce los datos de la Crypotmonedas:")

h1 = st.text_input("Cambio en 1 hora", value=0.0)
h24 = st.selectbox("Cambio en 24 hora", value=0.0)
d7 = st.selectbox("Cambio en 7 días",value=0.0)
d60 = st.selectbox("Cambio en 60 días",value=0.0)
d90 = st.selectbox("Cambio en 90 días", value=0.0)
ytd = st.number_input("Año hasta la fecha",value=0.0)
market_cap = st.selectbox("Capitalización bursátil", value=0.0)
volume_h24 = st.selectbox("Volumen en 24 horas", value=0.0)
volume_change_h24 = st.selectbox("Cambio de volumen en 24 horas", value=0.0)
volume_change_d30 = st.selectbox("Cambio de volumen en 30 días", value=0.0)
circulating  = st.selectbox("Circulando", value=0.0)
total_supply = st.selectbox("Suplemento total",value=0.0)
max_supply = st.selectbox("Máximo suplemento", value=0.0)
num_market_pairs  = st.selectbox("número de pares de mercado", value=0.0)

# Botón de predicción
if st.button("Predecir"):
    # Crear un diccionario con los datos de la cryptomoneda
    crypto_data = {
        "1h %": h1,
        "24h %": h24,
        "7d %": d7,
        "60d %": d60,
        "90d %": d90,
        "YTD %": ytd,
        "Market Cap": market_cap,
        "Volume (24h)": volume_h24,
        "Volume Change (24h)": volume_change_h24,
        "Volume Change (30d)": volume_change_d30,
        "Circulating ": circulating,
        "Total Supply": total_supply,
        "Max Supply": max_supply,
        "Num Market Pairs": num_market_pairs,
    }

    # Transformar los datos del cliente
    X_crypto = dv.transform([crypto_data])

    # Realizar la predicción
    y_pred_proba = model.predict_proba(X_crypto)[0][1]  # Probabilidad de churn

   # Mostrar resultado
    st.subheader("Resultado de la Predicción:")
    if y_pred_proba > 0.5:
        st.success(f"Se predice un **aumento** en el precio de la criptomoneda con una probabilidad de {y_pred_proba:.2f}")
    else:
        st.error(f"Se predice una **disminución** en el precio de la criptomoneda con una probabilidad de {y_pred_proba:.2f}")

    # Visualización (opcional)
    st.progress(y_pred_proba)
