import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ===== Model aur Scaler load karo (ek baar, app start hote hi) =====
model = load_model('electricity_model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# ===== Page Title =====
st.title("⚡ Electricity Consumption Predictor")
st.write("ANN model use karke, pichle 4 dino ke consumption se agla din predict karo")

# ===== Input Fields =====
st.subheader("Pichle dino ka consumption enter karo (kW mein):")

lag1 = st.number_input("Kal ka consumption", min_value=0.0, max_value=10.0, value=1.5, step=0.1)
lag2 = st.number_input("Parso ka consumption", min_value=0.0, max_value=10.0, value=1.5, step=0.1)
lag3 = st.number_input("3 din pehle ka consumption", min_value=0.0, max_value=10.0, value=1.5, step=0.1)
lag7 = st.number_input("1 hafta pehle ka consumption", min_value=0.0, max_value=10.0, value=1.5, step=0.1)

# ===== Predict Button (SIRF EK BAAR) =====
if st.button("Predict"):
    input_data = pd.DataFrame([[lag1, lag2, lag3, lag7]], 
                                columns=['Lag_1', 'Lag_2', 'Lag_3', 'Lag_7'])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled, verbose=0)
    pred_value = prediction[0][0]
    
    st.success(f"Predicted consumption for tomorrow: **{pred_value:.2f} kW**")
    
    # ===== Chart banana =====
    st.subheader("📊 Consumption Trend")
    
    days = ['7 days ago', '3 days ago', '2 days ago', 'Yesterday', 'Tomorrow (Predicted)']
    values = [lag7, lag3, lag2, lag1, pred_value]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(days[:-1], values[:-1], marker='o', color='#1B2A4A', linewidth=2, label='Actual (entered)')
    ax.plot(days[-2:], values[-2:], marker='o', color='#D64550', linewidth=2, linestyle='--', label='Predicted')
    ax.set_ylabel('Consumption (kW)')
    ax.legend()
    plt.xticks(rotation=15)
    
    st.pyplot(fig)