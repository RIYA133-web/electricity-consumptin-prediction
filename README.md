# ⚡ Electricity Consumption Prediction using ANN

A time series forecasting project that predicts next-day household electricity consumption using an Artificial Neural Network (ANN), built with TensorFlow/Keras and deployed as an interactive Streamlit web app.

## 📌 Overview

This project uses the UCI "Household Power Consumption" dataset (minute-wise readings, 2006–2010) to forecast daily electricity usage. Raw sensor data was cleaned, resampled to daily frequency, and transformed into lag-based features (previous 1, 2, 3, and 7-day consumption) to capture short-term trend and weekly seasonality — allowing a simple feed-forward ANN to model time-dependent patterns.

## 🎯 Results

- **Test MAE:** ~0.19 kW
- Model generalizes well with no overfitting (training/validation loss converge closely)
- Captures overall seasonal trend accurately; slightly under-predicts sharp, short-duration spikes

## 🛠️ Tech Stack

- **Python, Pandas, NumPy** – data cleaning & feature engineering
- **TensorFlow/Keras** – ANN model
- **Scikit-learn** – Min-Max scaling
- **Matplotlib** – exploratory data analysis
- **Streamlit** – web app deployment

## 📊 Dataset

[UCI Household Power Consumption Dataset](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set) (not included in this repo due to size — download separately and place in the project folder as `household_power_consumption.txt`)

## 🚀 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📂 Project Structure
├── electricity_notebook.ipynb # Full analysis: EDA, feature engineering, model training
├── app.py # Streamlit web app
├── electricity_model.h5 # Trained ANN model
├── scaler.pkl # Fitted MinMaxScaler
├── requirements.txt
└── README.md

## 🔮 Future Improvements

- LSTM/GRU architectures for richer temporal modeling
- Incorporate weather data as external features
- Multi-step forecasting (predict several days ahead)
- Deploy to Streamlit Community Cloud

---
Built by Riya Rai as part of summer training in Data Science & AI.