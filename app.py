import streamlit as st
import pickle
import numpy as np
st.markdown("<h1 style='text-align: center;'>Institute of Technology of Cambodia</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>Department of AMS</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>Machine Learning by Team-3</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>🚗 Auto MPG Prediction 📊</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>This is a simple app to predict the MPG of a car</p>", unsafe_allow_html=True)


# Load the model
with open("model.pkl", "rb") as f:
    theta = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    mean, max_, min_ = pickle.load(f)

# Input form
cylinders = st.number_input("🔘 Cylinders", min_value=3, max_value=8, value=4, step=1)
displacement = st.number_input("📏 Displacement", min_value=68, max_value=455, value=200, step=1)
horsepower = st.number_input("🐎 Horsepower", min_value=46, max_value=230, value=100, step=1)
weight = st.number_input("⚖️ Weight", min_value=1613, max_value=5140, value=3000, step=1)
acceleration = st.number_input("🚀 Acceleration", min_value=8.0, max_value=24.0, value=15.0, step=0.1)
model_year = st.number_input("📅 Model Year", min_value=70, max_value=82, value=76, step=1)
origin = st.number_input("🌍 Origin", min_value=1, max_value=3, value=1, step=1)

if st.button("Predict 🎯"):
    X_test = np.array([displacement, cylinders, horsepower, weight, acceleration, model_year, origin]).reshape(1, -1)
    X_test_scaled = (X_test - mean) / (max_ - min_)
    X_test_new = np.concatenate((np.ones((1, 1)), X_test_scaled), axis=1)
    y_predict = np.matmul(X_test_new, theta)
    st.write(f"Predicted MPG: {y_predict[0][0]:0.2f}")
    st.write("Made by: HENG Seaklong")

