import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px


st.title("Student Dropout Prediction App")
st.sidebar.header("User  Input Features")


model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("encoder.pkl")

df = pd.read_csv("data_clean.csv")
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

input_data = {}

for col in categorical_columns:
    options = df[col].dropna().unique().tolist()
    input_data[col] = st.sidebar.selectbox(f"{col}", options)

for col in numerical_columns:
    min_val, max_val = float(df[col].min()), float(df[col].max())
    input_data[col] = st.sidebar.slider(f"{col}", min_value=min_val, max_value=max_val, value=min_val)


input_df = pd.DataFrame([input_data])


for col in categorical_columns:
    if col in label_encoders:
        input_df[col] = label_encoders[col].transform(input_df[col])

missing_cols = [col for col in numerical_columns if col not in input_df.columns]
for col in missing_cols:
    input_df[col] = 0 

input_df = input_df[numerical_columns]

try:
    input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])
except ValueError as e:
    st.error(f"Error in scaling: {e}")
    st.stop()

if st.sidebar.button("Predict"):
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]
    st.write(f"## Prediction: {'Dropout' if prediction == 1 else 'Continue'}")
    st.write(f"### Confidence: {max(prediction_proba) * 100:.2f}%")

st.subheader("Dataset Overview")
st.write(df.head())
fig = px.histogram(df, x='Status', title='Distribusi Status Mahasiswa')
st.plotly_chart(fig)