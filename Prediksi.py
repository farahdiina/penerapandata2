import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Judul aplikasi
st.title("Aplikasi Prediksi Dropout Mahasiswa")
st.sidebar.header("Input Fitur")

# Memuat model dan komponen preprocessing
model = joblib.load("best_model.pkl")  # Memuat model terbaik
scaler = joblib.load("scaler.pkl")      # Memuat scaler
label_encoders = joblib.load("encoder.pkl")  # Memuat encoder untuk fitur kategorikal

# Memuat dataset untuk mendapatkan informasi fitur
df = pd.read_csv("data_clean.csv")  # Sesuaikan path jika perlu
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()  # Kolom kategorikal
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()  # Kolom numerikal

# Membuat dictionary untuk menyimpan input pengguna
input_data = {}

# Input sidebar untuk fitur kategorikal
for col in categorical_columns:
    options = df[col].dropna().unique().tolist()  # Mendapatkan opsi unik dari kolom
    input_data[col] = st.sidebar.selectbox(f"{col}", options)  # Menampilkan selectbox

# Input sidebar untuk fitur numerikal
for col in numerical_columns:
    min_val, max_val = float(df[col].min()), float(df[col].max())  # Mendapatkan nilai minimum dan maksimum
    input_data[col] = st.sidebar.slider(f"{col}", min_value=min_val, max_value=max_val, value=min_val)  # Menampilkan slider

# Mengonversi data input menjadi DataFrame
input_df = pd.DataFrame([input_data])

# Mengkode fitur kategorikal
for col in categorical_columns:
    if col in label_encoders:
        input_df[col] = label_encoders[col].transform(input_df[col])  # Mengkode fitur kategorikal

# Memastikan semua kolom numerikal ada
for col in numerical_columns:
    if col not in input_df.columns:
        input_df[col] = 0  # Mengisi kolom yang hilang dengan nilai default

# Memastikan urutan kolom sesuai dengan data pelatihan
input_df = input_df[numerical_columns]

# Melakukan scaling pada fitur numerikal
try:
    input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])  # Melakukan scaling
except ValueError as e:
    st.error(f"Kesalahan dalam scaling: {e}")  # Menampilkan pesan kesalahan jika terjadi
    st.stop()

# Tombol prediksi
if st.sidebar.button("Prediksi"):
    prediction = model.predict(input_df)[0]  # Melakukan prediksi
    prediction_proba = model.predict_proba(input_df)[0]  # Mendapatkan probabilitas prediksi
    st.write(f"## Prediksi: {'Dropout' if prediction == 1 else 'Lanjut'}")  # Menampilkan hasil prediksi
    st.write(f"### Keyakinan: {max(prediction_proba) * 100:.2f}%")  # Menampilkan tingkat keyakinan

# Menampilkan gambaran umum dataset
st.subheader("Gambaran Umum Dataset")
st.write(df.head())  # Menampilkan 5 baris pertama dari dataset

# Memvisualisasikan distribusi variabel target
fig = px.histogram(df, x='Status', title='Distribusi Status Mahasiswa')  # Membuat histogram
st.plotly_chart(fig)  # Menampilkan histogram
