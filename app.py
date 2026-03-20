import streamlit as st
import pandas as pd
import joblib
import numpy as np


MODEL_FILE = 'fraud_model.pkl'
SCALER_FILE = 'scaler.pkl'


try:
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
except FileNotFoundError:
    st.error(f"Error: Model files '{MODEL_FILE}' or '{SCALER_FILE}' not found. Please run 'train_model.py' first.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()


st.title("💳 Credit Card Fraud Detection")
st.markdown("Enter the transaction details to predict if it is fraudulent.")

input_cols = [f'V{i}' for i in range(1, 29)] + ['Amount']
input_data = {}

cols = st.columns(3)
col_index = 0

st.subheader("Transaction Features (V1-V28 are PCA-transformed)")
for col in input_cols:
    
    if col != 'Amount':
        with cols[col_index % 3]:
            input_data[col] = st.number_input(f'{col}', value=0.0, format="%.6f", key=col)
        col_index += 1

st.subheader("Transaction Amount")
input_data['Amount'] = st.number_input('Amount (Original Currency)', value=100.00, format="%.2f", key='Amount_Input')

if st.button('Predict Transaction Status'):
    
    input_df = pd.DataFrame([input_data])
    
    input_df['Amount'] = scaler.transform(input_df['Amount'].values.reshape(-1, 1))
    
    
    X_predict = input_df[[f'V{i}' for i in range(1, 29)] + ['Amount']]
    
    
    prediction = model.predict(X_predict)
    prediction_proba = model.predict_proba(X_predict)


    st.subheader("Prediction Result:")
    
    if prediction[0] == 1:
        st.error(f"🚨 FRAUDULENT TRANSACTION DETECTED")
        st.write(f"Confidence (Fraud): **{prediction_proba[0][1]:.2f}**")
    else:
        st.success(f"✅ LEGITIMATE TRANSACTION")
        st.write(f"Confidence (Legitimate): **{prediction_proba[0][0]:.2f}**")