# app.py
import streamlit as st
import numpy as np
import joblib

# ---------------------------
# Load trained model & scaler
# ---------------------------
model = joblib.load('model/wine_cultivar_model.pkl')
scaler = joblib.load('model/scaler.pkl')

# ---------------------------
# Title & description
# ---------------------------
st.title("🍷 Wine Cultivar Prediction System")
st.write("""
Enter the chemical properties of the wine sample below to predict its cultivar (origin/class). 
All fields are required.
""")

# ---------------------------
# User input fields
# ---------------------------
def user_input():
    alcohol = st.text_input("Alcohol")
    malic_acid = st.text_input("Malic Acid")
    ash = st.text_input("Ash")
    alcalinity_of_ash = st.text_input("Alcalinity of Ash")
    magnesium = st.text_input("Magnesium")
    flavanoids = st.text_input("Flavanoids")
    return [alcohol, malic_acid, ash, alcalinity_of_ash, magnesium, flavanoids]

inputs = user_input()

# ---------------------------
# Predict button
# ---------------------------
if st.button("Predict"):
    # Check if any input is empty
    if "" in inputs:
        st.warning("⚠️ Please fill in all fields to make a prediction!")
    else:
        try:
            # Convert inputs to float
            features = np.array([[float(i) for i in inputs]])
            
            # Scale features using same scaler as training
            features_scaled = scaler.transform(features)
            
            # Make prediction
            prediction = model.predict(features_scaled)[0]
            
            # Display result (Cultivar 1, 2, or 3)
            st.success(f"✅ The predicted wine cultivar is: Cultivar {prediction + 1}")
        
        except ValueError:
            st.error("⚠️ Please enter valid numeric values for all features.")
