import streamlit as st
import pickle
import numpy as np
import pandas as pd
import base64



# Define expected feature names
features = ['Age', 'ALT', 'BMI', 'DM.IFG', 'FBG', 'GGT', 'TG', 'AST.PLT']

# Load model and scaler
with open("fibrosis.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Streamlit UI
st.title("A Machine Learning Based Approach for Liver Fibrosis Diagnosis in NAFLD Using Biomarkers and Demographics")

st.markdown("### Enter patient details below:")

# Collect user input for all 8 features without +/- buttons
input_data = []

all_empty = True  # Track if all inputs are empty
for feature in features:
    value = st.text_input(f"Enter {feature}:", value="")  # Text input instead of number input
    try:
        value = float(value) if value.strip() else 0.0  # Convert input to float, default to 0.0 if empty
    except ValueError:
        st.error(f"Invalid input for {feature}. Please enter a numeric value.")
        st.stop()

    if value is not None:
        all_empty = False  # If at least one value is entered, set flag to False

    input_data.append(value if value is not None else 0.0)  # Default empty values to 0.0


# Predict button
if st.button("Predict"):
    if all_empty:
        st.warning("Please enter values before making a prediction.")
    else:
        try:
            # Convert input to DataFrame
            input_df = pd.DataFrame([input_data], columns=features)
            
            # Scale input
            scaled_input = scaler.transform(input_df)

            # Predict
            prediction = model.predict(scaled_input)
            
            # Mapping 0 -> Early Fibrosis, 1 -> Advanced Fibrosis
            fibrosis_stage = "Early Fibrosis" if prediction == 0 else "Advanced Fibrosis"
            
            # Display result
            st.success(f"Predicted Fibrosis Stage: {fibrosis_stage}")
        except Exception as e:
            st.error(f"Error: {e}")
