import streamlit as st
import joblib
import pandas as pd

# Load the trained model
try:
    model = joblib.load('logi.sav')
except FileNotFoundError:
    st.error("Model file 'logi.sav' not found. Please ensure the model is saved in the same directory.")
    st.stop()

# Feature names used during model training
feature_names = [
    'Delivery_Distance',
    'Traffic_Congestion',
    'Weather_Condition',
    'Delivery_Slot',
    'Driver_Experience',
    'Num_Stops',
    'Vehicle_Age',
    'Road_Condition_Score',
    'Package_Weight',
    'Fuel_Efficiency',
    'Warehouse_Processing_Time'
]

st.title('Delivery Delay Prediction App')
st.write('Enter the details below to predict if a delivery will be delayed.')

# Create input fields
input_data = {}

for feature in feature_names:

    if feature == 'Traffic_Congestion':
        input_data[feature] = st.slider(
            f'{feature} (1=Low, 5=High)', 1, 5, 3
        )

    elif feature == 'Weather_Condition':
        input_data[feature] = st.slider(
            f'{feature} (1=Good, 5=Bad)', 1, 5, 3
        )

    elif feature == 'Delivery_Slot':
        input_data[feature] = st.slider(
            f'{feature} (1, 2, 3)', 1, 3, 2
        )

    elif feature == 'Road_Condition_Score':
        input_data[feature] = st.slider(
            f'{feature} (1=Bad, 5=Good)', 1, 5, 3
        )

    elif feature == 'Driver_Experience':
        input_data[feature] = st.slider(
            f'{feature} (Years)', 0, 30, 5
        )

    elif feature == 'Num_Stops':
        input_data[feature] = st.slider(
            f'{feature}', 1, 10, 3
        )

    elif feature == 'Vehicle_Age':
        input_data[feature] = st.slider(
            f'{feature} (Years)', 0, 15, 3
        )

    elif feature == 'Warehouse_Processing_Time':
        input_data[feature] = st.slider(
            f'{feature} (Minutes)', 10, 120, 60
        )

    else:
        input_data[feature] = st.number_input(
            f'Enter {feature}', value=0.0
        )

# Convert input to DataFrame
input_df = pd.DataFrame([input_data])

# Prediction
if st.button('Predict Delivery Delay'):

    try:
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)[:, 1]

        st.subheader('Prediction Result:')

        if prediction[0] == 1:
            st.error(
                f"The delivery is predicted to be **DELAYED** "
                f"(Probability: {prediction_proba[0]:.2f})"
            )
        else:
            st.success(
                f"The delivery is predicted to be **ON TIME** "
                f"(Probability: {prediction_proba[0]:.2f})"
            )

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
