import streamlit as st
import joblib
import pandas as pd

# Load the trained model and scaler
logi_scaled = joblib.load('logi_scaled.sav')
scaler = joblib.load('scaler.sav')

# Define the feature names (same order as during training)
feature_names = [
    'Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition',
    'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age',
    'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency',
    'Warehouse_Processing_Time'
]

st.title('Delivery Delay Prediction App')
st.write('Enter the details below to predict if a delivery will be delayed.')

# Create input fields for each feature
Delivery_Distance = st.slider('Delivery Distance (km)', 0.0, 50.0, 25.0)
Traffic_Congestion = st.slider('Traffic Congestion (1-5)', 1, 5, 3)
Weather_Condition = st.slider('Weather Condition (1-5)', 1, 5, 3)
Delivery_Slot = st.slider('Delivery Slot (1-3)', 1, 3, 2)
Driver_Experience = st.slider('Driver Experience (years)', 0, 20, 10)
Num_Stops = st.slider('Number of Stops', 1, 10, 5)
Vehicle_Age = st.slider('Vehicle Age (years)', 0, 15, 7)
Road_Condition_Score = st.slider('Road Condition Score (1-5)', 1, 5, 3)
Package_Weight = st.slider('Package Weight (kg)', 0.0, 50.0, 25.0)
Fuel_Efficiency = st.slider('Fuel Efficiency (km/l)', 0.0, 20.0, 10.0)
Warehouse_Processing_Time = st.slider('Warehouse Processing Time (minutes)', 0, 120, 60)

# Create a DataFrame from user inputs
input_data = pd.DataFrame([{
    'Delivery_Distance': Delivery_Distance,
    'Traffic_Congestion': Traffic_Congestion,
    'Weather_Condition': Weather_Condition,
    'Delivery_Slot': Delivery_Slot,
    'Driver_Experience': Driver_Experience,
    'Num_Stops': Num_Stops,
    'Vehicle_Age': Vehicle_Age,
    'Road_Condition_Score': Road_Condition_Score,
    'Package_Weight': Package_Weight,
    'Fuel_Efficiency': Fuel_Efficiency,
    'Warehouse_Processing_Time': Warehouse_Processing_Time
}])

# Ensure the order of columns matches the training data
input_data = input_data[feature_names]

if st.button('Predict Delivery Delay'):
    # Scale the input data
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = logi_scaled.predict(input_scaled)
    prediction_proba = logi_scaled.predict_proba(input_scaled)[:, 1]

    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.error(f'The delivery is likely to be **DELAYED** (Probability: {prediction_proba[0]:.2f})')
    else:
        st.success(f'The delivery is likely to be **ON TIME** (Probability: {prediction_proba[0]:.2f})')

st.markdown("""
--- 
#### Model Details:
- **Model**: Logistic Regression
- **Features**: Delivery Distance, Traffic Congestion, Weather Condition, Delivery Slot, Driver Experience, Num Stops, Vehicle Age, Road Condition Score, Package Weight, Fuel Efficiency, Warehouse Processing Time.
""")
