import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('logi.sav')

st.title('Delivery Delay Prediction App')
st.write('Enter the details below to predict if a delivery will be delayed.')

# Input fields for each feature
delivery_distance = st.number_input('Delivery Distance (km)', min_value=0.0, value=20.0)
traffic_congestion = st.slider('Traffic Congestion (1-5, 5 is high)', 1, 5, 3)
weather_condition = st.slider('Weather Condition (1-5, 5 is severe)', 1, 5, 3)
delivery_slot = st.slider('Delivery Slot (1-3)', 1, 3, 2)
driver_experience = st.number_input('Driver Experience (years)', min_value=0, value=5)
num_stops = st.number_input('Number of Stops', min_value=0, value=2)
vehicle_age = st.number_input('Vehicle Age (years)', min_value=0, value=3)
road_condition_score = st.slider('Road Condition Score (1-5, 5 is excellent)', 1, 5, 3)
package_weight = st.number_input('Package Weight (kg)', min_value=0.0, value=10.0)
fuel_efficiency = st.number_input('Fuel Efficiency (km/l)', min_value=0.0, value=12.0)
warehouse_processing_time = st.number_input('Warehouse Processing Time (minutes)', min_value=0, value=60)


# Create a DataFrame from the input values
input_data = pd.DataFrame([{
    'Delivery_Distance': delivery_distance,
    'Traffic_Congestion': traffic_congestion,
    'Weather_Condition': weather_condition,
    'Delivery_Slot': delivery_slot,
    'Driver_Experience': driver_experience,
    'Num_Stops': num_stops,
    'Vehicle_Age': vehicle_age,
    'Road_Condition_Score': road_condition_score,
    'Package_Weight': package_weight,
    'Fuel_Efficiency': fuel_efficiency,
    'Warehouse_Processing_Time': warehouse_processing_time
}])

if st.button('Predict Delivery Delay'):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.error(f"Delivery is likely to be Delayed (Probability: {prediction_proba[0][1]:.2f})")
    else:
        st.success(f"Delivery is likely to be On Time (Probability: {prediction_proba[0][0]:.2f})")

st.write("\n--- Notes ---")
st.write("\n- **Delivery_Distance**: Distance for the delivery.")
st.write("- **Traffic_Congestion**: Level of traffic congestion (1=low, 5=high).")
st.write("- **Weather_Condition**: Severity of weather conditions (1=good, 5=severe).")
st.write("- **Delivery_Slot**: Designated time slot for delivery (1, 2, or 3).")
st.write("- **Driver_Experience**: Years of experience of the delivery driver.")
st.write("- **Num_Stops**: Number of stops the driver has to make.")
st.write("- **Vehicle_Age**: Age of the delivery vehicle.")
st.write("- **Road_Condition_Score**: Quality of road conditions (1=poor, 5=excellent).")
st.write("- **Package_Weight**: Weight of the package.")
st.write("- **Fuel_Efficiency**: Fuel efficiency of the vehicle.")
st.write("- **Warehouse_Processing_Time**: Time taken for package processing at the warehouse.")
