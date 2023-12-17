import streamlit as st
import pandas as pd
import joblib

# Load the saved models
loaded_model = joblib.load('Arima_temperature_humidity_prediction_model.pkl')
loaded_temperature_model = joblib.load('Arima_temperature_model.pkl')
loaded_humidity_model = joblib.load('Arima_humidity_model.pkl')


def predict_next_hour(user_input_temperature, user_input_humidity):
    # Create a dataframe with user input
    user_df = pd.DataFrame([[user_input_temperature, user_input_humidity]], columns=['Temperature', 'Humidity'])

    # Predict for the next hour using the loaded model
    next_hour_prediction = loaded_model.predict(user_df)

    return next_hour_prediction

def main():
    st.title("Temperature and Humidity Predictor")
    st.write("Enter the current temperature and humidity to predict the next hour's values.")

    # Get user input
    user_input_temperature = st.number_input("Enter the current temperature")
    user_input_humidity = st.number_input("Enter the current humidity")

    if st.button("Predict"):
        prediction = predict_next_hour(user_input_temperature, user_input_humidity)
        st.write("Predicted temperature and humidity for the next hour:")
        st.write(f"Temperature: {prediction[0][0]}")
        st.write(f"Humidity: {prediction[0][1]}")

if __name__ == "__main__":
    main()
