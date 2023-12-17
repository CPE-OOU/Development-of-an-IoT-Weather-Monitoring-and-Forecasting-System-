import joblib
from statsmodels.tsa.arima.model import ARIMAResults

# Load the saved ARIMA models
loaded_temperature_model = joblib.load('Arima_temperature_model.pkl')
loaded_humidity_model = joblib.load('Arima_humidity_model.pkl')

# Ask user for input
user_input_temperature = float(input("Enter the current temperature: "))
user_input_humidity = float(input("Enter the current humidity: "))

# Predict for the next hour using the loaded models
next_hour_temperature_prediction = loaded_temperature_model.forecast(steps=1).iloc[0]
next_hour_humidity_prediction = loaded_humidity_model.forecast(steps=1).iloc[0]

# Output the predicted values for the next hour
print("Predicted temperature and humidity for the next hour:")
print(f"Temperature: {next_hour_temperature_prediction}")
print(f"Humidity: {next_hour_humidity_prediction}")
