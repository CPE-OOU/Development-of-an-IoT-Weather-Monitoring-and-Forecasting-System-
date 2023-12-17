import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import joblib

# Load the dataset without parsing dates initially
data = pd.read_csv('CompiledCleanedData.csv')

# Convert 'Timestamp' column to datetime format
data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='%d/%m/%Y %H:%M')

# Set Timestamp as index
data.set_index('Timestamp', inplace=True)

# Resample data to hourly frequency if not already in hourly intervals
data = data.resample('H').mean().ffill()  # Adjust the resampling frequency as needed

# Selecting features and target variable
# Assuming 'Temperature' and 'Humidity' columns exist in the dataset
temperature = data['Temperature']
humidity = data['Humidity']

# Train the ARIMA model for temperature
temperature_model = ARIMA(temperature, order=(5, 1, 0))  # ARIMA parameters might need tuning
temperature_model_fit = temperature_model.fit()

# Train the ARIMA model for humidity
humidity_model = ARIMA(humidity, order=(5, 1, 0))  # ARIMA parameters might need tuning
humidity_model_fit = humidity_model.fit()

# Make predictions for the next hour based on user input
user_input_temperature = float(input("Enter the current temperature: "))
user_input_humidity = float(input("Enter the current humidity: "))

# Predict for the next hour
next_hour_temperature_prediction = temperature_model_fit.forecast(steps=1).iloc[0]
next_hour_humidity_prediction = humidity_model_fit.forecast(steps=1).iloc[0]

# Output the predicted values for the next hour
print("Predicted temperature and humidity for the next hour:")
print(f"Temperature: {next_hour_temperature_prediction}")
print(f"Humidity: {next_hour_humidity_prediction}")

# Save the models for future use
joblib.dump(temperature_model_fit, 'Arima_temperature_model.pkl')
joblib.dump(humidity_model_fit, 'Arima_humidity_model.pkl')
