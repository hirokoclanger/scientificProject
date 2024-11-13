import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import matplotlib.pyplot as plt

# Load the saved model
model = load_model('lstm_model.keras')
# Set random seeds for reproducibility

# Load your data (using your provided loading steps)
nao_df = pd.read_excel('combined_data.xlsx', sheet_name='NAO')
npi_df = pd.read_excel('combined_data.xlsx', sheet_name='NPI')
oni_df = pd.read_excel('combined_data.xlsx', sheet_name='ONI')
#storage_df = pd.read_excel('combined_data.xlsx', sheet_name='STORAGE')
consumption_df = pd.read_excel('combined_data.xlsx', sheet_name='CONSUMPTION')
consumption_df.dropna(inplace=True)
# Calculate total consumption as the sum of "Residential" and "Commercial"
#consumption_df['Total_Consumption'] = consumption_df['Residential'] + consumption_df['Commercial']
# Merge dataframes on 'Date'
merged_df = nao_df.merge(oni_df, on='Date', how='outer') \
    .merge(consumption_df, on='Date', how='outer') #\
   # .merge(storage_df, on='Date', how='outer')

# Ensure 'Date' is in datetime format and set it as the index
merged_df['Date'] = pd.to_datetime(merged_df['Date'])
merged_df.set_index('Date', inplace=True)

# Filter data by date range if dates are in the index
df = merged_df[(merged_df.index >= '2015-09-01') & (merged_df.index <= '2024-02-01')]

# Scale the data
# Select input features (all columns except 'Total_Consumption') and target
X_data = df.drop(columns=['U.S. Natural Gas Residential Consumption (MMcf)']).values  # Input features
y_data = df['U.S. Natural Gas Residential Consumption (MMcf)'].values  # Target variable
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X_data)
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y_data.reshape(-1, 1))

# Prepare the last sequence from the training data
sequence_length = 60  # Make sure this matches the sequence length used during training
num_features = X_data.shape[1]
last_sequence = X_scaled[-sequence_length:].reshape(1, sequence_length, num_features)

# Generate 9 future predictions recursively
num_predictions = 9
predictions = []
for _ in range(num_predictions):
    # Make a prediction for the next time step
    next_pred = model.predict(last_sequence)
    predictions.append(next_pred[0, 0])  # Store the prediction

    # Update the sequence by appending the new prediction and removing the oldest time step
    next_pred_scaled = np.repeat(next_pred, num_features).reshape(1, 1, num_features)
    last_sequence = np.append(last_sequence[:, 1:, :], next_pred_scaled, axis=1)

# Inverse transform predictions to get back to the original scale
predictions_original_scale = scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1))

# Generate corresponding dates for the predictions
future_dates = pd.date_range(start='2024-03-01', periods=num_predictions, freq='MS')

# Create DataFrame for predictions
predicted_df = pd.DataFrame(data=predictions_original_scale, index=future_dates,
                            columns=['U.S. Natural Gas Residential Consumption (MMcf)'])
print(predicted_df)

# Plot the predictions along with historical data
plt.figure(figsize=(14, 7))
plt.plot(df['Total_Consumption'], label="Historical Total Consumption")
plt.plot(predicted_df, label="Predicted Total Consumption", linestyle='--', color="blue")
plt.title("Total Consumption Forecast for Sep - May 2025")
plt.xlabel("Date")
plt.ylabel("Total Consumption")
plt.legend()
plt.show()