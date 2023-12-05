import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go

# Load dataset
dataset = np.loadtxt('./archive/eurusd_hour.csv', delimiter=',', skiprows=1, usecols=(2, 3, 4, 5))

# Choose the number of previous hours (sequence length) to consider
sequence_length = 36

# Prepare input and output data
X, y = [], []

# Create input-output pairs
# Create input-output pairs for predicting the price 12 hours later
target_index = sequence_length + 12  # 12 hours later

for i in range(len(dataset) - target_index - 1):
    X.append(dataset[i:(i + sequence_length)])  # input is an array of arrays of price data of length = sequence_length
    y.append(dataset[i + target_index, 3])  # target is the close price 12 hours later

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshaping the input data to fit an LSTM model (samples, time steps, features)
X_train = np.reshape(X_train, (X_train.shape[0], sequence_length, 4))  # Assuming 4 features (O, H, L, C)
X_test = np.reshape(X_test, (X_test.shape[0], sequence_length, 4))

# Creating the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(sequence_length, 4)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# Compiling the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Training the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Making predictions on the test set
predictions = model.predict(X_test).flatten()

# Creating x-axis values corresponding to the number of predictions
x_values = range(len(predictions))

# Plotting predicted and actual prices against the number of predictions
plt.plot(x_values, y_test, label='Actual')
plt.plot(x_values, predictions, label='Predicted')

plt.title('Predicted vs Actual Closing Prices')
plt.xlabel('Number of Predictions')
plt.ylabel('Closing Price')
plt.legend()
plt.show()

# Get the live dataset
live_data = yf.download(tickers = 'USDJPY=X' ,period ='3mo', interval = '1h')

ohlc_data = live_data[['Open', 'High', 'Low', 'Close']]
ohlc_array = ohlc_data.values
ohlc_1_to_4 = ohlc_array[:, 0:4]
live_dataset = np.array(ohlc_1_to_4 / 100)

actual_values = live_dataset[:, 3]

# Reshape the live dataset to fit the model input shape
live_X = []

for i in range(len(live_dataset) - sequence_length - 1):
    live_X.append(live_dataset[i:(i + sequence_length)])  # input is an array of arrays of price data of length = sequence_length

live_X = np.array(live_X)
live_X = np.reshape(live_X, (live_X.shape[0], sequence_length, 4))  # Assuming 4 features (O, H, L, C)

# Predict using the trained model
live_predictions = model.predict(live_X).flatten()

# Plotting predicted and actual prices on live data
plt.plot(live_predictions, label='Predicted on Live Data')
plt.plot(actual_values[sequence_length+1:], label='Actual on Live Data')

plt.title('Predicted Closing Prices on Live Data')
plt.xlabel('Interval')
plt.ylabel('Closing Price')
plt.legend()
plt.grid(True, linestyle='-', color='gray', linewidth=0.5)
plt.show()
