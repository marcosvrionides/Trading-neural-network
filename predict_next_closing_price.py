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
dataset = pd.read_csv('./archive/eurusd_hour.csv')
dataset = dataset[['BO', 'BH', 'BL', 'BC']]

# Choose the number of previous hours (sequence length) to consider
sequence_length = 36

# Prepare input and output data
X, y = [], []

# Create input-output pairs for predicting the price 12 hours later
target_index = sequence_length + 12  # 12 hours later

for i in range(len(dataset) - target_index - 1):
    X.append(dataset.iloc[i:(i + sequence_length)])  # input is an array of arrays of price data of length = sequence_length
    y.append(dataset.iloc[i + target_index, 3])  # target is the close price 12 hours later

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

# Plotting predicted and actual prices against the number of predictions
plt.plot(y_test, label='Actual')
plt.plot(predictions, label='Predicted')

plt.title('Predicted vs Actual Closing Prices')
plt.xlabel('Number of Predictions')
plt.ylabel('Closing Price')
plt.legend()
plt.show()

# Get the live dataset
live_data = yf.download(tickers = 'USDJPY=X' ,period ='3mo', interval = '1h')
ohlc_data = live_data[['Open', 'High', 'Low', 'Close']]
ohlc_array = ohlc_data.values
live_dataset = np.array(ohlc_array / 100)

target_values = live_dataset[:, 3] # Forth column is the closing price

# Predicting on Live Data 12 hours into the future
live_X = []
dates = []

# Predicting 12 hours into the future from the last available data point in the live dataset
for i in range(len(live_dataset) - sequence_length):
    dates.append(ohlc_data.index[i + sequence_length])
    live_X.append(live_dataset[i:(i + sequence_length)])  # Input is an array of arrays of price data of length = sequence_length

live_X = np.array(live_X)
live_X = np.reshape(live_X, (live_X.shape[0], sequence_length, 4))  # Assuming 4 features (O, H, L, C)

# Predict using the trained model for future 12 hours
live_predictions = model.predict(live_X).flatten()

# Plotting predicted and actual prices 12 hours into the future on live data
plt.plot(dates, live_predictions, label='Predicted 12 Hours in Future on Live Data')
plt.plot(dates, target_values[sequence_length:], label='Actual on Live Data')

plt.title('Predicted Closing Prices 12 Hours in Future on Live Data')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.grid(True, linestyle='-', color='gray', linewidth=0.5)
plt.show()
