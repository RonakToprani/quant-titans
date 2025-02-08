import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Fetch stock data using Yahoo Finance
ticker = "TSLA"
df = yf.download(ticker, start="2019-01-01", end="2025-01-01")

# Use only 'Close' price
df = df[['Close']]

# Preprocess the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# Prepare the data for training and testing
prediction_days = 60  # Lookback period

x_train, y_train = [], []
split_ratio = 0.8  # Train/Test split (80% train, 20% test)
split_idx = int(len(scaled_data) * split_ratio)

for x in range(prediction_days, split_idx):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build the LSTM model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    LSTM(units=50, return_sequences=False),
    Dense(units=25),
    Dense(units=1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32, verbose=1)

# Backtesting the strategy
x_test, y_test = [], []

for x in range(split_idx, len(scaled_data) - 1):
    x_test.append(scaled_data[x-prediction_days:x, 0])
    y_test.append(scaled_data[x, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)
real_prices = df['Close'].iloc[split_idx+prediction_days:].values  # Actual close prices

# Calculate strategy returns
capital = 10000  # Initial investment
position = 0
capital_history = []

# Ensure both are 1D arrays
predicted_prices = predicted_prices.flatten()
real_prices = real_prices.flatten()

# Trim predicted_prices to match real_prices length
predicted_prices = predicted_prices[:len(real_prices)]

# Debugging: Print new lengths
print(f"Adjusted Length of predicted_prices: {len(predicted_prices)}")
print(f"Adjusted Length of real_prices: {len(real_prices)}")



for i in range(len(predicted_prices) - 1):
    if predicted_prices[i] > real_prices[i]:  # Buy Signal
        position = capital / real_prices[i]
        capital = 0
    elif predicted_prices[i] < real_prices[i] and position > 0:  # Sell Signal
        capital = position * real_prices[i]
        position = 0
    capital_history.append(capital + position * real_prices[i])

# Plot performance
plt.figure(figsize=(12, 6))
plt.plot(real_prices, label="Actual Price", color="blue")
plt.plot(predicted_prices, label="Predicted Price", color="red", linestyle="dashed")
plt.legend()
plt.title(f"{ticker} Price Prediction & Backtest")
plt.show()

# Plot capital over time
plt.figure(figsize=(12, 6))
plt.plot(capital_history, label="Strategy Capital", color="green")
plt.axhline(10000, color="gray", linestyle="dashed", label="Initial Capital")
plt.legend()
plt.title("Backtest Performance")
plt.show()

print(f"Final capital: ${capital + position * real_prices[-1]:,.2f}")
