import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Fetch stock data using Yahoo Finance
ticker = "CTRA"
df = yf.download(ticker, start="2019-01-01", end="2022-01-01")
df = df[['Close']]

# Preprocess the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# Prepare the data for training and testing
prediction_days = 60  # Lookback period

split_ratio = 0.8  # Train/Test split
split_idx = int(len(scaled_data) * split_ratio)

x_train = np.array([scaled_data[i - prediction_days:i, 0] for i in range(prediction_days, split_idx)])
y_train = np.array([scaled_data[i, 0] for i in range(prediction_days, split_idx)])

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

# Build the LSTM model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    LSTM(units=50, return_sequences=False),
    Dense(units=25),
    Dense(units=1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model with a validation split to avoid overfitting
model.fit(x_train, y_train, epochs=15, batch_size=64, verbose=1, validation_split=0.1)

# Backtesting preparation
x_test = np.array([scaled_data[i - prediction_days:i, 0] for i in range(split_idx, len(scaled_data) - 1)])
y_test = np.array([scaled_data[i, 0] for i in range(split_idx, len(scaled_data) - 1)])

x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)
real_prices = df['Close'].iloc[split_idx+prediction_days:].values

# Ensure both arrays are 1D and match in length
predicted_prices = predicted_prices.flatten()[:len(real_prices)]
real_prices = real_prices.flatten()

print(f"Adjusted Length of predicted_prices: {len(predicted_prices)}")
print(f"Adjusted Length of real_prices: {len(real_prices)}")

# Vectorized Strategy Parameters
capital = 10000  # Initial investment
position = 0  # Number of shares
capital_history = np.zeros(len(real_prices))
buy_signals = []
sell_signals = []

stop_loss = 0.97  # Stop-Loss at -3%
take_profit = 1.05  # Take-Profit at +5%
confidence_threshold = 0.02  # Minimum % change required to trade

# Precompute price changes for faster execution
predicted_changes = (predicted_prices[1:] - real_prices[:-1]) / real_prices[:-1]

# Buy & Sell Mask
buy_mask = (predicted_changes > confidence_threshold) & (capital > 0)
sell_mask = (predicted_changes < -confidence_threshold) & (position > 0)

for i in range(len(real_prices) - 1):
    if i % 500 == 0:
        print(f"Processing trade {i}/{len(real_prices)-1}...")

    if buy_mask[i]:  # Buy Signal
        position = capital / real_prices[i]
        buy_price = real_prices[i]
        buy_signals.append((i, real_prices[i]))
        capital = 0

    if sell_mask[i]:  # Sell Signal
        current_price = real_prices[i]
        if (
            current_price <= buy_price * stop_loss  # Stop-Loss Trigger
            or current_price >= buy_price * take_profit  # Take-Profit Trigger
        ):
            capital = position * current_price
            sell_signals.append((i, real_prices[i]))
            position = 0

    capital_history[i] = capital + position * real_prices[i]

# Convert buy/sell signals for plotting
buy_signals = np.array(buy_signals)
sell_signals = np.array(sell_signals)

# Plot actual vs predicted prices with buy/sell markers
plt.figure(figsize=(12, 6))
plt.plot(real_prices, label="Actual Price", color="blue")
plt.plot(predicted_prices, label="Predicted Price", color="red", linestyle="dashed")

if buy_signals.size > 0:
    plt.scatter(buy_signals[:, 0], buy_signals[:, 1], marker="^", color="green", label="Buy", s=100)
if sell_signals.size > 0:
    plt.scatter(sell_signals[:, 0], sell_signals[:, 1], marker="v", color="red", label="Sell", s=100)

plt.legend()
plt.title(f"{ticker} Price Prediction & Backtest")
plt.show()

# Plot capital over time
plt.figure(figsize=(12, 6))
plt.plot(capital_history, label="Strategy Capital", color="green")
plt.axhline(10000, color="gray", linestyle="dashed", label="Initial Capital")
plt.legend()
plt.title("Capital Growth Over Time")
plt.show()

print(f"Final capital: ${capital + position * real_prices[-1]:,.2f}")
