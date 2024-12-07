def backtest(data, predictions, initial_balance=10000):
    """Simulates a trading strategy based on model predictions."""
    balance = initial_balance
    shares = 0
    for i in range(len(predictions)):
        if predictions[i] == 1:  # Buy signal
            shares = balance // data['Close'].iloc[i]
            balance -= shares * data['Close'].iloc[i]
        elif predictions[i] == 0 and shares > 0:  # Sell signal
            balance += shares * data['Close'].iloc[i]
            shares = 0
    return balance

if __name__ == "__main__":
    import pandas as pd
    from model_training import train_random_forest

    # Load data and train model
    data = pd.read_csv("processed_data.csv")
    features = list(data['Scaled_Features'])
    labels = data['Target']

    rf_model = train_random_forest(features, labels)
    predictions = rf_model.predict(features[-len(labels):])

    # Run backtesting
    final_balance = backtest(data[-len(labels):], predictions)
    print("Final Balance:", final_balance)
