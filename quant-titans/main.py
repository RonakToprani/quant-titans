import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn for better visuals
sns.set(style="whitegrid")

# Function to fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data['Returns'] = stock_data['Close'].pct_change()
    print(f"Data fetched for {ticker}!")
    return stock_data

# Function to identify earnings events
def get_earnings_events(ticker):
    print(f"Fetching earnings dates for {ticker}...")
    stock = yf.Ticker(ticker)
    earnings_dates = stock.calendar
    if earnings_dates.empty:
        print(f"No earnings dates available for {ticker}.")
        return None
    earnings_dates = earnings_dates.transpose()
    print(f"Earnings dates retrieved!")
    return earnings_dates

# Function to backtest a strategy based on earnings
def backtest_earnings_strategy(stock_data, earnings_dates, pre_days=5, post_days=5):
    print("Backtesting strategy...")
    results = []
    
    for earnings_date in earnings_dates.index:
        earnings_date = pd.to_datetime(earnings_date)
        window_start = earnings_date - pd.Timedelta(days=pre_days)
        window_end = earnings_date + pd.Timedelta(days=post_days)

        if window_start not in stock_data.index or window_end not in stock_data.index:
            continue

        pre_event_data = stock_data.loc[window_start:earnings_date]
        post_event_data = stock_data.loc[earnings_date:window_end]

        pre_return = (pre_event_data['Close'].iloc[-1] / pre_event_data['Close'].iloc[0]) - 1
        post_return = (post_event_data['Close'].iloc[-1] / post_event_data['Close'].iloc[0]) - 1

        results.append({'Earnings Date': earnings_date, 'Pre Return': pre_return, 'Post Return': post_return})

    results_df = pd.DataFrame(results)
    print("Backtesting completed!")
    return results_df

# Function to visualize results
def visualize_results(stock_data, earnings_results):
    print("Visualizing data...")
    
    # Plot stock price
    plt.figure(figsize=(14, 7))
    plt.plot(stock_data['Close'], label="Stock Price")
    plt.title("Stock Price Over Time")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

    # Plot pre and post earnings returns
    plt.figure(figsize=(10, 6))
    sns.barplot(data=earnings_results.melt(id_vars=['Earnings Date'], var_name="Event Period", value_name="Return"),
                x='Earnings Date', y='Return', hue='Event Period')
    plt.title("Pre and Post Earnings Returns")
    plt.xticks(rotation=45)
    plt.show()

# Main script
if __name__ == "__main__":
#    ticker = input("Enter stock ticker (e.g., AAPL): ")
    ticker = input("Enter stock ticker (e.g., AAPL): ")
    
    start_date = input("Enter start date (YYYY-MM-DD): ")
    end_date = input("Enter end date (YYYY-MM-DD): ")
    
    # Fetch stock data
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    
    # Get earnings events
    earnings_events = get_earnings_events(ticker)
    if earnings_events is not None:
        earnings_results = backtest_earnings_strategy(stock_data, earnings_events)
        
        # Visualize the data and results
        visualize_results(stock_data, earnings_results)
    else:
        print("No earnings events to backtest.")
