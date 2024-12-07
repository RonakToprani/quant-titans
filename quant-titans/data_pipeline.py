import yfinance as yf
import pandas as pd
from ta import add_all_ta_features
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
import feedparser

def fetch_stock_data(ticker, start_date, end_date):
    """Fetch historical stock data with technical indicators."""
    data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    data = add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
    data['Return'] = data['Close'].pct_change()
    return data.dropna()

def fetch_news_sentiment(rss_url):
    """Fetch average sentiment score from an RSS feed."""
    feed = feedparser.parse(rss_url)
    sentiments = [TextBlob(entry.title).sentiment.polarity for entry in feed.entries]
    return sum(sentiments) / len(sentiments) if sentiments else 0

def preprocess_data(data):
    """Add labels and scale features for model training."""
    # Add target labels for medium-term prediction (5-day future price increase)
    data['Target'] = (data['Close'].shift(-5) > data['Close']).astype(int)

    # Prepare features and scale
    features = data.drop(columns=['Target', 'Close']).values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Clean dataset
    data = data.dropna()
    return data, features_scaled, scaler

if __name__ == "__main__":
    # Example pipeline execution
    ticker = "AAPL"
    rss_url = "https://www.theverge.com/rss/index.xml"
    data = fetch_stock_data(ticker, "2015-01-01", "2024-01-01")
    sentiment_score = fetch_news_sentiment(rss_url)
    data['Sentiment'] = sentiment_score
    data, features_scaled, scaler = preprocess_data(data)
    data.to_csv("processed_data.csv", index=False)
