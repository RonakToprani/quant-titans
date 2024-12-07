import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
import numpy as np

def train_random_forest(features, labels):
    """Train a Random Forest classifier."""
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    return model

def train_lstm(features, labels):
    """Train an LSTM model."""
    features = features.reshape((features.shape[0], features.shape[1], 1))
    X_train, X_test = features[:int(len(features)*0.8)], features[int(len(features)*0.8):]
    y_train, y_test = labels[:int(len(labels)*0.8)], labels[int(len(labels)*0.8):]

    model = Sequential([
        SimpleRNN(50, activation='relu', input_shape=(X_train.shape[1], 1)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    return model

if __name__ == "__main__":
    data = pd.read_csv("processed_data.csv")
    features = np.array(list(data['Scaled_Features']))
    labels = data['Target'].values

    # Train Random Forest
    rf_model = train_random_forest(features, labels)

    # Train LSTM
    lstm_model = train_lstm(features, labels)
