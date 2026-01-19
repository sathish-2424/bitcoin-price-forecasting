import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
import os

MODEL_PATH = "bitcoin_lstm.h5"
scaler = MinMaxScaler(feature_range=(0, 1))

def create_sequences(data, steps=60):
    X, y = [], []
    for i in range(steps, len(data)):
        X.append(data[i-steps:i, 0])
        y.append(data[i, 0])
    X = np.array(X).reshape(-1, steps, 1)
    y = np.array(y)
    return X, y

def train_or_load_model(close_prices):
    scaled = scaler.fit_transform(close_prices)

    X, y = create_sequences(scaled)

    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
    else:
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(60, 1)),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y, epochs=10, batch_size=32)
        model.save(MODEL_PATH)

    return model, scaled