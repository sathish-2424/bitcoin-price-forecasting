import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
import os

MODEL_PATH = "bitcoin_lstm.h5"
SCALER_PATH = "scaler.pkl"
SEQUENCE_LENGTH = 60

def create_sequences(data, steps=SEQUENCE_LENGTH):
    X, y = [], []
    for i in range(steps, len(data)):
        X.append(data[i-steps:i, 0])
        y.append(data[i, 0])
    return np.array(X).reshape(-1, steps, 1), np.array(y)

def train_or_load_model(close_prices):
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = load_model(MODEL_PATH)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
    else:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(close_prices)
        X, y = create_sequences(scaled)
        
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(SEQUENCE_LENGTH, 1)),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y, epochs=10, batch_size=32, verbose=0)
        
        model.save(MODEL_PATH)
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(scaler, f)
    
    return model, scaler