import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "bitcoin_lstm.h5"
SCALER_PATH = "scaler.pkl"
SEQUENCE_LENGTH = 60

def create_sequences(data, steps=SEQUENCE_LENGTH):
    """Create sequences for LSTM training"""
    X, y = [], []
    for i in range(steps, len(data)):
        X.append(data[i-steps:i, 0])
        y.append(data[i, 0])
    return np.array(X).reshape(-1, steps, 1), np.array(y)

def train_or_load_model(close_prices):
    """Train new model or load existing one with its scaler"""
    try:
        # Check if both model and scaler exist
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            logger.info("Loading existing model and scaler")
            model = load_model(MODEL_PATH)
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            return model, scaler
        
        # Train new model
        logger.info("Training new model")
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(close_prices)
        
        X, y = create_sequences(scaled)
        
        if len(X) < 10:
            raise ValueError(f"Insufficient training data: {len(X)} sequences")
        
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(SEQUENCE_LENGTH, 1)),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y, epochs=20, batch_size=16, verbose=0)
        
        # Save model and scaler
        model.save(MODEL_PATH)
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(scaler, f)
        logger.info("Model and scaler saved")
        
        return model, scaler
    
    except Exception as e:
        logger.error(f"Error in train_or_load_model: {e}")
        raise