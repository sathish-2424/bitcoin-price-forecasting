# Bitcoin Price Prediction

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense


# Page Config (MUST BE FIRST)

st.set_page_config(
    page_title="Bitcoin Price Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 Bitcoin Price Prediction")


# Sidebar

st.sidebar.header("⚙️ Model Settings")

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
lookback = st.sidebar.slider("Lookback Days", 30, 120, 60)
epochs = st.sidebar.slider("Training Epochs", 5, 25, 10)
train_model = st.sidebar.checkbox("Train New Model", value=False)


# Load Data (CACHED FOR SPEED)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data(start, end):
    df = yf.download("BTC-USD", start=start, end=end, interval="1d", progress=False)

    # ✅ FIX: Remove MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df[["Close"]]


# Load/Save Model & Scaler (PERSISTENT CACHING)

MODEL_PATH = "bitcoin_lstm.h5"
SCALER_PATH = "scaler.pkl"

@st.cache_resource
def load_or_train_model(data_close, lookback_val, epochs_val, force_train=False):
    """Load existing model or train a new one"""
    
    # Check if we should load existing model
    if not force_train and os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            model = load_model(MODEL_PATH)
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            st.info("✅ Loaded cached model")
            return model, scaler
        except Exception as e:
            st.warning(f"Cache error: {e}. Training new model...")
    
    # Train new model
    with st.spinner("🔄 Training LSTM model..."):
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data_close.values.reshape(-1, 1))
        
        X, y = [], []
        for i in range(lookback_val, len(scaled_data)):
            X.append(scaled_data[i - lookback_val:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Lightweight model for faster training
        model = Sequential([
            LSTM(32, return_sequences=True, input_shape=(lookback_val, 1), activation='relu'),
            LSTM(32, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer="adam", loss="mean_squared_error")
        model.fit(X, y, epochs=epochs_val, batch_size=32, verbose=0)
        
        # Save model and scaler
        model.save(MODEL_PATH)
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(scaler, f)
    
    st.success("✅ Model trained and cached")
    return model, scaler

# Load Data & Train Model

data = load_data(start_date, end_date)

if len(data) < lookback + 10:
    st.error("⚠️ Not enough data for this date range")
    st.stop()

model, scaler = load_or_train_model(data["Close"], lookback, epochs, force_train=train_model)


# Historical Price Chart

st.subheader("📊 Historical Bitcoin Prices")
st.line_chart(data["Close"])


# Data Preparation (LAZY LOADING)

scaled_data = scaler.transform(data[["Close"]])

X, y = [], []
for i in range(lookback, len(scaled_data)):
    X.append(scaled_data[i - lookback:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
X = X.reshape(X.shape[0], X.shape[1], 1)


# Next Day Prediction

last_sequence = scaled_data[-lookback:]
last_sequence = last_sequence.reshape(1, lookback, 1)

@st.cache_data(ttl=3600)
def predict_next_day(last_seq, _model, _scaler):
    """Cached prediction"""
    prediction = _model.predict(last_seq, verbose=0)
    return _scaler.inverse_transform(prediction)[0][0]

next_day_price = predict_next_day(last_sequence, model, scaler)

current_price = data.iloc[-1]["Close"]
price_change = next_day_price - current_price


# Metrics

st.subheader("🔮 Prediction Summary")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Current Price", f"${current_price:,.2f}")
with col2:
    st.metric("Predicted Next Day", f"${next_day_price:,.2f}", f"{price_change:,.2f}")
with col3:
    pct_change = (price_change / current_price * 100) if current_price != 0 else 0
    st.metric("Change %", f"{pct_change:.2f}%")


# Tabs for Better Performance

tab1, tab2, tab3 = st.tabs(["📈 Analysis", "🔮 Forecast", "📋 Data"])

with tab1:
    st.subheader("🤖 Actual vs Predicted Prices (Training)")
    
    # Lazy compute predictions only when needed
    train_predictions = model.predict(X, verbose=0)
    train_predictions = scaler.inverse_transform(train_predictions)
    actual_prices = scaler.inverse_transform(y.reshape(-1, 1))
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(actual_prices, label="Actual", linewidth=2)
    ax.plot(train_predictions, label="Predicted", linewidth=2, alpha=0.7)
    ax.legend()
    ax.set_xlabel("Days")
    ax.set_ylabel("Price (USD)")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

with tab2:
    st.subheader("📅 7-Day Bitcoin Forecast")
    
    @st.cache_data(ttl=3600)
    def forecast_next_days(last_seq, future_days, _model, _scaler, _lookback):
        """Cached 7-day forecast"""
        temp_input = list(last_seq.flatten())
        future_predictions = []
        
        for _ in range(future_days):
            x_input = np.array(temp_input[-_lookback:])
            x_input = x_input.reshape(1, _lookback, 1)
            yhat = _model.predict(x_input, verbose=0)
            temp_input.append(yhat[0][0])
            future_predictions.append(yhat[0][0])
        
        future_predictions = _scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
        return future_predictions.flatten()
    
    future_days = 7
    future_predictions = forecast_next_days(last_sequence, future_days, model, scaler, lookback)
    
    future_df = pd.DataFrame({
        "Day": range(1, future_days + 1),
        "Predicted Price ($)": [f"${x:,.2f}" for x in future_predictions]
    })
    
    st.dataframe(future_df, use_container_width=True)

with tab3:
    st.subheader("📋 Last 10 Days Prices")
    last_10 = data.tail(10).copy()
    last_10["Close"] = last_10["Close"].apply(lambda x: f"${x:,.2f}")
    st.dataframe(last_10, use_container_width=True)


# Footer

st.markdown("---")
