import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from model import train_or_load_model, SEQUENCE_LENGTH

st.set_page_config(page_title="Bitcoin Price Prediction", layout="centered")

# Initialize session state
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def fetch_hourly_data():
    try:
        btc = yf.download("BTC-USD", period="7d", interval="1h", progress=False)
        btc = btc[['Close']].dropna()
        if len(btc) < SEQUENCE_LENGTH:
            raise ValueError("Insufficient data for prediction")
        return btc
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

st.title("🚀 Bitcoin Price Prediction (Hourly)")

# Check if refresh is needed
if datetime.now() - st.session_state.last_update > timedelta(minutes=5):
    st.session_state.last_update = datetime.now()
    st.cache_data.clear()
    st.rerun()

try:
    btc = fetch_hourly_data()
    model, scaler = train_or_load_model(btc[['Close']].values)
    
    # Predict next hour
    last_60 = scaler.transform(btc[['Close']].tail(SEQUENCE_LENGTH).values).reshape(1, SEQUENCE_LENGTH, 1)
    prediction = model.predict(last_60, verbose=0)
    predicted_price = scaler.inverse_transform(prediction)[0][0]
    
    # Validate prediction
    if predicted_price < 0:
        st.warning("Prediction validation warning")
    
    # Dashboard
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current BTC Price", f"${btc['Close'].iloc[-1]:,.2f}")
    with col2:
        st.metric("Predicted Next Hour", f"${predicted_price:,.2f}")
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    actual_prices = btc['Close'].values
    time_index = btc.index
    
    ax.plot(time_index, actual_prices, label="Actual Price", linewidth=2)
    
    predicted_time = time_index[-1] + pd.Timedelta(hours=1)
    ax.plot([time_index[-1], predicted_time], [actual_prices[-1], predicted_price],
            linestyle="--", marker="o", label="Predicted", linewidth=2)
    
    ax.set_title("Bitcoin Price Prediction")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    st.success("✅ Model loaded and prediction updated")

except Exception as e:
    st.error(f"Error: {e}")