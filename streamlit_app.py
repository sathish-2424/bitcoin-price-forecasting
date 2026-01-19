import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from model import train_or_load_model, SEQUENCE_LENGTH
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Live Bitcoin Price Prediction",
    layout="wide"
)

# Initialize session state
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def fetch_hourly_data():
    """Fetch 7 days of hourly Bitcoin data from Yahoo Finance"""
    try:
        logger.info("Fetching hourly Bitcoin data...")
        btc = yf.download(
            "BTC-USD",
            period="7d",
            interval="1h",
            progress=False
        )
        
        # Clean data
        btc = btc[['Close']].dropna()
        
        if len(btc) < SEQUENCE_LENGTH + 1:
            raise ValueError(
                f"Insufficient data: got {len(btc)} rows, need {SEQUENCE_LENGTH + 1}"
            )
        
        logger.info(f"Successfully fetched {len(btc)} data points")
        return btc
    
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        raise

# Page title
st.title("🚀 Live Bitcoin Price Prediction")
st.caption("Hourly LSTM Model | Auto-refresh every 5 minutes")

# Check if refresh is needed (every 5 minutes)
if datetime.now() - st.session_state.last_update > timedelta(minutes=5):
    st.session_state.last_update = datetime.now()
    st.cache_data.clear()
    st.rerun()

try:
    # Fetch data
    btc = fetch_hourly_data()
    current_price = btc['Close'].iloc[-1]
    
    # Train or load model
    with st.spinner("Loading model..."):
        model, scaler = train_or_load_model(btc[['Close']].values)
    
    # Make prediction
    last_sequence = scaler.transform(
        btc[['Close']].tail(SEQUENCE_LENGTH).values
    ).reshape(1, SEQUENCE_LENGTH, 1)
    
    prediction = model.predict(last_sequence, verbose=0)
    predicted_price = scaler.inverse_transform(prediction)[0][0]
    
    # Calculate price change
    price_change = predicted_price - current_price
    price_change_pct = (price_change / current_price) * 100
    
    # Metrics dashboard
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Current BTC Price",
            f"${current_price:,.2f}",
            delta=None
        )
    
    with col2:
        st.metric(
            "Predicted Next Hour",
            f"${predicted_price:,.2f}",
            delta=f"${price_change:,.2f} ({price_change_pct:+.2f}%)"
        )
    
    with col3:
        confidence = "🟢 High" if abs(price_change_pct) < 2 else "🟡 Medium" if abs(price_change_pct) < 5 else "🔴 Low"
        st.metric("Prediction Confidence", confidence)
    
    # Plot
    st.subheader("Price Trend & Forecast")
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    actual_prices = btc['Close'].values
    time_index = btc.index
    
    # Plot actual prices
    ax.plot(
        time_index,
        actual_prices,
        label="Actual BTC Price",
        linewidth=2.5,
        color="#1f77b4"
    )
    
    # Plot prediction
    predicted_time = time_index[-1] + pd.Timedelta(hours=1)
    ax.plot(
        [time_index[-1], predicted_time],
        [current_price, predicted_price],
        linestyle="--",
        marker="o",
        markersize=8,
        label="Predicted Next Hour",
        linewidth=2.5,
        color="#ff7f0e"
    )
    
    # Add shaded area for uncertainty
    ax.fill_between(
        [time_index[-1], predicted_time],
        [current_price - (current_price * 0.01), predicted_price - (predicted_price * 0.01)],
        [current_price + (current_price * 0.01), predicted_price + (predicted_price * 0.01)],
        alpha=0.2,
        color="#ff7f0e",
        label="±1% Uncertainty"
    )
    
    ax.set_title("Bitcoin Price: Actual vs Predicted", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Price (USD)", fontsize=11)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Format y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    
    # Info section
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(
            f"""
            **Model Details:**
            - Architecture: 2-layer LSTM (50 units each)
            - Sequence Length: {SEQUENCE_LENGTH} hours
            - Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            - Data Points: {len(btc)}
            """
        )
    
    with col2:
        st.success(
            f"""
            **✅ Prediction Status:**
            - Model Status: Active
            - Last Refresh: {st.session_state.last_update.strftime('%H:%M:%S')}
            - Next Refresh: {(st.session_state.last_update + timedelta(minutes=5)).strftime('%H:%M:%S')}
            - Data Source: Yahoo Finance
            """
        )

except ValueError as e:
    st.error(f"❌ Data Error: {e}")
    st.info("The Bitcoin hourly data requires at least 61 data points. Yahoo Finance may not have enough data at this moment. Please try again in a few minutes.")
    
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    st.error(f"❌ Error: {e}")
    st.info("Please check the logs for details and try refreshing the page.")