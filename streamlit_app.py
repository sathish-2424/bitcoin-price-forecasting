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
def fetch_bitcoin_data(period="60d", interval="1h"):
    """
    Fetch Bitcoin data with fallback strategy.
    First tries hourly data, falls back to daily if needed.
    """
    try:
        logger.info(f"Attempting to fetch {period} of {interval} Bitcoin data...")
        
        btc = yf.download(
            "BTC-USD",
            period=period,
            interval=interval,
            progress=False
        )
        
        # Ensure we have a DataFrame with proper columns
        if isinstance(btc, pd.DataFrame):
            btc = btc[['Close']].copy()
        else:
            raise ValueError("yfinance returned unexpected format")
        
        btc = btc.dropna()
        
        if len(btc) < SEQUENCE_LENGTH + 1:
            logger.warning(f"Got {len(btc)} rows with {interval}. Attempting daily data...")
            # Fallback to daily data
            btc = yf.download(
                "BTC-USD",
                period="1y",
                interval="1d",
                progress=False
            )
            
            if isinstance(btc, pd.DataFrame):
                btc = btc[['Close']].copy()
            else:
                raise ValueError("yfinance returned unexpected format")
            
            btc = btc.dropna()
        
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
st.caption("LSTM Neural Network | Forecasts next trading period")

# Check if refresh is needed (every 5 minutes)
if datetime.now() - st.session_state.last_update > timedelta(minutes=5):
    st.session_state.last_update = datetime.now()
    st.cache_data.clear()
    st.rerun()

try:
    # Fetch data with spinner
    with st.spinner("📊 Fetching Bitcoin data..."):
        btc = fetch_bitcoin_data()
    
    # Ensure btc is properly formatted
    if not isinstance(btc, pd.DataFrame):
        raise ValueError("Data format error: expected DataFrame")
    
    current_price = float(btc['Close'].iloc[-1])
    
    # Get close prices as numpy array
    close_prices = btc['Close'].values.reshape(-1, 1)
    
    # Train or load model with spinner
    with st.spinner("🤖 Loading prediction model..."):
        model, scaler = train_or_load_model(close_prices)
    
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
            "📈 Current BTC Price",
            f"${current_price:,.2f}",
            delta=None
        )
    
    with col2:
        delta_color = "green" if price_change >= 0 else "red"
        st.metric(
            "🔮 Predicted Price",
            f"${predicted_price:,.2f}",
            delta=f"${price_change:,.2f} ({price_change_pct:+.2f}%)"
        )
    
    with col3:
        if abs(price_change_pct) < 1:
            confidence = "🟢 Very High"
        elif abs(price_change_pct) < 2:
            confidence = "🟢 High"
        elif abs(price_change_pct) < 5:
            confidence = "🟡 Medium"
        else:
            confidence = "🔴 Low"
        st.metric("Confidence Level", confidence)
    
    # Plot
    st.subheader("📊 Price Trend & Forecast")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    actual_prices = btc['Close'].values
    time_index = btc.index
    
    # Plot actual prices
    ax.plot(
        time_index,
        actual_prices,
        label="Actual BTC Price",
        linewidth=2.5,
        color="#1f77b4",
        alpha=0.8
    )
    
    # Plot prediction (extend one period forward)
    predicted_time = time_index[-1] + (time_index[-1] - time_index[-2])
    ax.plot(
        [time_index[-1], predicted_time],
        [current_price, predicted_price],
        linestyle="--",
        marker="o",
        markersize=8,
        label="Predicted Next Period",
        linewidth=2.5,
        color="#ff7f0e"
    )
    
    # Add shaded area for uncertainty (±1.5%)
    uncertainty = predicted_price * 0.015
    ax.fill_between(
        [time_index[-1], predicted_time],
        [current_price - (current_price * 0.015), predicted_price - uncertainty],
        [current_price + (current_price * 0.015), predicted_price + uncertainty],
        alpha=0.2,
        color="#ff7f0e",
        label="±1.5% Uncertainty Band"
    )
    
    ax.set_title("Bitcoin Price: Historical vs Forecast", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Price (USD)", fontsize=12)
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")
    
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
            **🧠 Model Architecture:**
            - Type: 2-Layer LSTM Neural Network
            - Layer 1: 50 LSTM units (ReLU activation)
            - Layer 2: 50 LSTM units (ReLU activation)
            - Output: Dense layer with 1 unit
            - Sequence Length: {SEQUENCE_LENGTH} periods
            - Training Data Points: {len(btc)}
            """
        )
    
    with col2:
        next_refresh = st.session_state.last_update + timedelta(minutes=5)
        time_until_refresh = (next_refresh - datetime.now()).total_seconds()
        
        st.success(
            f"""
            **✅ System Status:**
            - Model Status: Active & Trained
            - Data Source: Yahoo Finance
            - Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            - Next Auto-Refresh: {next_refresh.strftime('%H:%M:%S')}
            - Time Until Refresh: {int(time_until_refresh)}s
            """
        )
    
    st.divider()
    st.caption(
        "⚠️ Disclaimer: This is a machine learning model for educational purposes only. "
        "Bitcoin prices are volatile and unpredictable. Always conduct thorough research "
        "before making investment decisions."
    )

except ValueError as e:
    st.error(f"❌ Data Error: {e}")
    st.warning(
        "The system couldn't fetch enough Bitcoin data. This can happen if:\n"
        "- Yahoo Finance API is temporarily unavailable\n"
        "- Network connection issues\n"
        "- Market data delays\n\n"
        "**Solution:** Please wait a moment and refresh the page."
    )

except Exception as e:
    logger.error(f"Unexpected error: {e}", exc_info=True)
    st.error(f"❌ Unexpected Error: {str(e)}")
    st.info(
        "An unexpected error occurred. Please try one of the following:\n"
        "1. Refresh the page\n"
        "2. Clear browser cache (Ctrl+Shift+Delete)\n"
        "3. Try again in a few minutes\n\n"
        "If the problem persists, check your internet connection."
    )