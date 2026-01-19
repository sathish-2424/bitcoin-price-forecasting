import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from model import train_or_load_model, SEQUENCE_LENGTH
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Bitcoin Price Prediction", layout="wide")

if 'last_update' not in st.session_state:
st.session_state.last_update = datetime.now()
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []

@st.cache_data(ttl=1800)
def fetch_bitcoin_data(period="60d", interval="1h"):
    try:
        logger.info(f"Fetching {period} of {interval} Bitcoin data...")
        btc = yf.download("BTC-USD", period=period, interval=interval, progress=False)
        
        if isinstance(btc, pd.DataFrame):
            btc = btc[['Close']].copy()
        else:
            raise ValueError("yfinance returned unexpected format")
        
        btc = btc.dropna()
        
        if len(btc) < SEQUENCE_LENGTH + 1:
            logger.warning(f"Got {len(btc)} rows with {interval}. Fetching daily data...")
            btc = yf.download("BTC-USD", period="1y", interval="1d", progress=False)
            if isinstance(btc, pd.DataFrame):
                btc = btc[['Close']].copy()
            btc = btc.dropna()
        
        if len(btc) < SEQUENCE_LENGTH + 1:
            raise ValueError(f"Insufficient data: got {len(btc)} rows, need {SEQUENCE_LENGTH + 1}")
        
        logger.info(f"Successfully fetched {len(btc)} data points")
        return btc
    
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        raise

st.title("🚀 Bitcoin Price Prediction")

if datetime.now() - st.session_state.last_update > timedelta(minutes=5):
    st.session_state.last_update = datetime.now()
    st.cache_data.clear()
    st.rerun()

try:
    with st.spinner("📊 Fetching Bitcoin data..."):
        btc = fetch_bitcoin_data()
    
    if not isinstance(btc, pd.DataFrame):
        raise ValueError("Data format error: expected DataFrame")
    
    current_price = float(btc['Close'].iloc[-1])
    close_prices = btc['Close'].values.reshape(-1, 1)
    
    with st.spinner("🤖 Loading model..."):
        model, scaler = train_or_load_model(close_prices)
    
    last_sequence = scaler.transform(btc[['Close']].tail(SEQUENCE_LENGTH).values).reshape(1, SEQUENCE_LENGTH, 1)
    prediction = model.predict(last_sequence, verbose=0)
    predicted_price = scaler.inverse_transform(prediction)[0][0]
    
    price_change = predicted_price - current_price
    price_change_pct = (price_change / current_price) * 100
    
    new_prediction = {
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Current Price': f"${current_price:,.2f}",
        'Predicted Price': f"${predicted_price:,.2f}",
        'Change': f"${price_change:,.2f}",
        'Change %': f"{price_change_pct:+.2f}%",
        'Direction': '📈 Up' if price_change >= 0 else '📉 Down'
    }
    
    st.session_state.predictions_history.insert(0, new_prediction)
    if len(st.session_state.predictions_history) > 10:
        st.session_state.predictions_history = st.session_state.predictions_history[:10]
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📈 Current BTC", f"${current_price:,.2f}")
    with col2:
        st.metric("🔮 Predicted", f"${predicted_price:,.2f}", delta=f"{price_change_pct:+.2f}%")
    with col3:
        confidence = "🟢 High" if abs(price_change_pct) < 2 else "🟡 Medium" if abs(price_change_pct) < 5 else "🔴 Low"
        st.metric("Confidence", confidence)
    
    # Prediction History Table
    st.subheader("📋 Last 10 Predictions")
    if st.session_state.predictions_history:
        history_df = pd.DataFrame(st.session_state.predictions_history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
    
        

    st.subheader("🕯️ Candlestick Chart (Last 30 Days)")
    
    try:
        daily_data = yf.download("BTC-USD", period="30d", interval="1d", progress=False)
        
        if isinstance(daily_data, pd.DataFrame) and len(daily_data) > 0:
            daily_data = daily_data[['Open', 'High', 'Low', 'Close']].copy()
            
            fig_candle, ax_candle = plt.subplots(figsize=(14, 6))
            
            x = np.arange(len(daily_data))
            width = 0.6
            
            for i in range(len(daily_data)):
                open_price = float(daily_data['Open'].iloc[i])
                close_price = float(daily_data['Close'].iloc[i])
                high_price = float(daily_data['High'].iloc[i])
                low_price = float(daily_data['Low'].iloc[i])
                
                color = 'green' if close_price >= open_price else 'red'
                
                # High-Low line
                ax_candle.plot([i, i], [low_price, high_price], color=color, linewidth=1.5)
                
                # Open-Close rectangle
                height = close_price - open_price
                bottom = min(open_price, close_price)
                ax_candle.bar(i, height, width/10, bottom=bottom, color=color, edgecolor='black', linewidth=0.5)
            
            ax_candle.set_xticks(x[::3])
            ax_candle.set_xticklabels([daily_data.index[i].strftime('%Y-%m-%d') for i in range(0, len(daily_data), 3)], rotation=45)
            ax_candle.set_title("Bitcoin Candlestick Chart (Last 30 Days)", fontsize=14, fontweight="bold")
            ax_candle.set_xlabel("Date", fontsize=12)
            ax_candle.set_ylabel("Price (USD)", fontsize=12)
            ax_candle.grid(True, alpha=0.3)
            ax_candle.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
            
            plt.tight_layout()
            st.pyplot(fig_candle, use_container_width=True)
    except Exception as e:
        logger.warning(f"Candlestick chart error: {e}")

except Exception as e:
    logger.error(f"Error: {e}", exc_info=True)
    st.error(f"❌ Error: {str(e)}")
    st.info("Please refresh the page and try again.")