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
    page_title="Bitcoin Price Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# Bitcoin Price Prediction\nA machine learning-powered forecasting app"
    }
)

# Custom CSS for improved UI
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .section-header {
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 0.25rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for prediction history
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []
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

# Page title and sidebar
st.markdown("<h1 style='text-align: center; color: #f7931a;'>🚀 Bitcoin Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>ML-Powered Bitcoin Forecasting Dashboard</p>", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    refresh_rate = st.selectbox(
        "Auto-refresh rate:",
        ["Every 5 minutes", "Every 10 minutes", "Manual"],
        key="refresh_rate"
    )
    
    st.markdown("---")
    st.markdown("### 📚 About")
    st.markdown("""
    This dashboard uses LSTM neural networks to predict Bitcoin prices based on historical data.
    
    **Features:**
    - Real-time BTC data
    - ML-powered predictions
    - Trend analysis
    - Prediction history tracking
    """)
    
    st.markdown("---")
    st.info("Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Check if refresh is needed
refresh_minutes = 5 if "5 minutes" in refresh_rate else (10 if "10 minutes" in refresh_rate else 999)
if datetime.now() - st.session_state.last_update > timedelta(minutes=refresh_minutes):
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
    
    price_change = predicted_price - current_price
    price_change_pct = (price_change / current_price) * 100
    
    # Store prediction in history
    new_prediction = {
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Current Price': f"${current_price:,.2f}",
        'Predicted Price': f"${predicted_price:,.2f}",
        'Change': f"${price_change:,.2f}",
        'Change %': f"{price_change_pct:+.2f}%",
        'Direction': '📈 Up' if price_change >= 0 else '📉 Down'
    }
    
    # Keep only last 10 predictions
    st.session_state.predictions_history.insert(0, new_prediction)
    if len(st.session_state.predictions_history) > 10:
        st.session_state.predictions_history = st.session_state.predictions_history[:10]
    
    # === KEY METRICS SECTION ===
    st.markdown("<div class='section-header'><h2>📊 Key Metrics</h2></div>", unsafe_allow_html=True)
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric(
            "Current Price",
            f"${current_price:,.2f}",
            delta=None
        )
    
    with metric_col2:
        st.metric(
            "Predicted Price",
            f"${predicted_price:,.2f}",
            delta=f"${price_change:,.2f}",
            delta_color="normal"
        )
    
    with metric_col3:
        st.metric(
            "% Change",
            f"{price_change_pct:+.2f}%",
            delta=None,
            delta_color="off"
        )
    
    with metric_col4:
        if abs(price_change_pct) < 1:
            confidence = "🟢 Very High"
            conf_desc = "Very confident"
        elif abs(price_change_pct) < 2:
            confidence = "🟢 High"
            conf_desc = "Confident"
        elif abs(price_change_pct) < 5:
            confidence = "🟡 Medium"
            conf_desc = "Moderate"
        else:
            confidence = "🔴 Low"
            conf_desc = "High uncertainty"
        
        st.metric("Confidence", confidence, conf_desc)
    
    # === PREDICTION DIRECTION BOX ===
    st.markdown("<br>", unsafe_allow_html=True)
    
    if price_change >= 0:
        st.markdown(f"""
        <div class='success-box'>
            <strong>📈 Price Prediction: UP</strong><br>
            Expected to rise by <strong>${abs(price_change):,.2f}</strong> ({abs(price_change_pct):+.2f}%)
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='warning-box'>
            <strong>📉 Price Prediction: DOWN</strong><br>
            Expected to decline by <strong>${abs(price_change):,.2f}</strong> ({abs(price_change_pct):+.2f}%)
        </div>
        """, unsafe_allow_html=True)
    
    # === CHARTS SECTION ===
    st.markdown("<div class='section-header'><h2>📈 Charts & Analysis</h2></div>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Price Forecast", "Candlestick", "Prediction History"])
    
    with tab1:
        st.subheader("Price Trend & Forecast")
        
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
    
    with tab2:
        st.subheader("Candlestick Chart (Last 30 Days)")
        
        # Create candlestick data from daily prices
        daily_data = yf.download("BTC-USD", period="30d", interval="1d", progress=False)
        
        if isinstance(daily_data, pd.DataFrame) and len(daily_data) > 0:
            daily_data = daily_data[['Open', 'High', 'Low', 'Close']].copy()
            
            fig_candle, ax_candle = plt.subplots(figsize=(14, 6))
            
            # Color for candlesticks
            colors = ['green' if daily_data['Close'].iloc[i] >= daily_data['Open'].iloc[i] 
                      else 'red' for i in range(len(daily_data))]
            
            width = 0.6
            width2 = 0.05
            
            x = np.arange(len(daily_data))
            
            # Draw high-low lines
            for i in range(len(daily_data)):
                ax_candle.plot([i, i], [daily_data['Low'].iloc[i], daily_data['High'].iloc[i]], 
                              color=colors[i], linewidth=1.5)
            
            # Draw open-close rectangles
            for i in range(len(daily_data)):
                open_price = daily_data['Open'].iloc[i]
                close_price = daily_data['Close'].iloc[i]
                height = close_price - open_price
                bottom = min(open_price, close_price)
                
                ax_candle.bar(i, height, width2, bottom=bottom, color=colors[i], edgecolor='black', linewidth=0.5)
            
            ax_candle.set_xticks(x[::3])
            ax_candle.set_xticklabels([daily_data.index[i].strftime('%Y-%m-%d') for i in range(0, len(daily_data), 3)], rotation=45)
            ax_candle.set_title("Bitcoin Candlestick Chart (Last 30 Days)", fontsize=14, fontweight="bold")
            ax_candle.set_xlabel("Date", fontsize=12)
            ax_candle.set_ylabel("Price (USD)", fontsize=12)
            ax_candle.grid(True, alpha=0.3, linestyle="--")
            ax_candle.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
            
            plt.tight_layout()
            st.pyplot(fig_candle, use_container_width=True)
        else:
            st.warning("Unable to fetch candlestick data")
    
    with tab3:
        st.subheader("Prediction History")
        
        if st.session_state.predictions_history:
            history_df = pd.DataFrame(st.session_state.predictions_history)
            
            # Custom styling for dataframe
            st.dataframe(
                history_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Timestamp": st.column_config.TextColumn("⏰ Time", width="medium"),
                    "Current Price": st.column_config.TextColumn("💰 Current", width="small"),
                    "Predicted Price": st.column_config.TextColumn("🔮 Predicted", width="small"),
                    "Change": st.column_config.TextColumn("Δ ($)", width="small"),
                    "Change %": st.column_config.TextColumn("Δ (%)", width="small"),
                    "Direction": st.column_config.TextColumn("📊 Direction", width="small"),
                }
            )
            
            # Export button
            csv = history_df.to_csv(index=False)
            st.download_button(
                label="📥 Download History as CSV",
                data=csv,
                file_name="btc_predictions.csv",
                mime="text/csv"
            )
        else:
            st.info("📋 No prediction history yet. Predictions will appear here as they are generated.")

except Exception as e:
    st.error(f"❌ An error occurred: {str(e)}")
    st.info("💡 Try refreshing the page or check your internet connection.")