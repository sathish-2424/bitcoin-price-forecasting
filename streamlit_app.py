import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import timedelta
from xgboost import XGBRegressor
import plotly.express as px
import warnings
import os

# Suppress warnings for cleaner UI
warnings.filterwarnings("ignore")

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Bitcoin Price Prediction",
    layout="wide",
    page_icon="⚡"
)

# ================= CONFIG =================
TICKER = "BTC-USD"
PERIOD = "2y"  # Reduced to 2y for faster responsiveness, increase if needed
INTERVAL = "1d"
FUTURE_DAYS = 14
MIN_ROWS = 60  # Increased slightly to ensure rolling windows have data
BACKUP_FILE = "btc_backup.csv"

# Features used for training
FEATURES = [
    "price_lag1",
    "price_lag7",
    "ma_7",
    "ma_30",
    "volatility",
    "rsi"
]

# ================= INDICATORS =================
def add_technical_indicators(df):
    """
    Adds technical indicators to the DataFrame.
    """
    df = df.copy()

    # Lag features
    df["price_lag1"] = df["price"].shift(1)
    df["price_lag7"] = df["price"].shift(7)

    # Moving Averages
    df["ma_7"] = df["price"].rolling(7).mean()
    df["ma_30"] = df["price"].rolling(30).mean()
    
    # Volatility (Standard Deviation)
    df["volatility"] = df["price"].rolling(7).std()

    # RSI (Relative Strength Index)
    delta = df["price"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    return df

# ================= LOAD DATA (ROBUST) =================
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def load_data():
    df = pd.DataFrame()

    # ---- Try Yahoo Finance ----
    try:
        df = yf.download(
            TICKER,
            period=PERIOD,
            interval=INTERVAL,
            progress=False,
            auto_adjust=True  # Gets simplified OHLC data
        )
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        df = pd.DataFrame()

    # ---- Fallback to CSV ----
    if df.empty and os.path.exists(BACKUP_FILE):
        st.warning("Yahoo Finance unavailable. Using cached BTC data.")
        df = pd.read_csv(BACKUP_FILE)
        # Ensure 'Date' is datetime if loaded from CSV
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)

    if df.empty:
        return pd.DataFrame()

    # ---- Fix YFinance MultiIndex Issue ----
    # Newer yfinance versions return MultiIndex columns (Price, Ticker)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # ---- Standardize Columns ----
    df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]

    # Handle 'close' vs 'adj_close'
    if "close" in df.columns:
        df = df.rename(columns={"close": "price"})
    elif "adj_close" in df.columns:
        df = df.rename(columns={"adj_close": "price"})
    else:
        return pd.DataFrame()

    # Reset index to make Date a column
    df = df.reset_index()
    
    # Standardize Date column name
    if "Date" in df.columns:
        df.rename(columns={"Date": "date"}, inplace=True)
    
    # Ensure numeric
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # Add Indicators
    df = add_technical_indicators(df)
    
    # Create Target (Next Day Price)
    df["target"] = df["price"].shift(-1)

    # Remove NaNs created by rolling windows and shifting
    df = df.dropna().reset_index(drop=True)
    
    return df

# ================= TRAIN MODEL =================
@st.cache_resource
def train_model(df):
    # Split data to respect time series (train on past, test on recent)
    # However, for final forecasting, we usually retrain on ALL available data
    X = df[FEATURES]
    y = df["target"]

    model = XGBRegressor(
        n_estimators=500,        # Increased estimators
        learning_rate=0.01,      # Lower LR for better generalization
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)
    return model

# ================= FUTURE PREDICTION =================
def predict_future(df, model, days):
    # Start with the very last available row from historical data
    last_row = df.iloc[[-1]].copy()
    
    # We need a history buffer to calculate rolling averages for the new predictions
    # We take the last 60 days to ensure we have enough data for ma_30 and rsi
    history = df.tail(60).copy()
    
    future_preds = []
    current_date = pd.to_datetime(last_row["date"].values[0])

    for _ in range(days):
        # 1. Update indicators based on current history
        # We only need to calculate indicators for the last row to predict the next
        history_with_indicators = add_technical_indicators(history)
        
        # 2. Get the latest row features
        latest_features = history_with_indicators.iloc[[-1]][FEATURES]
        
        # 3. Predict next price
        pred_price = model.predict(latest_features)[0]
        pred_price = max(float(pred_price), 1.0) # Ensure price doesn't go negative

        # 4. Advance Date (BTC trades every day, NO weekend skipping)
        current_date += timedelta(days=1)

        # 5. Append prediction to history so next iteration uses it
        new_row = pd.DataFrame({
            "date": [current_date],
            "price": [pred_price]
        })
        
        # Concat and keep size manageable (sliding window)
        history = pd.concat([history, new_row], ignore_index=True)
        # Keep only last 60 rows to prevent slowdown
        history = history.tail(60)

        future_preds.append({"date": current_date, "predicted_price": pred_price})

    return pd.DataFrame(future_preds)

# ================= UI LAYOUT =================
st.title("⚡ Bitcoin Price Prediction")

# 1. Load Data
with st.spinner("Downloading market data..."):
    df = load_data()

if df.empty or len(df) < MIN_ROWS:
    st.error("Unable to load sufficient BTC data. Try again later.")
    st.stop()

# 2. Train Model
model = train_model(df)

# 3. Predict
future_df = predict_future(df, model, FUTURE_DAYS)

# 4. Metrics
current_price = df["price"].iloc[-1]
last_pred_price = future_df["predicted_price"].iloc[-1]
change_pct = ((last_pred_price - current_price) / current_price) * 100
color = "normal" if change_pct > 0 else "inverse"

st.markdown("---")
c1, c2, c3 = st.columns(3)
c1.metric("Current Price", f"${current_price:,.2f}")
c2.metric(f"Price in {FUTURE_DAYS} Days", f"${last_pred_price:,.2f}")
c3.metric("Expected Change", f"{change_pct:+.2f}%", delta_color=color)
st.markdown("---")

# 5. Plotting
# Combine history (last 90 days) and future for a seamless chart
history_chart = df.tail(90)[["date", "price"]].copy()
history_chart["Type"] = "Historical"

future_chart = future_df.rename(columns={"predicted_price": "price"})
future_chart["Type"] = "Forecast"

plot_df = pd.concat([history_chart, future_chart], ignore_index=True)

fig = px.line(
    plot_df, 
    x="date", 
    y="price", 
    color="Type",
    color_discrete_map={"Historical": "cyan", "Forecast": "orange"},
    title=f"BTC-USD Price Forecast ({FUTURE_DAYS} Days)"
)

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    hovermode="x unified",
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)

with st.expander("🔎 View Forecast Data"):
    st.dataframe(future_df.style.format({"predicted_price": "${:,.2f}"}))
