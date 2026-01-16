import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import plotly.express as px
import warnings

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Bitcoin Price Prediction",
    layout="wide",
    page_icon="⚡"
)
warnings.filterwarnings("ignore")

# ================= CONFIGURATION =================
TICKER = "BTC-USD"
PERIOD = "2y" # Reduced for speed in demo, increase for production
INTERVAL = "1d"
LOOKBACK = 60
FUTURE_DAYS = 14
FEATURES = ["price", "ma_7", "ma_30", "volatility", "rsi"]

# ================= HELPER: INDICATOR CALCULATION =================
def add_technical_indicators(df):
    """
    Calculates technical indicators. 
    Refactored into a function so it can be called during the prediction loop
    to prevent 'frozen feature' logic errors.
    """
    df = df.copy()
    # Moving Averages
    df["ma_7"] = df["price"].rolling(window=7).mean()
    df["ma_30"] = df["price"].rolling(window=30).mean()
    
    # Volatility
    df["volatility"] = df["price"].rolling(window=7).std()
    
    # RSI Calculation
    delta = df["price"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))
    
    return df

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    """
    Optimized data loading with robust column handling.
    """
    try:
        df = yf.download(TICKER, period=PERIOD, interval=INTERVAL, progress=False)
        
        # Reset index to make Date a column
        df.reset_index(inplace=True)
        
        # Handle yfinance multi-index columns (common issue in v0.2+)
        if isinstance(df.columns, pd.MultiIndex):
            # Flatten columns: 'Close' -> 'price', keep 'Date'
            try:
                # Find the Close column dynamically
                price_col = df["Close"] if "Close" in df.columns else df.iloc[:, 1]
                if isinstance(price_col, pd.DataFrame):
                    price_col = price_col.iloc[:, 0] # Take first column if still DF
            except:
                price_col = df.iloc[:, 1] # Fallback to 2nd column
                
            df_clean = pd.DataFrame({
                "date": df["Date"].iloc[:, 0] if isinstance(df["Date"], pd.DataFrame) else df["Date"],
                "price": price_col
            })
        else:
            # Standard single index handling
            df_clean = df[["Date", "Close"]].rename(columns={"Date": "date", "Close": "price"})

        # Ensure numeric
        df_clean["price"] = pd.to_numeric(df_clean["price"], errors='coerce')
        
        # Calculate Features
        df_clean = add_technical_indicators(df_clean)
        
        # Drop NaNs created by rolling windows
        df_clean.dropna(inplace=True)
        
        return df_clean
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# ================= DSA OPTIMIZATION: SLIDING WINDOW =================
def create_sequences_vectorized(data, lookback):
    """
    Optimized Sequence Generation (DSA Approach).
    Instead of a Python for-loop (O(N)), we use numpy striding/slicing 
    for faster memory access.
    """
    n_samples = len(data) - lookback
    
    # Create X (Features)
    # This creates a view of the array with shape (n_samples, lookback, n_features)
    # It is significantly faster than list appending for large datasets.
    X = np.array([data[i:i+lookback] for i in range(n_samples)])
    
    # Create y (Target - the price of the next day)
    y = data[lookback:, 0] # Assuming index 0 is 'price'
    
    return X, y

# ================= TRAIN MODEL =================
@st.cache_resource
def train_gru_model(df):
    # Prepare Data
    feature_data = df[FEATURES].values
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(feature_data)
    
    X, y = create_sequences_vectorized(scaled_data, LOOKBACK)
    
    if len(X) == 0:
        return None, None

    # Split Data (No shuffling for Time Series!)
    split = int(len(X) * 0.9)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Build Model
    model = Sequential([
        Input(shape=(LOOKBACK, len(FEATURES))),
        GRU(64, return_sequences=False), # Simplified architecture for stability
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1) # Linear output for regression
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    model.fit(
        X_train, y_train,
        epochs=25, # Reduced for demo speed
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
        verbose=0
    )
    
    return model, scaler

# ================= OPTIMIZED PREDICTION LOOP =================
def predict_recursive(model, scaler, df, days_ahead):
    """
    Corrects the logic error: We must RECALCULATE features (MA, RSI) 
    after every predicted price, otherwise the model sees inconsistent data.
    """
    # 1. Get sufficient history to calculate max rolling window (30)
    # We need at least LOOKBACK + 30 days of history
    history_df = df.iloc[-(LOOKBACK + 40):].copy() 
    
    future_predictions = []
    last_date = df["date"].iloc[-1]
    
    for _ in range(days_ahead):
        # A. Recalculate indicators on the CURRENT history
        # This ensures MA_7 and RSI change as we add predicted prices
        temp_df = add_technical_indicators(history_df)
        
        # B. Get the last LOOKBACK rows of features
        valid_features = temp_df[FEATURES].tail(LOOKBACK).values
        
        # C. Scale
        scaled_input = scaler.transform(valid_features)
        X_input = scaled_input.reshape(1, LOOKBACK, len(FEATURES))
        
        # D. Predict Scaled Price
        pred_scaled = model.predict(X_input, verbose=0)[0][0]
        
        # E. Inverse Transform (Trick: Create dummy array to inverse transform only column 0)
        dummy = np.zeros((1, len(FEATURES)))
        dummy[0, 0] = pred_scaled
        pred_price = scaler.inverse_transform(dummy)[0][0]
        
        # F. Append to history
        last_date += timedelta(days=1)
        
        new_row = pd.DataFrame({
            "date": [last_date],
            "price": [pred_price],
            # Fill others with NaN initially, they get recalc'd in next loop step A
            "ma_7": [np.nan], "ma_30": [np.nan], "volatility": [np.nan], "rsi": [np.nan]
        })
        
        history_df = pd.concat([history_df, new_row], ignore_index=True)
        
        future_predictions.append({"date": last_date, "predicted_price": pred_price})
    
    return pd.DataFrame(future_predictions)

# ================= UI LAYOUT =================
st.title("⚡ Bitcoin Price Prediction")
st.markdown("""
<style>
    .metric-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #4F4F4F;
    }
</style>
""", unsafe_allow_html=True)

# 1. Load
with st.spinner("Loading Data..."):
    df = load_data()

if df.empty:
    st.error("Could not load data. Please check connection.")
    st.stop()

# 2. Train
with st.spinner("Training GRU Model (this may take a moment)..."):
    model, scaler = train_gru_model(df)

if not model:
    st.error("Not enough data to train.")
    st.stop()

# 3. Predict
future_df = predict_recursive(model, scaler, df, FUTURE_DAYS)

# 4. Display Stats
current_price = df["price"].iloc[-1]
pred_price = future_df["predicted_price"].iloc[-1]
delta = ((pred_price - current_price) / current_price) * 100
color = "normal" if delta == 0 else ("inverse" if delta < 0 else "normal")

col1, col2, col3 = st.columns(3)
col1.metric("Current Price", f"${current_price:,.2f}")
col2.metric(f"Price in {FUTURE_DAYS} Days", f"${pred_price:,.2f}")
col3.metric("Projected ROI", f"{delta:+.2f}%", delta_color=color)

# 5. Visualization
st.subheader("Forecast Visualization")

# Combine for plotting
hist_data = df.tail(90)[["date", "price"]].copy()
hist_data["Type"] = "Historical"
hist_data.rename(columns={"price": "Value"}, inplace=True)

fut_data = future_df.copy()
fut_data["Type"] = "Forecast"
fut_data.rename(columns={"predicted_price": "Value"}, inplace=True)

# Add a connecting line (last hist point to first future point)
connector = pd.DataFrame([{
    "date": hist_data["date"].iloc[-1],
    "Value": hist_data["Value"].iloc[-1],
    "Type": "Forecast"
}])
fut_data = pd.concat([connector, fut_data], ignore_index=True)

plot_df = pd.concat([hist_data, fut_data], ignore_index=True)

fig = px.line(
    plot_df, 
    x="date", 
    y="Value", 
    color="Type",
    color_discrete_map={"Historical": "cyan", "Forecast": "orange"},
    title="BTC-USD Prediction"
)
fig.update_layout(xaxis_title="Date", yaxis_title="Price (USD)", hovermode="x unified")

st.plotly_chart(fig, use_container_width=True)

with st.expander("Show Raw Data"):
    st.dataframe(future_df)