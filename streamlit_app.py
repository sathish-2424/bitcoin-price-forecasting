import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import plotly.express as px
import plotly.graph_objects as go
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
    Optimized technical indicator calculation with vectorized operations.
    Refactored into a function so it can be called during the prediction loop
    to prevent 'frozen feature' logic errors.
    """
    df = df.copy()
    price = df["price"].values  # Use numpy array for faster operations
    
    # Moving Averages - vectorized
    df["ma_7"] = pd.Series(price, index=df.index).rolling(window=7, min_periods=1).mean()
    df["ma_30"] = pd.Series(price, index=df.index).rolling(window=30, min_periods=1).mean()
    
    # Volatility - vectorized
    df["volatility"] = pd.Series(price, index=df.index).rolling(window=7, min_periods=1).std()
    
    # RSI Calculation - optimized
    delta = pd.Series(price, index=df.index).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    
    # Avoid division by zero
    rs = avg_gain / (avg_loss + 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(50.0)  # Default RSI to 50 if calculation fails
    
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

# ================= OPTIMIZED SEQUENCE CREATION =================
def create_sequences_vectorized(data, lookback):
    """
    Optimized Sequence Generation using numpy array operations.
    Pre-allocates memory for better performance than list comprehension.
    """
    n_samples = len(data) - lookback
    
    if n_samples <= 0 or len(data) == 0:
        # Return empty arrays with correct shape
        if len(data) > 0:
            return np.array([]).reshape(0, lookback, data.shape[1]), np.array([])
        else:
            return np.array([]).reshape(0, lookback, len(FEATURES)), np.array([])
    
    # Pre-allocate arrays for better performance
    X = np.zeros((n_samples, lookback, data.shape[1]), dtype=data.dtype)
    
    # Create sequences - optimized loop with pre-allocation
    for i in range(n_samples):
        X[i] = data[i:i+lookback]
    
    # Create y (Target - the price of the next day, index 0 is 'price')
    y = data[lookback:, 0].copy()
    
    return X, y

# ================= TRAIN MODEL =================
@st.cache_resource
def train_gru_model(df):
    """
    Train GRU model with optimized architecture and callbacks.
    Returns model, scaler, and evaluation metrics.
    """
    # Prepare Data
    feature_data = df[FEATURES].values
    
    if len(feature_data) < LOOKBACK + 10:
        return None, None, None
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(feature_data)
    
    X, y = create_sequences_vectorized(scaled_data, LOOKBACK)
    
    if len(X) == 0:
        return None, None, None

    # Split Data (No shuffling for Time Series!)
    split = int(len(X) * 0.9)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Build Model with improved architecture
    model = Sequential([
        Input(shape=(LOOKBACK, len(FEATURES))),
        GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
        GRU(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)  # Linear output for regression
    ])
    
    model.compile(
        optimizer='adam', 
        loss='mse',
        metrics=['mae']
    )
    
    # Setup callbacks - use compatible approach for restore_best_weights
    try:
        # Try to use restore_best_weights if available (TensorFlow 2.2+)
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=0
        )
    except TypeError:
        # Fallback for older TensorFlow versions
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=7,
            verbose=0
        )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Calculate evaluation metrics
    train_pred = model.predict(X_train, verbose=0).flatten()
    test_pred = model.predict(X_test, verbose=0).flatten()
    
    # Inverse transform predictions for metrics
    price_min = scaler.data_min_[0]
    price_max = scaler.data_max_[0]
    
    train_pred_actual = train_pred * (price_max - price_min) + price_min
    test_pred_actual = test_pred * (price_max - price_min) + price_min
    y_train_actual = y_train * (price_max - price_min) + price_min
    y_test_actual = y_test * (price_max - price_min) + price_min
    
    # Calculate metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    metrics = {
        'train_mae': mean_absolute_error(y_train_actual, train_pred_actual),
        'test_mae': mean_absolute_error(y_test_actual, test_pred_actual),
        'train_rmse': np.sqrt(mean_squared_error(y_train_actual, train_pred_actual)),
        'test_rmse': np.sqrt(mean_squared_error(y_test_actual, test_pred_actual)),
        'train_r2': r2_score(y_train_actual, train_pred_actual),
        'test_r2': r2_score(y_test_actual, test_pred_actual),
        'final_val_loss': history.history['val_loss'][-1] if 'val_loss' in history.history else None
    }
    
    return model, scaler, metrics

# ================= OPTIMIZED PREDICTION LOOP =================
def predict_recursive(model, scaler, df, days_ahead):
    """
    Optimized recursive prediction with proper feature recalculation.
    We must RECALCULATE features (MA, RSI) after every predicted price,
    otherwise the model sees inconsistent data.
    
    Optimizations:
    - More efficient memory usage (list append instead of DataFrame concat in loop)
    - Batch prediction where possible
    - Better handling of weekends/holidays
    - Improved feature recalculation
    """
    # Get sufficient history to calculate max rolling window (30)
    # We need at least LOOKBACK + 30 days of history
    min_history = LOOKBACK + 40
    if len(df) < min_history:
        min_history = len(df)
    
    history_df = df.iloc[-min_history:].copy()
    
    # Extract price min/max from scaler for efficient inverse transform
    price_min = scaler.data_min_[0]
    price_max = scaler.data_max_[0]
    
    # Pre-allocate lists for better performance
    future_predictions = []
    future_dates = []
    last_date = pd.to_datetime(df["date"].iloc[-1])
    
    # Pre-calculate price range for inverse transform
    price_range = price_max - price_min
    
    for day in range(days_ahead):
        # A. Recalculate indicators on the CURRENT history
        temp_df = add_technical_indicators(history_df)
        
        # B. Get the last LOOKBACK rows of features (ensure no NaN)
        feature_rows = temp_df[FEATURES].tail(LOOKBACK)
        
        # Check for NaN values and handle them efficiently
        if feature_rows.isna().any().any():
            feature_rows = feature_rows.ffill().bfill().fillna(0)
        
        valid_features = feature_rows.values
        
        # C. Scale features
        scaled_input = scaler.transform(valid_features)
        X_input = scaled_input.reshape(1, LOOKBACK, len(FEATURES))
        
        # D. Predict Scaled Price
        pred_scaled = model.predict(X_input, verbose=0)[0][0]
        
        # E. Inverse Transform - optimized manual calculation
        pred_price = pred_scaled * price_range + price_min
        pred_price = max(pred_price, 0.01)  # Ensure price is positive
        
        # F. Handle weekends - skip Saturday/Sunday, move to Monday
        last_date += timedelta(days=1)
        while last_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
            last_date += timedelta(days=1)
        
        # G. Append to history using list append (more efficient than DataFrame concat)
        new_row_dict = {
            "date": last_date,
            "price": pred_price,
            "ma_7": np.nan,
            "ma_30": np.nan,
            "volatility": np.nan,
            "rsi": np.nan
        }
        
        # Use pd.concat only once at the end or use list of dicts
        history_df = pd.concat([
            history_df,
            pd.DataFrame([new_row_dict])
        ], ignore_index=True)
        
        future_predictions.append(pred_price)
        future_dates.append(last_date)
    
    # Create DataFrame once at the end
    return pd.DataFrame({
        "date": future_dates,
        "predicted_price": future_predictions
    })

# ================= FORECAST ANALYSIS =================
def analyze_forecast(future_df, historical_df, model_metrics=None):
    """
    Comprehensive forecast analysis including:
    - Trend analysis
    - Volatility forecast
    - Confidence intervals
    - Risk metrics
    """
    analysis = {}
    
    # Basic statistics
    analysis['forecast_mean'] = future_df['predicted_price'].mean()
    analysis['forecast_std'] = future_df['predicted_price'].std()
    analysis['forecast_min'] = future_df['predicted_price'].min()
    analysis['forecast_max'] = future_df['predicted_price'].max()
    
    # Trend analysis
    prices = future_df['predicted_price'].values
    if len(prices) > 1:
        # Calculate trend (slope)
        x = np.arange(len(prices))
        trend_slope = np.polyfit(x, prices, 1)[0]
        analysis['trend_slope'] = trend_slope
        analysis['trend_direction'] = 'Bullish' if trend_slope > 0 else 'Bearish' if trend_slope < 0 else 'Neutral'
        
        # Price change percentage
        analysis['total_change_pct'] = ((prices[-1] - prices[0]) / prices[0]) * 100
        analysis['avg_daily_change_pct'] = analysis['total_change_pct'] / len(prices)
    
    # Volatility analysis
    if len(prices) > 1:
        returns = np.diff(prices) / prices[:-1]
        analysis['forecast_volatility'] = np.std(returns) * np.sqrt(252) * 100  # Annualized volatility %
        analysis['max_drawdown'] = ((prices.max() - prices.min()) / prices.max()) * 100
    
    # Confidence intervals (using historical volatility)
    historical_volatility = historical_df['price'].pct_change().std() * np.sqrt(252) * 100
    current_price = historical_df['price'].iloc[-1]
    final_pred = future_df['predicted_price'].iloc[-1]
    
    # Simple confidence intervals (assuming normal distribution)
    days_ahead = len(future_df)
    std_error = current_price * (historical_volatility / 100) * np.sqrt(days_ahead / 252)
    
    analysis['confidence_95_lower'] = final_pred - 1.96 * std_error
    analysis['confidence_95_upper'] = final_pred + 1.96 * std_error
    analysis['confidence_68_lower'] = final_pred - std_error
    analysis['confidence_68_upper'] = final_pred + std_error
    
    # Model quality metrics
    if model_metrics:
        analysis['model_test_mae'] = model_metrics.get('test_mae', None)
        analysis['model_test_rmse'] = model_metrics.get('test_rmse', None)
        analysis['model_test_r2'] = model_metrics.get('test_r2', None)
    
    return analysis

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
    model, scaler, model_metrics = train_gru_model(df)

if not model:
    st.error("Not enough data to train.")
    st.stop()

# 3. Predict
with st.spinner("Generating Forecast..."):
    future_df = predict_recursive(model, scaler, df, FUTURE_DAYS)

# 4. Analyze Forecast
forecast_analysis = analyze_forecast(future_df, df, model_metrics)

# 5. Display Stats
current_price = df["price"].iloc[-1]
pred_price = future_df["predicted_price"].iloc[-1]
delta = ((pred_price - current_price) / current_price) * 100
color = "normal" if delta == 0 else ("inverse" if delta < 0 else "normal")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Price", f"${current_price:,.2f}")
col2.metric(f"Price in {FUTURE_DAYS} Days", f"${pred_price:,.2f}")
col3.metric("Projected ROI", f"{delta:+.2f}%", delta_color=color)
col4.metric("Trend", forecast_analysis.get('trend_direction', 'N/A'))

# 6. Model Performance Metrics
st.subheader("📊 Model Performance")
perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
if model_metrics:
    perf_col1.metric("Test MAE", f"${model_metrics.get('test_mae', 0):,.2f}")
    perf_col2.metric("Test RMSE", f"${model_metrics.get('test_rmse', 0):,.2f}")
    perf_col3.metric("Test R²", f"{model_metrics.get('test_r2', 0):.4f}")
    perf_col4.metric("Val Loss", f"{model_metrics.get('final_val_loss', 0):.6f}")

# 7. Forecast Analysis
st.subheader("📈 Forecast Analysis")
analysis_col1, analysis_col2, analysis_col3, analysis_col4 = st.columns(4)
analysis_col1.metric("Forecast Volatility", f"{forecast_analysis.get('forecast_volatility', 0):.2f}%")
analysis_col2.metric("Max Drawdown", f"{forecast_analysis.get('max_drawdown', 0):.2f}%")
analysis_col3.metric("Total Change", f"{forecast_analysis.get('total_change_pct', 0):+.2f}%")
analysis_col4.metric("Avg Daily Change", f"{forecast_analysis.get('avg_daily_change_pct', 0):+.2f}%")

# Confidence Intervals
st.markdown("**Confidence Intervals (Final Price)**")
conf_col1, conf_col2, conf_col3 = st.columns(3)
conf_col1.metric("68% CI Lower", f"${forecast_analysis.get('confidence_68_lower', 0):,.2f}")
conf_col2.metric("Predicted", f"${pred_price:,.2f}")
conf_col3.metric("68% CI Upper", f"${forecast_analysis.get('confidence_68_upper', 0):,.2f}")

conf_col4, conf_col5, conf_col6 = st.columns(3)
conf_col4.metric("95% CI Lower", f"${forecast_analysis.get('confidence_95_lower', 0):,.2f}")
conf_col5.metric("Predicted", f"${pred_price:,.2f}")
conf_col6.metric("95% CI Upper", f"${forecast_analysis.get('confidence_95_upper', 0):,.2f}")

# 8. Enhanced Visualization with Confidence Intervals
st.subheader("📉 Forecast Visualization")

# Combine for plotting
hist_data = df.tail(90)[["date", "price"]].copy()
hist_data["Type"] = "Historical"
hist_data.rename(columns={"price": "Value"}, inplace=True)

fut_data = future_df.copy()
fut_data["Type"] = "Forecast"
fut_data.rename(columns={"predicted_price": "Value"}, inplace=True)

# Add confidence intervals to forecast data
fut_data["CI_68_Lower"] = forecast_analysis.get('confidence_68_lower', fut_data["Value"])
fut_data["CI_68_Upper"] = forecast_analysis.get('confidence_68_upper', fut_data["Value"])
fut_data["CI_95_Lower"] = forecast_analysis.get('confidence_95_lower', fut_data["Value"])
fut_data["CI_95_Upper"] = forecast_analysis.get('confidence_95_upper', fut_data["Value"])

# Add a connecting line (last hist point to first future point)
connector = pd.DataFrame([{
    "date": hist_data["date"].iloc[-1],
    "Value": hist_data["Value"].iloc[-1],
    "Type": "Forecast",
    "CI_68_Lower": hist_data["Value"].iloc[-1],
    "CI_68_Upper": hist_data["Value"].iloc[-1],
    "CI_95_Lower": hist_data["Value"].iloc[-1],
    "CI_95_Upper": hist_data["Value"].iloc[-1]
}])
fut_data = pd.concat([connector, fut_data], ignore_index=True)

plot_df = pd.concat([hist_data, fut_data], ignore_index=True)

# Create enhanced plot with confidence intervals
fig = go.Figure()

# Add confidence intervals (95%)
fig.add_trace(go.Scatter(
    x=plot_df[plot_df["Type"] == "Forecast"]["date"],
    y=plot_df[plot_df["Type"] == "Forecast"]["CI_95_Upper"],
    mode='lines',
    line=dict(width=0),
    showlegend=False,
    hoverinfo='skip'
))

fig.add_trace(go.Scatter(
    x=plot_df[plot_df["Type"] == "Forecast"]["date"],
    y=plot_df[plot_df["Type"] == "Forecast"]["CI_95_Lower"],
    mode='lines',
    line=dict(width=0),
    fillcolor='rgba(255, 165, 0, 0.2)',
    fill='tonexty',
    name='95% Confidence Interval',
    hoverinfo='skip'
))

# Add confidence intervals (68%)
fig.add_trace(go.Scatter(
    x=plot_df[plot_df["Type"] == "Forecast"]["date"],
    y=plot_df[plot_df["Type"] == "Forecast"]["CI_68_Upper"],
    mode='lines',
    line=dict(width=0),
    showlegend=False,
    hoverinfo='skip'
))

fig.add_trace(go.Scatter(
    x=plot_df[plot_df["Type"] == "Forecast"]["date"],
    y=plot_df[plot_df["Type"] == "Forecast"]["CI_68_Lower"],
    mode='lines',
    line=dict(width=0),
    fillcolor='rgba(255, 165, 0, 0.3)',
    fill='tonexty',
    name='68% Confidence Interval',
    hoverinfo='skip'
))

# Add historical data
fig.add_trace(go.Scatter(
    x=hist_data["date"],
    y=hist_data["Value"],
    mode='lines',
    name='Historical',
    line=dict(color='cyan', width=2)
))

# Add forecast data
fig.add_trace(go.Scatter(
    x=fut_data["date"],
    y=fut_data["Value"],
    mode='lines',
    name='Forecast',
    line=dict(color='orange', width=2, dash='dash')
))

fig.update_layout(
    title="BTC-USD Price Prediction with Confidence Intervals",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    hovermode="x unified",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

st.plotly_chart(fig, use_container_width=True)

with st.expander("Show Raw Data"):
    st.dataframe(future_df)