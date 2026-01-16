import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import Huber
from tensorflow.keras import backend as K
import plotly.express as px
import warnings

warnings.filterwarnings("ignore")

# ================= PAGE CONFIG =================
st.set_page_config(page_title="BTC Forecast — LogReturn + GRU", layout="wide", page_icon="⚡")

# ================= DEFAULT CONFIG (tweakable in sidebar) =================
TICKER = "BTC-USD"
PERIOD = "2y"
INTERVAL = "1d"

# Sidebar controls
st.sidebar.title("Model & Forecast Settings")
LOOKBACK = st.sidebar.number_input("Lookback (days)", value=30, min_value=5, max_value=180, step=5)
FUTURE_DAYS = st.sidebar.number_input("Forecast days (business days)", value=14, min_value=1, max_value=60, step=1)
EPOCHS = st.sidebar.number_input("Epochs", value=40, min_value=1, max_value=500, step=5)
BATCH_SIZE = st.sidebar.selectbox("Batch size", [16, 32, 64], index=1)
PERIOD_INPUT = st.sidebar.selectbox("History period", ["1y", "2y", "3y", "5y"], index=1)

# Features used by the model (log_return must be first)
FEATURES = ["log_return", "ma_7", "ma_30", "volatility", "rsi"]

# ================= TECHNICAL INDICATORS =================
def add_technical_indicators(df):
    df = df.copy()
    # Ensure price exists
    if "price" not in df.columns:
        raise ValueError("DataFrame must include 'price' column")

    # Moving averages
    df["ma_7"] = df["price"].rolling(window=7, min_periods=1).mean()
    df["ma_30"] = df["price"].rolling(window=30, min_periods=1).mean()

    # Volatility (7-day std)
    df["volatility"] = df["price"].rolling(window=7, min_periods=1).std().fillna(0.0)

    # RSI (14)
    delta = df["price"].diff().fillna(0.0)
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(50.0)

    # Log return (target) — computed from price
    df["log_return"] = np.log(df["price"] / df["price"].shift(1))
    df["log_return"].fillna(0.0, inplace=True)

    return df

# ================= DATA LOADING =================
@st.cache_data
def load_data(ticker=TICKER, period=PERIOD_INPUT, interval=INTERVAL):
    try:
        raw = yf.download(ticker, period=period, interval=interval, progress=False)
        if raw is None or raw.empty:
            return pd.DataFrame()
        raw = raw.reset_index()
        # Normalize to 'date' + 'price'
        if "Close" in raw.columns:
            df = raw[["Date", "Close"]].rename(columns={"Date": "date", "Close": "price"})
        else:
            cols = list(raw.columns)
            df = raw[[cols[0], cols[1]]].rename(columns={cols[0]: "date", cols[1]: "price"})
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df.dropna(subset=["price"], inplace=True)
        df = add_technical_indicators(df)
        # drop initial NaNs (if any)
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    except Exception as e:
        st.error(f"Data load error: {e}")
        return pd.DataFrame()

# ================= SEQUENCE CREATION (vectorized) =================
def create_sequences_vectorized(combined_scaled, lookback):
    """
    combined_scaled: np.ndarray shape (N, F) where first column is scaled target (log_return)
    returns X (n_samples, lookback, F) and y (n_samples,) where y is next-step scaled target
    """
    N = combined_scaled.shape[0]
    F = combined_scaled.shape[1]
    if N <= lookback:
        return np.empty((0, lookback, F), dtype=np.float32), np.empty((0,), dtype=np.float32)

    try:
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(combined_scaled, window_shape=lookback, axis=0)
        # windows shape: (N - lookback + 1, lookback, F)
        # we want windows[ : N - lookback ] to align with targets at index lookback: (N - lookback)
        X = windows[:-1].astype(np.float32)
    except Exception:
        # fallback
        n_samples = N - lookback
        X = np.zeros((n_samples, lookback, F), dtype=np.float32)
        for i in range(n_samples):
            X[i] = combined_scaled[i:i+lookback]
    # y: scaled target at positions lookback..end
    y = combined_scaled[lookback:, 0].astype(np.float32)
    return X, y

# ================= TRAINING FUNCTION =================
@st.cache_resource
def train_model(df, lookback=LOOKBACK, epochs=EPOCHS, batch_size=BATCH_SIZE):
    # Prepare raw arrays
    feature_df = df[["log_return", "ma_7", "ma_30", "volatility", "rsi"]].copy()
    if feature_df.shape[0] < lookback + 10:
        return None, None, None, None

    # Split price/target scaler and other features scaler
    price_scaler = MinMaxScaler()                # for log_return only
    feature_scaler = MinMaxScaler()              # for other technicals

    # Fit scalers
    price_vals = feature_df[["log_return"]].values.astype(np.float32)
    other_vals = feature_df[["ma_7", "ma_30", "volatility", "rsi"]].values.astype(np.float32)

    price_scaled = price_scaler.fit_transform(price_vals)
    other_scaled = feature_scaler.fit_transform(other_vals)

    combined_scaled = np.hstack([price_scaled, other_scaled])

    # Create sequences
    X, y = create_sequences_vectorized(combined_scaled, lookback)
    if len(X) == 0:
        return None, None, None, None

    # time-series split (no shuffle)
    split = int(len(X) * 0.9)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # clear TF session to reduce memory on reruns
    K.clear_session()

    # Model (smaller but effective)
    model = Sequential([
        Input(shape=(lookback, combined_scaled.shape[1])),
        GRU(64, return_sequences=True),
        GRU(32, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dropout(0.1),
        Dense(1, activation="linear")
    ])
    model.compile(optimizer="adam", loss=Huber(delta=1.0), metrics=["mae"])

    early_stopping = EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, verbose=0)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", patience=4, factor=0.5, min_lr=1e-6, verbose=0)

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs,
              batch_size=batch_size, callbacks=[early_stopping, reduce_lr], verbose=0)

    return model, price_scaler, feature_scaler, combined_scaled.shape[1]

# ================= RECURSIVE PREDICTION (recalculates indicators each step) =================
def predict_recursive(model, price_scaler, feature_scaler, df, days_ahead, lookback=LOOKBACK):
    """
    Recursive forecast:
    - builds inputs from unscaled features each loop
    - scales log_return separately and other features separately then hstacks
    - reconstructs price from predicted log return: price_next = last_price * exp(pred_log_return)
    """
    if model is None:
        return pd.DataFrame()

    # Keep a small history window
    max_rolling = 30
    min_history = max(lookback + 5, lookback + max_rolling)
    history = df.iloc[-min_history:].copy().reset_index(drop=True)

    last_price = float(df["price"].iloc[-1])
    future_rows = []
    last_date = pd.to_datetime(df["date"].iloc[-1])

    for i in range(days_ahead):
        # Recalculate indicators on current history
        temp = add_technical_indicators(history)

        # Take last lookback rows of raw (unscaled) features
        feat = temp[FEATURES].tail(lookback).copy()

        # Fill any NaNs sensibly
        if feat.isna().any().any():
            feat = feat.ffill().bfill().fillna(0.0)

        # Separate columns for scalers
        price_col = feat[["log_return"]].values.astype(np.float32)
        other_cols = feat[["ma_7", "ma_30", "volatility", "rsi"]].values.astype(np.float32)

        # Scale separately and combine
        price_col_scaled = price_scaler.transform(price_col)
        other_cols_scaled = feature_scaler.transform(other_cols)
        scaled_input = np.hstack([price_col_scaled, other_cols_scaled]).astype(np.float32)

        X_input = scaled_input.reshape(1, lookback, scaled_input.shape[1])

        pred_scaled = float(model.predict(X_input, verbose=0)[0][0])
        # invert scale to get predicted log return
        pred_log_return = price_scaler.inverse_transform(np.array([[pred_scaled]]))[0][0]

        # reconstruct price
        pred_price = last_price * float(np.exp(pred_log_return))
        pred_price = float(max(pred_price, 0.01))

        # advance date skipping weekends
        next_date = last_date + timedelta(days=1)
        while next_date.weekday() >= 5:
            next_date += timedelta(days=1)
        last_date = next_date

        # append to history for next iteration
        new_row = {
            "date": last_date,
            "price": pred_price,
            # add log_return so next loop can use it as an input (unscaled)
            "log_return": pred_log_return,
            "ma_7": np.nan,
            "ma_30": np.nan,
            "volatility": np.nan,
            "rsi": np.nan
        }
        history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)
        last_price = pred_price

        future_rows.append({"date": last_date, "predicted_price": pred_price, "pred_log_return": pred_log_return})

    return pd.DataFrame(future_rows)

# ================= UI + Run =================
st.title("⚡ BTC Forecast — Log-Return GRU (Accuracy-focused)")

with st.spinner("Loading historical data..."):
    df = load_data(period=PERIOD_INPUT)

if df.empty:
    st.error("No data loaded. Try changing the history period or check your connection.")
    st.stop()

st.markdown(f"**Data loaded:** {len(df)} rows — last date: {df['date'].iloc[-1].date()} — last price: ${df['price'].iloc[-1]:,.2f}")

with st.spinner("Training model (this can take a little while)..."):
    model, price_scaler, feature_scaler, feature_count = train_model(df, lookback=LOOKBACK, epochs=EPOCHS, batch_size=BATCH_SIZE)

if model is None:
    st.error("Not enough data to train reliably with current settings (increase history or reduce lookback).")
    st.stop()

with st.spinner("Generating forecast..."):
    future_df = predict_recursive(model, price_scaler, feature_scaler, df, days_ahead=FUTURE_DAYS, lookback=LOOKBACK)

if future_df.empty:
    st.error("Forecast failed or returned no data.")
    st.stop()

# Metrics
current_price = float(df["price"].iloc[-1])
pred_price = float(future_df["predicted_price"].iloc[-1])
delta_pct = ((pred_price - current_price) / current_price) * 100.0

col1, col2, col3 = st.columns(3)
col1.metric("Current Price", f"${current_price:,.2f}")
col2.metric(f"Predicted Price ({FUTURE_DAYS} days)", f"${pred_price:,.2f}")
col3.metric("Projected change", f"{delta_pct:+.2f}%")

# Visualization
st.subheader("Prediction Chart")
hist_data = df[["date", "price"]].copy().tail(180).rename(columns={"price": "Value"})
hist_data["Type"] = "Historical"

fut_plot = future_df[["date", "predicted_price"]].copy().rename(columns={"predicted_price": "Value"})
fut_plot["Type"] = "Forecast"

# connector to visually link last historical to first forecast
connector = pd.DataFrame([{
    "date": hist_data["date"].iloc[-1],
    "Value": hist_data["Value"].iloc[-1],
    "Type": "Forecast"
}])
fut_plot = pd.concat([connector, fut_plot], ignore_index=True)

plot_df = pd.concat([hist_data, fut_plot], ignore_index=True)

fig = px.line(plot_df, x="date", y="Value", color="Type", title="BTC Price Forecast (log-return model)")
fig.update_layout(xaxis_title="Date", yaxis_title="Price (USD)", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

with st.expander("Show forecast table"):
    st.dataframe(future_df.reset_index(drop=True))

with st.expander("Model & scalers info"):
    st.write("Model summary:")
    model.summary(print_fn=lambda s: st.text(s))
    st.write("Price scaler min/max (log_return):", getattr(price_scaler, "data_min_", None), getattr(price_scaler, "data_max_", None))
    st.write("Feature scaler min/max:", getattr(feature_scaler, "data_min_", None), getattr(feature_scaler, "data_max_", None))

st.markdown("""
### Notes & next steps
- This model predicts **log returns**, which improves stationarity and typically leads to better generalization.
- Consider **walk-forward retraining** in a live setting (train on expanding window or retrain regularly).
- To estimate uncertainty, consider MC-dropout, or ensemble several trained models and show quantiles.
- If you want, I can:
  - add walk-forward retraining and rolling evaluation,
  - add prediction intervals,
  - convert model to directly predict multi-day sequences (non-recursive).
""")
