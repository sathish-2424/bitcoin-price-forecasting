# streamlit_app.py
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
PERIOD = "2y"
INTERVAL = "1d"
LOOKBACK = 60
FUTURE_DAYS = 14
FEATURES = ["price", "ma_7", "ma_30", "volatility", "rsi"]
EPOCHS = 25

# ================= HELPER: INDICATOR CALCULATION =================
def add_technical_indicators(df):
    df = df.copy()
    df["ma_7"] = df["price"].rolling(window=7).mean()
    df["ma_30"] = df["price"].rolling(window=30).mean()
    df["volatility"] = df["price"].rolling(window=7).std()

    delta = df["price"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()

    rs = avg_gain / (avg_loss.replace(0, np.nan))
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"].fillna(50, inplace=True)  # neutral for early rows

    return df

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    try:
        df = yf.download(TICKER, period=PERIOD, interval=INTERVAL, progress=False)
        df.reset_index(inplace=True)
        # handle common multiindex issue robustly
        if isinstance(df.columns, pd.MultiIndex):
            # prefer Close column
            try:
                price_col = df["Close"]
                if isinstance(price_col, pd.DataFrame):
                    price_col = price_col.iloc[:, 0]
            except Exception:
                price_col = df.iloc[:, 1]
            date_col = df["Date"] if "Date" in df else df.index
            df_clean = pd.DataFrame({"date": date_col, "price": price_col})
        else:
            df_clean = df[["Date", "Close"]].rename(columns={"Date": "date", "Close": "price"})

        df_clean["price"] = pd.to_numeric(df_clean["price"], errors="coerce")
        df_clean = add_technical_indicators(df_clean)
        df_clean.dropna(inplace=True)
        df_clean.reset_index(drop=True, inplace=True)
        return df_clean
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# ================= DSA OPTIMIZATION: SLIDING WINDOW (vectorized) =================
def create_sequences_vectorized(scaled_features, scaled_price, lookback):
    """
    Vectorized sequence creation using numpy sliding windows.
    scaled_features: (n_samples, n_features)
    scaled_price: (n_samples,) or (n_samples,1)
    Returns X shape (n_windows, lookback, n_features) and y shape (n_windows,)
    """
    n_samples = scaled_features.shape[0]
    n_features = scaled_features.shape[1]
    if n_samples <= lookback:
        return np.empty((0, lookback, n_features)), np.empty((0,))

    try:
        # sliding_window_view is fast and memory-friendly (view)
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(scaled_features, window_shape=lookback, axis=0)
        # windows shape: (n_windows, lookback, n_features)
        X = windows.reshape(-1, lookback, n_features)
    except Exception:
        # fallback (less optimal) for older numpy
        X = np.array([scaled_features[i:i+lookback] for i in range(n_samples - lookback)])

    # y aligns with the window's "next" target (index lookback..end)
    y = scaled_price[lookback:]
    return X, y

# ================= TRAIN MODEL =================
@st.cache_resource
def train_gru_model(df):
    # features (X) and price (y) from df
    feature_data = df[FEATURES].values  # columns in the order of FEATURES
    price_data = df["price"].values.reshape(-1, 1)

    # Scalers: features scaler and price scaler (target)
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler_features.fit_transform(feature_data)

    price_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_price = price_scaler.fit_transform(price_data).reshape(-1)

    X, y = create_sequences_vectorized(scaled_features, scaled_price, LOOKBACK)
    if len(X) == 0:
        return None, None, None

    # train/test split (time-series)
    split_idx = int(len(X) * 0.9)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = Sequential([
        Input(shape=(LOOKBACK, len(FEATURES))),
        GRU(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
        verbose=0
    )
    return model, scaler_features, price_scaler

# ================= OPTIMIZED PREDICTION LOOP (corrected) =================
def predict_recursive(model, scaler_features, price_scaler, df, days_ahead):
    # ensure we have enough history
    history_df = df.iloc[-(LOOKBACK + 40):].copy().reset_index(drop=True)
    future = []
    last_date = pd.to_datetime(history_df["date"].iloc[-1])

    for _ in range(days_ahead):
        # recalc indicators on current history (includes any predicted prices)
        temp = add_technical_indicators(history_df)

        # get last LOOKBACK rows of features (must match FEATURES order)
        window = temp[FEATURES].tail(LOOKBACK).values
        # scale features using feature scaler
        scaled_window = scaler_features.transform(window)
        X_input = scaled_window.reshape(1, LOOKBACK, len(FEATURES))

        # model predicts scaled price (target scaler used during training)
        pred_scaled = model.predict(X_input, verbose=0)[0][0]
        pred_price = price_scaler.inverse_transform(np.array([[pred_scaled]]))[0][0]

        # append to history_df so next loop step recalculates indicators including this prediction
        last_date = last_date + timedelta(days=1)
        new_row = {
            "date": last_date,
            "price": pred_price,
            "ma_7": np.nan,
            "ma_30": np.nan,
            "volatility": np.nan,
            "rsi": np.nan
        }
        history_df = pd.concat([history_df, pd.DataFrame([new_row])], ignore_index=True)
        future.append({"date": last_date, "predicted_price": float(pred_price)})

    return pd.DataFrame(future)

# ================= UI LAYOUT =================
st.title("⚡ Bitcoin Price Prediction")

with st.spinner("Loading Data..."):
    df = load_data()

if df.empty:
    st.error("Could not load data. Please check connection.")
    st.stop()

with st.spinner("Training model..."):
    model, scaler_features, price_scaler = train_gru_model(df)

if model is None:
    st.error("Not enough data to train.")
    st.stop()

future_df = predict_recursive(model, scaler_features, price_scaler, df, FUTURE_DAYS)

# Display metrics
current_price = df["price"].iloc[-1]
pred_price = future_df["predicted_price"].iloc[-1]
delta = ((pred_price - current_price) / current_price) * 100
col1, col2, col3 = st.columns(3)
col1.metric("Current Price", f"${current_price:,.2f}")
col2.metric(f"Price in {FUTURE_DAYS} Days", f"${pred_price:,.2f}")
col3.metric("Projected ROI", f"{delta:+.2f}%")

# Plot
st.subheader("Forecast Visualization")
hist_data = df.tail(90)[["date", "price"]].copy()
hist_data["Type"] = "Historical"
hist_data.rename(columns={"price": "Value"}, inplace=True)

fut_data = future_df.copy()
fut_data["Type"] = "Forecast"
fut_data.rename(columns={"predicted_price": "Value"}, inplace=True)

connector = pd.DataFrame([{
    "date": hist_data["date"].iloc[-1],
    "Value": hist_data["Value"].iloc[-1],
    "Type": "Forecast"
}])
fut_data = pd.concat([connector, fut_data], ignore_index=True)
plot_df = pd.concat([hist_data, fut_data], ignore_index=True)

fig = px.line(plot_df, x="date", y="Value", color="Type",
              color_discrete_map={"Historical": "cyan", "Forecast": "orange"},
              title="BTC-USD Prediction")
fig.update_layout(xaxis_title="Date", yaxis_title="Price (USD)", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

with st.expander("Show Raw Forecast Data"):
    st.dataframe(future_df)