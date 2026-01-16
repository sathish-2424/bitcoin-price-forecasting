import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import plotly.express as px
import warnings

warnings.filterwarnings("ignore")

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Bitcoin Price Prediction",
    layout="wide",
    page_icon="📈"
)

# ================= CONFIG =================
TICKER = "BTC-USD"
INTERVAL = "1d"
PERIOD = "5y"
LOOKBACK = 60
FUTURE_DAYS = 14

FEATURES = ["price", "ma_7", "ma_30", "volatility", "rsi"]

# ================= LOAD DATA =================
@st.cache_data

def load_data():
    df = yf.download(TICKER, period=PERIOD, interval=INTERVAL, progress=False)

    # ===== ROBUST yfinance handling =====
    if isinstance(df.columns, pd.MultiIndex):
        close_data = df.xs("Close", axis=1, level=0)

        # If Series → convert
        if isinstance(close_data, pd.Series):
            df = close_data.to_frame(name="price")
        else:
            # If DataFrame → rename column
            df = close_data.copy()
            df.columns = ["price"]

    else:
        df = df[["Close"]].rename(columns={"Close": "price"})

    # Index → column
    df.reset_index(inplace=True)
    df.rename(columns={"Date": "date", "index": "date"}, inplace=True)

    # ===== FEATURES =====
    df["ma_7"] = df["price"].rolling(7).mean()
    df["ma_30"] = df["price"].rolling(30).mean()
    df["volatility"] = df["price"].rolling(7).std()

    # ===== RSI (SAFE) =====
    delta = df["price"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(14, min_periods=14).mean()
    avg_loss = loss.rolling(14, min_periods=14).mean()

    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # ===== CLEAN =====
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    return df



df = load_data()

# ================= SEQUENCE CREATION =================
def create_sequences(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


# ================= TRAIN MODEL =================
@st.cache_resource
def train_model(df):
    feature_df = df[FEATURES].copy()

    if len(feature_df) <= LOOKBACK:
        raise ValueError("Not enough data to create sequences")

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(feature_df.values)

    X, y = create_sequences(scaled, LOOKBACK)

    if len(X) == 0:
        raise ValueError("Sequence creation failed – empty dataset")

    split = int(len(X) * 0.95)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        GRU(128, return_sequences=True, input_shape=(LOOKBACK, len(FEATURES))),
        Dropout(0.3),
        GRU(64),
        Dropout(0.3),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="huber")

    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
        verbose=0
    )

    return model, scaler

# ================= FUTURE PREDICTION =================
def predict_future(df, model, scaler):
    buffer = df[FEATURES].tail(LOOKBACK).copy()
    future = []
    last_date = df["date"].iloc[-1]

    for _ in range(FUTURE_DAYS):
        scaled_buf = scaler.transform(buffer.values)
        X_input = scaled_buf.reshape(1, LOOKBACK, len(FEATURES))

        pred_scaled = model.predict(X_input, verbose=0)[0][0]

        # Inverse only PRICE
        dummy = np.zeros((1, len(FEATURES)))
        dummy[0, 0] = pred_scaled
        pred_price = scaler.inverse_transform(dummy)[0][0]

        last_date += timedelta(days=1)

        new_row = buffer.iloc[-1].copy()
        new_row[:] = buffer.iloc[-1].values
        new_row["price"] = pred_price

        buffer = pd.concat([buffer, pd.DataFrame([new_row])]).tail(LOOKBACK)

        future.append({
            "date": last_date,
            "predicted_price": float(pred_price)
        })

    return pd.DataFrame(future)

# ================= UI =================
st.title("📈 Bitcoin Price Prediction")

if future_df.empty:
    st.error("❌ Prediction failed: insufficient historical data.")
    st.stop()

last_price = float(df["price"].iloc[-1])
future_price = float(future_df["predicted_price"].iloc[-1])
change_pct = ((future_price - last_price) / last_price) * 100

c1, c2, c3 = st.columns(3)
c1.metric("Last Price", f"${last_price:,.0f}")
c2.metric("14-Day Forecast", f"${future_price:,.0f}")
c3.metric("Expected Change", f"{change_pct:.2f}%")


plot_df = pd.concat([
    df[["date", "price"]].rename(columns={"price": "value"}).assign(Type="Actual"),
    future_df.rename(columns={"predicted_price": "value"}).assign(Type="Forecast")
])

fig = px.line(
    plot_df,
    x="date",
    y="value",
    color="Type",
    title="Bitcoin Price — Actual vs Forecast",
    labels={"value": "Price (USD)", "date": "Date"}
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("📅 Future Price Predictions")
st.dataframe(
    future_df.assign(predicted_price=lambda x: x["predicted_price"].round(2)),
    use_container_width=True
)
