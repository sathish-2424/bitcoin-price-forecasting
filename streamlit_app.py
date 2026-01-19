# =========================================================
# Bitcoin Price Prediction Dashboard
# LSTM + Streamlit (Full Version with Extra Charts)
# =========================================================

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Bitcoin Price Prediction",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Bitcoin Price Prediction")

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.header("⚙️ Settings")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
lookback = st.sidebar.slider("Lookback Days", 30, 120, 60)
epochs = st.sidebar.slider("Training Epochs", 5, 50, 10)

# ---------------------------------------------------------
# Load Data
# ---------------------------------------------------------
@st.cache_data
def load_data(start, end):
    df = yf.download("BTC-USD", start=start, end=end, interval="1d")
    return df[["Close"]]

data = load_data(start_date, end_date)

# ---------------------------------------------------------
# Display Price Chart
# ---------------------------------------------------------
st.subheader("📊 Historical Bitcoin Prices")
st.line_chart(data)

# ---------------------------------------------------------
# Technical Indicators
# ---------------------------------------------------------
data["MA20"] = data["Close"].rolling(window=20).mean()
data["MA50"] = data["Close"].rolling(window=50).mean()

# RSI Function
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

data["RSI"] = calculate_rsi(data["Close"])
data["Daily Change (%)"] = data["Close"].pct_change() * 100

# ---------------------------------------------------------
# Data Preprocessing
# ---------------------------------------------------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[["Close"]])

X, y = [], []
for i in range(lookback, len(scaled_data)):
    X.append(scaled_data[i - lookback:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
X = X.reshape(X.shape[0], X.shape[1], 1)

# ---------------------------------------------------------
# LSTM Model
# ---------------------------------------------------------
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")

with st.spinner("🔄 Training LSTM Model..."):
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)

st.success("✅ Model Training Completed")

# ---------------------------------------------------------
# Next Day Prediction
# ---------------------------------------------------------
last_sequence = scaled_data[-lookback:]
last_sequence = last_sequence.reshape(1, lookback, 1)

prediction = model.predict(last_sequence)
predicted_price = scaler.inverse_transform(prediction)

# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------
st.subheader("🔮 Prediction Summary")

col1, col2 = st.columns(2)
current_price = data.iloc[-1]["Close"]
price_change = predicted_price[0][0] - current_price

col1.metric("Current Price", f"${current_price:,.2f}")
col2.metric("Predicted Next Day Price", f"${predicted_price[0][0]:,.2f}",
            f"{price_change:,.2f}")

# ---------------------------------------------------------
# Moving Average Chart
# ---------------------------------------------------------
st.subheader("📊 Moving Average Analysis")

plt.figure(figsize=(12,5))
plt.plot(data["Close"], label="BTC Close")
plt.plot(data["MA20"], label="MA 20", linestyle="--")
plt.plot(data["MA50"], label="MA 50", linestyle="--")
plt.legend()
st.pyplot(plt)

# ---------------------------------------------------------
# RSI Chart
# ---------------------------------------------------------
st.subheader("📐 RSI Indicator")

plt.figure(figsize=(12,4))
plt.plot(data["RSI"], label="RSI")
plt.axhline(70, linestyle="--")
plt.axhline(30, linestyle="--")
plt.legend()
st.pyplot(plt)

# ---------------------------------------------------------
# Daily Volatility Chart
# ---------------------------------------------------------
st.subheader("📉 Daily Price Change (%)")
st.line_chart(data["Daily Change (%)"])

# ---------------------------------------------------------
# Actual vs Predicted Chart
# ---------------------------------------------------------
st.subheader("🤖 Actual vs Predicted Prices")

train_predictions = model.predict(X)
train_predictions = scaler.inverse_transform(train_predictions)
actual_prices = scaler.inverse_transform(y.reshape(-1,1))

plt.figure(figsize=(12,5))
plt.plot(actual_prices, label="Actual")
plt.plot(train_predictions, label="Predicted")
plt.legend()
st.pyplot(plt)

# ---------------------------------------------------------
# 7-Day Future Prediction
# ---------------------------------------------------------
st.subheader("📅 7-Day Future Forecast")

future_days = 7
temp_input = list(last_sequence.flatten())
future_predictions = []

for _ in range(future_days):
    x_input = np.array(temp_input[-lookback:])
    x_input = x_input.reshape(1, lookback, 1)
    yhat = model.predict(x_input, verbose=0)
    temp_input.append(yhat[0][0])
    future_predictions.append(yhat[0][0])

future_predictions = scaler.inverse_transform(
    np.array(future_predictions).reshape(-1,1)
)

future_df = pd.DataFrame({
    "Day": range(1, future_days+1),
    "Predicted Price ($)": future_predictions.flatten()
})

st.table(future_df)

# Confidence Band
upper = future_predictions * 1.05
lower = future_predictions * 0.95

plt.figure(figsize=(10,4))
plt.plot(future_predictions, label="Prediction")
plt.fill_between(range(future_days),
                 lower.flatten(),
                 upper.flatten(),
                 alpha=0.3)
plt.legend()
st.pyplot(plt)

# ---------------------------------------------------------
# Last 10 Days Table
# ---------------------------------------------------------
st.subheader("📋 Last 10 Days Data")
st.dataframe(data.tail(10))

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown("🚀 **Built with LSTM, Yahoo Finance & Streamlit**")
