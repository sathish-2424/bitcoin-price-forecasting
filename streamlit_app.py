import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from model import train_or_load_model, scaler

st.set_page_config(
    page_title="Live Bitcoin Price Prediction",
    layout="centered"
)

# 🔄 AUTO REFRESH (5 minutes)
REFRESH_INTERVAL = 300
time.sleep(REFRESH_INTERVAL)
st.experimental_rerun()

# 📥 Fetch hourly Bitcoin data
def fetch_hourly_data():
    btc = yf.download(
        "BTC-USD",
        period="7d",
        interval="1h",
        progress=False
    )
    btc = btc[['Close']]
    btc.dropna(inplace=True)
    return btc

st.title("🚀 Live Bitcoin Price Prediction (Hourly)")
st.caption("Auto-refresh every 5 minutes | Data: Yahoo Finance")

btc = fetch_hourly_data()

# 🧠 Train or Load Model
model, scaled_data = train_or_load_model(btc[['Close']].values)

# 🔮 Predict next hour price
last_60 = scaled_data[-60:].reshape(1, 60, 1)
prediction = model.predict(last_60)
predicted_price = scaler.inverse_transform(prediction)[0][0]

# 📊 Dashboard
st.metric(
    "Current BTC Price ($)",
    f"{btc['Close'].iloc[-1]:,.2f}"
)

st.metric(
    "Predicted Next Hour Price ($)",
    f"{predicted_price:,.2f}"
)

st.line_chart(btc['Close'])

# Prepare data for plotting
actual_prices = btc['Close'].values
time_index = btc.index

# Add predicted price as last point
predicted_series = list(actual_prices)
predicted_series.append(predicted_price)

predicted_time_index = list(time_index)
predicted_time_index.append(time_index[-1] + pd.Timedelta(hours=1))

# Plot
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(time_index, actual_prices, label="Actual BTC Price", linewidth=2)
ax.plot(
    predicted_time_index[-2:], 
    predicted_series[-2:], 
    linestyle="--",
    marker="o",
    label="Predicted Next Hour Price"
)

ax.set_title("Bitcoin Price Prediction (Hourly)")
ax.set_xlabel("Time")
ax.set_ylabel("Price (USD)")
ax.legend()
ax.grid(True)

# Show in Streamlit
st.pyplot(fig)


st.success("Model loaded from disk (.h5) and prediction updated successfully")