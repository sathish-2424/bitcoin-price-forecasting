# =====================================================
# Bitcoin Price Forecasting (XGBoost – FINAL SAFE)
# =====================================================

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import timedelta
from xgboost import XGBRegressor
import plotly.express as px
import warnings
import os

warnings.filterwarnings("ignore")

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Bitcoin Price Prediction",
    layout="wide",
    page_icon="⚡"
)

# ================= CONFIG =================
TICKER = "BTC-USD"
PERIOD = "5y"
INTERVAL = "1d"
FUTURE_DAYS = 14
MIN_ROWS = 40
BACKUP_FILE = "btc_backup.csv"

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
    df = df.copy()

    df["price_lag1"] = df["price"].shift(1)
    df["price_lag7"] = df["price"].shift(7)

    df["ma_7"] = df["price"].rolling(7).mean()
    df["ma_30"] = df["price"].rolling(30).mean()
    df["volatility"] = df["price"].rolling(7).std()

    delta = df["price"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)

    df["rsi"] = 100 - (100 / (1 + rs))
    return df

# ================= LOAD DATA (YAHOO + CSV SAFE) =================
@st.cache_data
def load_data():
    df = pd.DataFrame()

    # ---- Try Yahoo Finance ----
    try:
        df = yf.download(
            TICKER,
            period=PERIOD,
            interval=INTERVAL,
            progress=False,
            auto_adjust=False
        )
    except Exception:
        df = pd.DataFrame()

    # ---- Fallback to CSV ----
    if df.empty and os.path.exists(BACKUP_FILE):
        st.warning("Yahoo Finance unavailable. Using cached BTC data.")
        df = pd.read_csv(BACKUP_FILE)

    if df.empty:
        return pd.DataFrame()

    # ---- Normalize columns safely ----
    df = df.reset_index(drop=False)

    df.columns = [
        str(col).strip().lower().replace(" ", "_")
        for col in df.columns
    ]

    # ---- Ensure date column exists ----
    if "date" not in df.columns:
        if "timestamp" in df.columns:
            df = df.rename(columns={"timestamp": "date"})
        else:
            return pd.DataFrame()

    # ---- Ensure close price exists ----
    if "close" not in df.columns:
        return pd.DataFrame()

    df = df[["date", "close"]].rename(columns={"close": "price"})
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df = add_technical_indicators(df)
    df["target"] = df["price"].shift(-1)

    df = df.dropna().reset_index(drop=True)
    return df

# ================= TRAIN MODEL =================
@st.cache_resource
def train_model(df):
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    model.fit(df[FEATURES], df["target"])
    return model

# ================= FUTURE PREDICTION =================
def predict_future(df, model, days):
    history = df[["date", "price"]].copy()
    future = []

    last_date = pd.to_datetime(history["date"].iloc[-1])

    for _ in range(days):
        history = add_technical_indicators(history)
        X = history[FEATURES].ffill().bfill().iloc[-1:]

        price = float(model.predict(X)[0])
        price = max(price, 1.0)

        last_date += timedelta(days=1)
        while last_date.weekday() >= 5:
            last_date += timedelta(days=1)

        history = pd.concat(
            [history, pd.DataFrame({"date": [last_date], "price": [price]})],
            ignore_index=True
        )

        future.append({"date": last_date, "predicted_price": price})

    return pd.DataFrame(future)

# ================= UI =================
st.title("⚡ Bitcoin Price Prediction")

df = load_data()

if df.empty or len(df) < MIN_ROWS:
    st.error("Unable to load sufficient BTC data.")
    st.stop()

model = train_model(df)
future_df = predict_future(df, model, FUTURE_DAYS)

current_price = df["price"].iloc[-1]
future_price = future_df["predicted_price"].iloc[-1]

expected_change = ((future_price - current_price) / current_price) * 100

# ================= METRICS =================
c1, c2, c3 = st.columns(3)
c1.metric("Current BTC Price", f"${current_price:,.2f}")
c2.metric(f"Price After {FUTURE_DAYS} Days", f"${future_price:,.2f}")
c3.metric("Expected Change", f"{expected_change:+.2f}%")

# ================= PLOT =================
plot_df = pd.concat([
    df.tail(90)[["date", "price"]].rename(columns={"price": "Value"}).assign(Type="Historical"),
    future_df.rename(columns={"predicted_price": "Value"}).assign(Type="Forecast")
])

fig = px.line(plot_df, x="date", y="Value", color="Type")
st.plotly_chart(fig, use_container_width=True)

with st.expander("Show Forecast Data"):
    st.dataframe(future_df)
