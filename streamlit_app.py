# =====================================================
# Bitcoin Price Forecasting
# =====================================================

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import timedelta
from xgboost import XGBRegressor
import plotly.express as px
import warnings

warnings.filterwarnings("ignore")

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Bitcoin Price Prediction",
    layout="wide",
    page_icon="⚡"
)

# ================= CONFIG =================
TICKER = "BTC-USD"
PERIOD = "2y"
INTERVAL = "1d"
FUTURE_DAYS = 14

FEATURES = [
    "price_lag1",
    "price_lag7",
    "ma_7",
    "ma_30",
    "volatility",
    "rsi"
]

# ================= TECHNICAL INDICATORS =================
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

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    df = yf.download(TICKER, period=PERIOD, interval=INTERVAL, progress=False)
    df = df.reset_index()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    df = df[["date", "close"]].rename(columns={"close": "price"})
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df = add_technical_indicators(df)

    # Target = next day price
    df["target"] = df["price"].shift(-1)

    df.dropna(inplace=True)
    return df

# ================= TRAIN MODEL =================
@st.cache_resource
def train_model(df):
    X = df[FEATURES]
    y = df["target"]

    split = int(len(df) * 0.9)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    return model

# ================= FUTURE PREDICTION =================
def predict_future(df, model, days):
    if df.empty:
        return pd.DataFrame(columns=["date", "predicted_price"])

    history = df.copy()

    if len(history) == 0:
        return pd.DataFrame(columns=["date", "predicted_price"])

    future_preds = []
    last_date = pd.to_datetime(history["date"].iloc[-1])

    for _ in range(days):
        history = add_technical_indicators(history)

        if history[FEATURES].isna().any().any():
            history[FEATURES] = history[FEATURES].ffill().bfill()

        X_last = history[FEATURES].iloc[-1:]

        pred_price = model.predict(X_last)[0]
        pred_price = max(pred_price, 1)

        last_date += timedelta(days=1)
        while last_date.weekday() >= 5:
            last_date += timedelta(days=1)

        history = pd.concat(
            [history, pd.DataFrame({"date": [last_date], "price": [pred_price]})],
            ignore_index=True
        )

        future_preds.append({
            "date": last_date,
            "predicted_price": pred_price
        })

    return pd.DataFrame(future_preds)

# ================= UI =================
st.title("⚡ Bitcoin Price Prediction")

df = load_data()
model = train_model(df)

future_df = predict_future(df, model, FUTURE_DAYS)

current_price = df["price"].iloc[-1]
future_price = future_df["predicted_price"].iloc[-1]

# ================= METRICS =================
col1, col2, col3 = st.columns(3)

col1.metric(
    "Current BTC Price",
    f"${current_price:,.2f}"
)

col2.metric(
    f"Price After {FUTURE_DAYS} Days",
    f"${future_price:,.2f}"
)

# Safe + clean expected change calculation
expected_change = (
    (future_price - current_price) / current_price * 100
    if current_price != 0 else 0
)

col3.metric(
    "Expected Change",
    f"{expected_change:+.2f}%"
)


# ================= PLOT =================
st.subheader("📈 BTC Price Forecast")

hist_df = df.tail(90)[["date", "price"]].rename(columns={"price": "Value"})
hist_df["Type"] = "Historical"

future_plot = future_df.rename(columns={"predicted_price": "Value"})
future_plot["Type"] = "Forecast"

plot_df = pd.concat([hist_df, future_plot], ignore_index=True)

fig = px.line(
    plot_df,
    x="date",
    y="Value",
    color="Type",
    title="Bitcoin Price Forecast",
)
fig.update_layout(xaxis_title="Date", yaxis_title="Price (USD)")

st.plotly_chart(fig, use_container_width=True)

# ================= DATA TABLE =================
with st.expander("Show Forecast Data"):
    st.dataframe(future_df)