# =====================================================
# Bitcoin Price Prediction – XGBoost ML Pipeline
# =====================================================

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import timedelta
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import plotly.express as px
import warnings

warnings.filterwarnings("ignore")

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Bitcoin ML Price Prediction",
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
    df["target"] = df["price"].shift(-1)
    df.dropna(inplace=True)

    return df

# ================= TRAIN XGBOOST =================
@st.cache_resource
def train_xgb(df):
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

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    return model

# ================= TRAIN LIGHTGBM =================
@st.cache_resource
def train_lgb(df):
    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(df[FEATURES], df["target"])
    return model

# ================= WALK FORWARD =================
def walk_forward_validation(df, window=500, step=30):
    preds, actuals = [], []

    for start in range(0, len(df) - window - step, step):
        train = df.iloc[start:start+window]
        test = df.iloc[start+window:start+window+step]

        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            objective="reg:squarederror"
        )
        model.fit(train[FEATURES], train["target"])

        preds.extend(model.predict(test[FEATURES]))
        actuals.extend(test["target"])

    mae = np.mean(np.abs(np.array(actuals) - np.array(preds)))
    rmse = np.sqrt(np.mean((np.array(actuals) - np.array(preds)) ** 2))
    return mae, rmse

# ================= SIGNAL =================
def trading_signal(current, predicted, threshold=1.0):
    pct = (predicted - current) / current * 100
    if pct > threshold:
        return "BUY 🚀", pct
    elif pct < -threshold:
        return "SELL 🔻", pct
    else:
        return "HOLD ⚖️", pct

# ================= BACKTEST =================
def backtest(df, model, capital=10000):
    cash, btc = capital, 0
    equity = []

    for i in range(1, len(df)):
        X = df[FEATURES].iloc[i-1:i]
        pred = model.predict(X)[0]
        price = df["price"].iloc[i]

        if pred > price * 1.01 and cash > 0:
            btc = cash / price
            cash = 0
        elif pred < price * 0.99 and btc > 0:
            cash = btc * price
            btc = 0

        equity.append(cash + btc * price)

    roi = (equity[-1] - capital) / capital * 100
    return roi, equity

# ================= FUTURE PREDICTION =================
def predict_future(df, model, days):
    history = df.copy()
    preds = []
    last_date = history["date"].iloc[-1]

    for _ in range(days):
        history = add_technical_indicators(history)
        X = history[FEATURES].iloc[-1:]
        price = model.predict(X)[0]

        last_date += timedelta(days=1)
        while last_date.weekday() >= 5:
            last_date += timedelta(days=1)

        history = pd.concat([history, pd.DataFrame({
            "date": [last_date],
            "price": [price]
        })], ignore_index=True)

        preds.append({"date": last_date, "predicted_price": price})

    return pd.DataFrame(preds)

# ================= UI =================
st.title("⚡ Bitcoin Price Prediction (ML – XGBoost)")

df = load_data()
model = train_xgb(df)
lgb_model = train_lgb(df)

future = predict_future(df, model, FUTURE_DAYS)

current_price = df["price"].iloc[-1]
pred_price = future["predicted_price"].iloc[-1]

signal, change = trading_signal(current_price, pred_price)

roi, equity = backtest(df, model)
mae, rmse = walk_forward_validation(df)

col1, col2, col3 = st.columns(3)
col1.metric("Current Price", f"${current_price:,.2f}")
col2.metric(f"{FUTURE_DAYS} Day Forecast", f"${pred_price:,.2f}")
col3.metric("Signal", signal)

st.metric("Expected Change", f"{change:+.2f}%")
st.metric("Backtest ROI", f"{roi:.2f}%")
st.metric("Walk-Forward MAE", f"{mae:.2f}")
st.metric("Walk-Forward RMSE", f"{rmse:.2f}")

# ================= FEATURE IMPORTANCE =================
st.subheader("🔍 Feature Importance")
imp = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

fig = px.bar(imp, x="Importance", y="Feature", orientation="h")
st.plotly_chart(fig, use_container_width=True)

# ================= FORECAST PLOT =================
st.subheader("📈 Forecast")
plot_df = pd.concat([
    df.tail(90)[["date", "price"]].rename(columns={"price": "Value"}).assign(Type="History"),
    future.rename(columns={"predicted_price": "Value"}).assign(Type="Forecast")
])

fig = px.line(plot_df, x="date", y="Value", color="Type")
st.plotly_chart(fig, use_container_width=True)

st.dataframe(future)
