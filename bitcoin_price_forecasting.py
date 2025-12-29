### Bitcoin Price Forecasting
"""

! pip install yfinance tensorflow scikit-learn numpy pandas

"""### Imports"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

"""### CONFIG"""

TICKER = "BTC-USD"
INTERVAL = "1d"
PERIOD = "5y"
LOOKBACK = 60
FUTURE_DAYS = 14
FEATURES = ["price", "ma_7", "ma_30", "volatility"]

"""### LOAD LIVE DATA"""

df = yf.download(TICKER, period=PERIOD, interval=INTERVAL, progress=False)
df = df.reset_index()
df = df[["Date", "Close"]].rename(columns={"Date": "date", "Close": "price"})
df.dropna(inplace=True)

"""### FEATURE ENGINEERING"""

df["ma_7"] = df["price"].rolling(7).mean()
df["ma_30"] = df["price"].rolling(30).mean()
df["volatility"] = df["price"].rolling(7).std()
df.dropna(inplace=True)

"""### SCALE DATA"""

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[FEATURES].values)

print("  - Naive Baseline (previous day)")

naive_pred = y_test[:-1]
naive_results = evaluate_model(y_test[1:], naive_pred, "Naive", y_test_orig[1:])

"""### CREATE SEQUENCES"""

X, y = [], []
for i in range(LOOKBACK, len(scaled)):
    X.append(scaled[i-LOOKBACK:i])
    y.append(scaled[i, 0])  # price only

X, y = np.array(X), np.array(y)

split = int(0.95 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

"""### GRU MODEL"""

model = Sequential([
    GRU(64, return_sequences=True, input_shape=(LOOKBACK, len(FEATURES))),
    Dropout(0.2),
    GRU(32),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")

model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=32,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
    verbose=1
)

"""### FUTURE PREDICTION"""

buffer = df[FEATURES].tail(LOOKBACK).copy()
# Flatten MultiIndex column names to single-level strings for the scaler
buffer.columns = [col[0] if isinstance(col, tuple) else col for col in buffer.columns]
future = []

last_date = df["date"].iloc[-1]

for _ in range(FUTURE_DAYS):
    scaled_buf = scaler.transform(buffer.values)
    X_input = scaled_buf.reshape(1, LOOKBACK, len(FEATURES))

    pred_scaled = model.predict(X_input, verbose=0)[0][0]

    dummy = np.zeros((1, len(FEATURES)))
    dummy[0, 0] = pred_scaled
    pred_price = scaler.inverse_transform(dummy)[0, 0]

    next_date = last_date + timedelta(days=1)
    last_date = next_date

    new_row = {
        "price": pred_price,
        "ma_7": buffer["price"].rolling(7).mean().iloc[-1],
        "ma_30": buffer["price"].rolling(30).mean().iloc[-1],
        "volatility": buffer["price"].rolling(7).std().iloc[-1],
    }

    buffer = pd.concat([buffer, pd.DataFrame([new_row])]).tail(LOOKBACK)
    future.append({"date": next_date, "predicted_price": pred_price})

"""### OUTPUT"""

future_df = pd.DataFrame(future)
print(future_df)