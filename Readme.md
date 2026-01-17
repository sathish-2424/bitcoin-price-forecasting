---

# 📈 Bitcoin Price Prediction

## 🧠 Overview

This project builds a **machine learning forecasting system** to predict **Bitcoin (BTC) prices** using **XGBoost** gradient boosting regression models.

The model is trained on **historical Bitcoin market data** enriched with **technical indicators**, and it generates **14-day future price forecasts**.
Live BTC price data is fetched from **Yahoo Finance** using `yfinance`, and results are displayed via an **interactive Streamlit dashboard**.

---

## ✨ Key Features

* 📡 Live Bitcoin price data (`BTC-USD`)
* 🧮 Technical indicator–based feature engineering (RSI, Moving Averages, Volatility, Lag features)
* 🤖 XGBoost gradient boosting regression model
* 📆 14-day multi-step price forecasting
* 📊 Interactive Streamlit dashboard with Plotly visualizations
* 📌 Key performance indicators (KPIs) and trading signals
* 🔍 Feature importance analysis
* 📈 Historical vs forecasted price visualization
* 🚀 Ready for local or cloud deployment

---

## 📊 Dataset

* **Source:** Yahoo Finance (`yfinance`)
* **Asset:** Bitcoin (BTC-USD)
* **Interval:** Daily (`1d`)
* **Historical Window:** Last 2 years

### Engineered Features

| Feature       | Description                           |
| ------------- | ------------------------------------- |
| `price_lag1`  | Previous day's price                  |
| `price_lag7`  | Price 7 days ago                      |
| `ma_7`        | 7-day moving average                  |
| `ma_30`       | 30-day moving average                 |
| `volatility`  | 7-day rolling standard deviation      |
| `rsi`         | Relative Strength Index (14-period)   |

---

## 🏗️ Project Workflow

1. Fetch historical BTC price data from Yahoo Finance
2. Perform feature engineering (lags, moving averages, RSI, volatility)
3. Split data into train/test sets (90/10 split)
4. Train XGBoost regression model
5. Generate 14-day future price forecasts using iterative prediction
6. Calculate trading signals (BUY/SELL/HOLD) based on predictions
7. Visualize results with interactive Streamlit dashboard

---

## 🧠 Model Architecture

* **Model Type:** XGBoost Regressor (Gradient Boosting)
* **Key Hyperparameters:**
  * `n_estimators`: 500
  * `learning_rate`: 0.03
  * `max_depth`: 6
  * `subsample`: 0.8
  * `colsample_bytree`: 0.8
* **Objective:** reg:squarederror
* **Training:** 90% train, 10% test split with early stopping

---

## ⚙️ Model Configuration

| Parameter        | Value        |
| ---------------- | ------------ |
| Data Period      | 2 years      |
| Forecast Horizon | 14 days      |
| Train/Test Split | 90% / 10%    |
| N Estimators     | 500          |
| Learning Rate    | 0.03         |
| Max Depth        | 6            |

---

## 📊 Streamlit Dashboard Features

### 📌 KPIs

* Current Bitcoin price
* 14-day forecasted price
* Expected percentage change
* Trading signal (BUY 🚀 / SELL 🔻 / HOLD ⚖️)

### 📈 Visualizations

* Historical vs forecasted Bitcoin price line chart (Plotly)
* Feature importance bar chart
* 14-day forecast data table

### 📋 Additional Features (app.py)

* Backtest ROI calculation
* Walk-forward validation (MAE, RMSE)
* LightGBM model option

---

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install streamlit yfinance xgboost pandas numpy plotly scikit-learn
```

### 2️⃣ Run the Streamlit App

**Main Application:**
```bash
streamlit run streamlit_app.py
```

**Alternative Application (with backtesting):**
```bash
streamlit run app.py
```

### 3️⃣ Open in Browser

The app will automatically open at:
```
http://localhost:8501
```

---

## 📈 Example Model Output

The dashboard displays:
* Current BTC price
* 14-day forecasted prices in a table
* Interactive chart showing historical (90 days) and forecasted prices
* Feature importance rankings
* Trading signals based on predicted price movements

Example forecast output:
```
Date        | Predicted Price (USD)
------------|----------------------
2025-01-15  | $43,825.14
2025-01-16  | $44,102.78
...
```

## 📁 Project Files

* `streamlit_app.py` - Main Streamlit application (XGBoost-based)
* `app.py` - Enhanced version with backtesting and LightGBM support
* `bitcoin_price_forecasting.py` - GRU-based model (alternative implementation)
* `requirements.txt` - Python dependencies

---

## 🛠️ Tech Stack

| Category           | Tools                  |
| ------------------ | ---------------------- |
| Language           | Python                 |
| Machine Learning   | XGBoost, LightGBM      |
| Data Processing    | Pandas, NumPy          |
| Feature Engineering| Scikit-learn           |
| Financial Data API | yfinance               |
| Visualization      | Plotly                 |
| Web App            | Streamlit              |

---