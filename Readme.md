---

# 📈 Bitcoin Price Prediction

## 🧠 Overview

This project builds a **machine learning forecasting system** to predict **Bitcoin (BTC) prices** using **LSTM (Long Short-Term Memory)** neural networks.

The model is trained on **historical Bitcoin market data**, and it generates **next-day and 7-day future price forecasts**.
Live BTC price data is fetched from **Yahoo Finance** using `yfinance`, and results are displayed via an **interactive Streamlit dashboard**.

---

## ✨ Key Features

* 📡 Live Bitcoin price data (`BTC-USD`)
* � LSTM neural network model
* 📆 Next-day and 7-day price forecasting
* 📊 Interactive Streamlit dashboard with Matplotlib visualizations
* 📌 Key performance indicators (KPIs)
* 📈 Historical vs predicted price visualization
* 🚀 Ready for local or cloud deployment

---

## 📊 Dataset

* **Source:** Yahoo Finance (`yfinance`)
* **Asset:** Bitcoin (BTC-USD)
* **Interval:** Daily (`1d`)
* **Historical Window:** Configurable (default: 2 years)

---

## 🏗️ Project Workflow

1. Fetch historical BTC price data from Yahoo Finance
2. Preprocess data with MinMax scaling
3. Create sequences for LSTM training
4. Train LSTM model or load existing model
5. Generate next-day and 7-day price forecasts
6. Visualize results with interactive Streamlit dashboard

---

## 🧠 Model Architecture

* **Model Type:** LSTM Neural Network
* **Architecture:** 2 LSTM layers (50 units each) + Dense output
* **Activation:** ReLU for LSTM layers
* **Optimizer:** Adam (learning rate: 0.001)
* **Loss Function:** Mean Squared Error (MSE)
* **Sequence Length:** Configurable (default: 60 days)

---

## ⚙️ Model Configuration

| Parameter        | Value        |
| ---------------- | ------------ |
| Sequence Length  | 60 days      |
| Forecast Horizon | 1-7 days     |
| LSTM Units       | 50           |
| Learning Rate    | 0.001        |
| Epochs           | 25           |
| Batch Size       | 16           |

---

## 📊 Streamlit Dashboard Features

### 📌 KPIs

* Current Bitcoin price
* Next-day forecasted price
* Expected percentage change

### 📈 Visualizations

* Historical Bitcoin price line chart
* Actual vs predicted prices comparison
* 7-day forecast data table

### 📋 Additional Features

* Configurable date range and lookback period
* Model training epochs adjustment
* Force retrain option

---

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install streamlit yfinance pandas numpy scikit-learn tensorflow matplotlib
```

### 2️⃣ Run the Streamlit App

```bash
streamlit run streamlit_app.py
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
* Next-day forecasted price
* 7-day forecasted prices in a table
* Interactive chart showing actual vs predicted prices
* Historical price trends

Example forecast output:
```
Day | Predicted Price ($)
----|-------------------
1   | $43,825.14
2   | $44,102.78
...
7   | $45,234.56
```

## 📁 Project Files

* `streamlit_app.py` - Main Streamlit application (LSTM-based)
* `model.py` - LSTM model training and loading utilities
* `btc_data.csv` - Historical Bitcoin price data
* `bitcoin_lstm.h5` - Trained LSTM model
* `scaler.pkl` - Data scaler for preprocessing
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