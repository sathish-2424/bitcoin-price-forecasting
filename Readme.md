---

# 📈 Bitcoin Price Prediction

## 🧠 Overview

This project builds a **time-series forecasting system** to predict **Bitcoin (BTC) prices** using a **GRU (Gated Recurrent Unit) neural network**.

The model is trained on **historical Bitcoin market data** enriched with **technical indicators**, and it generates **14-day future price forecasts**.
Live BTC price data is fetched from **Yahoo Finance** using `yfinance`, and results are displayed via an **interactive Streamlit dashboard**.

---

## ✨ Key Features

* 📡 Live Bitcoin price data (`BTC-USD`)
* 🧮 Technical indicator–based feature engineering
* 🤖 GRU-based deep learning time-series model
* 📆 14-day multi-step price forecasting
* 📊 Interactive Streamlit dashboard
* 📌 Key performance indicators (KPIs)
* 🚀 Ready for local or cloud deployment

---

## 📊 Dataset

* **Source:** Yahoo Finance (`yfinance`)
* **Asset:** Bitcoin (BTC-USD)
* **Interval:** Daily (`1d`)
* **Historical Window:** Last 5 years

### Engineered Features

| Feature      | Description                      |
| ------------ | -------------------------------- |
| `price`      | Daily closing price              |
| `ma_7`       | 7-day moving average             |
| `ma_30`      | 30-day moving average            |
| `volatility` | 7-day rolling standard deviation |

---

## 🏗️ Project Workflow

1. Fetch historical BTC price data
2. Perform feature engineering
3. Normalize data using MinMaxScaler
4. Generate time-series sequences (lookback window)
5. Train GRU neural network
6. Predict future Bitcoin prices
7. Visualize results with Streamlit

---

## 🧠 Model Architecture

* **Model Type:** GRU (Gated Recurrent Unit)
* **Layers:**

  * GRU (64 units, return sequences = True)
  * Dropout (0.2)
  * GRU (32 units)
  * Dense (1)
* **Loss Function:** Mean Squared Error (MSE)
* **Optimizer:** Adam
* **Early Stopping:** Enabled

---

## ⚙️ Model Configuration

| Parameter        | Value   |
| ---------------- | ------- |
| Lookback Window  | 60 days |
| Forecast Horizon | 14 days |
| Epochs           | 30      |
| Batch Size       | 32      |
| Validation Split | 5%      |

---

## 📊 Streamlit Dashboard Features

### 📌 KPIs

* Last known Bitcoin price
* 14-day forecasted price
* Expected percentage change

### 📈 Visualizations

* Actual vs predicted Bitcoin price line chart
* 14-day forecast table

---

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies

```bash
pip install streamlit tensorflow yfinance pandas numpy scikit-learn plotly
```

### 2️⃣ Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

### 3️⃣ Open in Browser

```text
http://localhost:8501
```

---

## 📈 Example Model Output

| Date       | Predicted Price (USD) |
| ---------- | --------------------- |
| 2025-01-01 | 43,825.14             |
| 2025-01-02 | 44,102.78             |
| ...        | ...                   |

---

## 🛠️ Tech Stack

| Category           | Tools                       |
| ------------------ | --------------------------- |
| Language           | Python                      |
| Deep Learning      | TensorFlow / Keras          |
| Data Processing    | Pandas, NumPy               |
| Scaling            | Scikit-learn (MinMaxScaler) |
| Financial Data API | yfinance                    |
| Visualization      | Plotly                      |
| Web App            | Streamlit                   |

---