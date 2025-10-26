---

## 🪙 Bitcoin Price Forecasting Web App

An interactive **Streamlit** application that predicts Bitcoin prices using a blend of **machine learning** and **deep learning** models.
Upload your own BTC dataset and visualize forecasts, model performance, and technical indicators in real-time.

---

### 🚀 Features

* 📂 **CSV Upload Support** – Upload historical Bitcoin price data.
* 🧠 **Multiple ML Models**

  * Ridge Regression
  * Random Forest
  * Gradient Boosting
  * Dense Neural Network
  * LSTM
  * Ensemble (combined predictions)
* 📊 **Dynamic Visualizations** – Plotly charts for actual vs predicted prices.
* ⚙️ **Feature Engineering** – Automatically computes:

  * Moving Averages (7, 14, 30)
  * RSI (Relative Strength Index)
  * Volatility
  * Momentum
  * Rate of Change
* 🧾 **Performance Metrics** – MAE, RMSE, and R² scores for all models.
* 🔮 **Future Forecasting** – Predict Bitcoin prices for any future date.

---

### 🧩 Tech Stack

* **Frontend:** Streamlit + Plotly
* **Backend:** TensorFlow / Scikit-learn
* **Language:** Python 3.9+
* **Dependencies:**

  * `streamlit`
  * `pandas`, `numpy`
  * `tensorflow`, `scikit-learn`
  * `plotly`

---

### 📁 Dataset Format

Upload a CSV file containing the following columns:

| Column                | Description                      |
| :-------------------- | :------------------------------- |
| `Date`                | Date of observation (YYYY-MM-DD) |
| `Closing Price (USD)` | Bitcoin closing price in USD     |

**Example:**

```csv
Date,Closing Price (USD)
2021-01-01,29374.15
2021-01-02,32127.27
2021-01-03,32782.02
```

---

### ⚙️ Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/bitcoin-price-forecasting.git
   cd bitcoin-price-forecasting
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**

   ```bash
   streamlit run app.py
   ```

4. **Open in your browser:**
   👉 [http://localhost:8501](http://localhost:8501)

---

### 📈 How It Works

1. Upload a CSV file of Bitcoin closing prices.
2. Select a model and choose whether to display metrics or predictions.
3. The app trains all models, evaluates performance, and visualizes predictions.
4. Optionally, pick a future date to forecast the Bitcoin price.

---

### 📷 Example UI

* Dataset overview and metrics
* Model performance comparison
* Interactive Plotly chart (Actual vs Predicted)
* Future prediction display with price change indicator

---

### 🧠 Model Overview

| Model                 | Description                                          |
| :-------------------- | :--------------------------------------------------- |
| **Ridge Regression**  | Linear model with L2 regularization                  |
| **Random Forest**     | Ensemble of decision trees for non-linear patterns   |
| **Gradient Boosting** | Sequential tree-based boosting                       |
| **Dense NN**          | Fully connected neural network                       |
| **LSTM**              | Long Short-Term Memory network for sequence learning |
| **Ensemble**          | Average of all model predictions                     |

---

### 📜 License

This project is released under the **MIT License**.
You are free to use, modify, and distribute it.

---
