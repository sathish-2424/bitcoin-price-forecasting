import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import plotly.express as px
import plotly.graph_objects as go
import warnings

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Bitcoin Price Prediction ",
    layout="wide",
    page_icon="⚡"
)
warnings.filterwarnings("ignore")

# ================= CONFIGURATION =================
TICKER = "BTC-USD"
PERIOD = "2y" 
INTERVAL = "1d"
LOOKBACK = 60
FUTURE_DAYS = 14
FEATURES = ["price", "ma_7", "ma_30", "volatility", "rsi"]

# ================= HELPER: INDICATOR CALCULATION =================
def add_technical_indicators(df):
    """
    Calculates technical indicators. 
    """
    df = df.copy()
    # Moving Averages
    df["ma_7"] = df["price"].rolling(window=7).mean()
    df["ma_30"] = df["price"].rolling(window=30).mean()
    
    # Volatility
    df["volatility"] = df["price"].rolling(window=7).std()
    
    # RSI Calculation
    delta = df["price"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))
    
    return df

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    """
    Optimized data loading.
    """
    try:
        df = yf.download(TICKER, period=PERIOD, interval=INTERVAL, progress=False)
        df.reset_index(inplace=True)
        
        # Robust Column Handling
        if isinstance(df.columns, pd.MultiIndex):
            try:
                price_col = df["Close"] if "Close" in df.columns else df.iloc[:, 1]
                if isinstance(price_col, pd.DataFrame):
                    price_col = price_col.iloc[:, 0]
            except:
                price_col = df.iloc[:, 1]
                
            df_clean = pd.DataFrame({
                "date": df["Date"].iloc[:, 0] if isinstance(df["Date"], pd.DataFrame) else df["Date"],
                "price": price_col
            })
        else:
            df_clean = df[["Date", "Close"]].rename(columns={"Date": "date", "Close": "price"})

        df_clean["price"] = pd.to_numeric(df_clean["price"], errors='coerce')
        df_clean = add_technical_indicators(df_clean)
        df_clean.dropna(inplace=True)
        
        return df_clean
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# ================= SEQUENCE GENERATION =================
def create_sequences_vectorized(data, lookback):
    n_samples = len(data) - lookback
    X = np.array([data[i:i+lookback] for i in range(n_samples)])
    y = data[lookback:, 0] # Target is Price
    return X, y

# ================= TRAIN MODEL =================
@st.cache_resource
def train_gru_model(df):
    # Prepare Data
    feature_data = df[FEATURES].values
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(feature_data)
    
    X, y = create_sequences_vectorized(scaled_data, LOOKBACK)
    
    if len(X) == 0:
        return None, None, None, None, None

    # Split Data (Time Series Split - No Shuffle)
    split = int(len(X) * 0.9)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Build Model
    model = Sequential([
        Input(shape=(LOOKBACK, len(FEATURES))),
        GRU(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
        verbose=0
    )
    
    return model, scaler, X_test, y_test, split

# ================= OPTIMIZED PREDICTION LOOP =================
def predict_recursive_optimized(model, scaler, df, days_ahead):
    # OPTIMIZATION: Only use a small tail window for calculations
    # We don't need the 2-year history to calculate the MA_30 for tomorrow.
    # 100 days is enough buffer.
    working_df = df.iloc[-100:].copy() 
    
    future_predictions = []
    last_date = df["date"].iloc[-1]
    
    for _ in range(days_ahead):
        # 1. Recalc indicators on SMALL working dataframe (Fast)
        temp_df = add_technical_indicators(working_df)
        
        # 2. Get features
        valid_features = temp_df[FEATURES].tail(LOOKBACK).values
        
        # 3. Scale & Reshape
        scaled_input = scaler.transform(valid_features)
        X_input = scaled_input.reshape(1, LOOKBACK, len(FEATURES))
        
        # 4. Predict
        pred_scaled = model.predict(X_input, verbose=0)[0][0]
        
        # 5. Inverse Scale Price
        dummy = np.zeros((1, len(FEATURES)))
        dummy[0, 0] = pred_scaled
        pred_price = scaler.inverse_transform(dummy)[0][0]
        
        # 6. Update Buffer
        last_date += timedelta(days=1)
        
        new_row = pd.DataFrame({
            "date": [last_date],
            "price": [pred_price],
            "ma_7": [np.nan], "ma_30": [np.nan], "volatility": [np.nan], "rsi": [np.nan]
        })
        
        # OPTIMIZATION: Concat to small DF, then slice to keep it small
        working_df = pd.concat([working_df, new_row], ignore_index=True)
        if len(working_df) > 150:
            working_df = working_df.iloc[-100:]
            
        future_predictions.append({"date": last_date, "predicted_price": pred_price})
    
    return pd.DataFrame(future_predictions)

# ================= UI LAYOUT =================
st.title("⚡ Bitcoin Price Prediction & Analysis")

# --- 1. Load Data ---
with st.spinner("Loading Data..."):
    df = load_data()

if df.empty:
    st.error("Could not load data. API might be down.")
    st.stop()

# --- 2. Train Model ---
with st.spinner("Training Optimized GRU Model..."):
    model, scaler, X_test, y_test, split_idx = train_gru_model(df)

if not model:
    st.error("Training failed.")
    st.stop()

# --- 3. Run Forecast ---
future_df = predict_recursive_optimized(model, scaler, df, FUTURE_DAYS)

# --- 4. Main Dashboard ---
current_price = df["price"].iloc[-1]
pred_price = future_df["predicted_price"].iloc[-1]
delta = ((pred_price - current_price) / current_price) * 100
color = "normal" if delta == 0 else ("inverse" if delta < 0 else "normal")

c1, c2, c3 = st.columns(3)
c1.metric("Current Price", f"${current_price:,.2f}")
c2.metric(f"Forecast ({FUTURE_DAYS} Days)", f"${pred_price:,.2f}")
c3.metric("Projected ROI", f"{delta:+.2f}%", delta_color=color)

# --- 5. Visualization ---
tab1, tab2 = st.tabs(["🔮 Future Forecast", "📊 Model Accuracy Analysis"])

with tab1:
    st.subheader("Future Price Trajectory")
    
    # Prepare data for plotting
    hist_data = df.tail(90)[["date", "price"]].copy()
    hist_data["Type"] = "Historical"
    hist_data.rename(columns={"price": "Value"}, inplace=True)

    fut_data = future_df.copy()
    fut_data["Type"] = "Forecast"
    fut_data.rename(columns={"predicted_price": "Value"}, inplace=True)

    # Connector line
    connector = pd.DataFrame([{
        "date": hist_data["date"].iloc[-1],
        "Value": hist_data["Value"].iloc[-1],
        "Type": "Forecast"
    }])
    fut_data = pd.concat([connector, fut_data], ignore_index=True)
    plot_df = pd.concat([hist_data, fut_data], ignore_index=True)

    fig = px.line(
        plot_df, x="date", y="Value", color="Type",
        color_discrete_map={"Historical": "#00CC96", "Forecast": "#EF553B"},
        title="BTC-USD: 14-Day Forecast"
    )
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("How reliable is this model?")
    st.write("We test the model by asking it to predict the *past 10%* of data (Test Set) and comparing it to what actually happened.")
    
    # --- Generate Predictions on Test Set ---
    test_preds_scaled = model.predict(X_test, verbose=0)
    
    # Inverse Transform
    # We need to inverse transform (N, 1) back to (N, 5) format using dummy filler
    dummy_pred = np.zeros((len(test_preds_scaled), len(FEATURES)))
    dummy_pred[:, 0] = test_preds_scaled.flatten()
    test_preds_price = scaler.inverse_transform(dummy_pred)[:, 0]
    
    dummy_actual = np.zeros((len(y_test), len(FEATURES)))
    dummy_actual[:, 0] = y_test
    test_actual_price = scaler.inverse_transform(dummy_actual)[:, 0]
    
    # Metrics
    mae = mean_absolute_error(test_actual_price, test_preds_price)
    rmse = np.sqrt(mean_squared_error(test_actual_price, test_preds_price))
    
    m1, m2 = st.columns(2)
    m1.metric("MAE (Avg Error)", f"${mae:.2f}")
    m2.metric("RMSE (Root Mean Sq Error)", f"${rmse:.2f}")
    
    # Plot Test vs Actual
    # Get dates for test set
    test_dates = df["date"].iloc[split_idx + LOOKBACK:].values
    
    fig_test = go.Figure()
    fig_test.add_trace(go.Scatter(x=test_dates, y=test_actual_price, name="Actual Price", line=dict(color="#00CC96")))
    fig_test.add_trace(go.Scatter(x=test_dates, y=test_preds_price, name="Model Prediction", line=dict(color="#AB63FA", dash="dot")))
    
    fig_test.update_layout(title="Validation: Actual vs Predicted (Test Data)", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig_test, use_container_width=True)