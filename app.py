import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import datetime
import plotly.graph_objects as go
import plotly.express as px

# ============= PAGE CONFIG =============
st.set_page_config(page_title="Bitcoin Price Forecasting", layout="wide", initial_sidebar_state="expanded")

# ============= CUSTOM CSS =============
st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .prediction-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ============= TITLE & DESCRIPTION =============
col1, col2 = st.columns([3, 1])
with col1:
    st.title("Bitcoin Price Forecasting")
    st.markdown("*Advanced ML models for predicting BTC prices*")
with col2:
    st.metric("Current Date", datetime.date.today().strftime("%b %d, %Y"))

st.divider()

# ============= SIDEBAR CONFIGURATION =============
with st.sidebar:
    st.header("Configuration")
    
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"], 
                                     help="Required columns: 'Date' and 'Closing Price (USD)'")
    
    st.divider()
    st.subheader("Model Settings")
    model_choice = st.selectbox("Select Model", 
        ["Ridge Regression", "Random Forest", "Gradient Boosting", "Dense Neural Network", "LSTM", "Ensemble"])
    
    show_metrics = st.checkbox("Show Performance Metrics", value=True)
    show_predictions = st.checkbox("Show Test Predictions", value=True)

# ============= LOAD & VALIDATE DATA =============
if not uploaded_file:
    st.warning("Please upload a CSV file to get started")
    st.info("Required columns: 'Date' and 'Closing Price (USD)'")
    st.stop()

try:
    df = pd.read_csv(uploaded_file, parse_dates=["Date"], index_col=["Date"])
    if "Closing Price (USD)" not in df.columns:
        st.error("Missing 'Closing Price (USD)' column")
        st.stop()
except Exception as e:
    st.error(f"Error loading file: {str(e)}")
    st.stop()

if len(df) < 31:
    st.error("Dataset too small. Need at least 31 rows.")
    st.stop()

# ============= DATASET OVERVIEW =============
with st.expander("Dataset Overview", expanded=True):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Current Price", f"${df['Closing Price (USD)'].iloc[-1]:,.2f}")
    with col3:
        st.metric("Min Price", f"${df['Closing Price (USD)'].min():,.2f}")
    with col4:
        st.metric("Max Price", f"${df['Closing Price (USD)'].max():,.2f}")
    
    st.dataframe(df.head(10), use_container_width=True)

prices = df["Closing Price (USD)"].values

# ============= FEATURE ENGINEERING =============
def calculate_rsi(prices, period=14):
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed>=0].sum()/period
    down = -seed[seed<0].sum()/period
    rs = up/down if down!=0 else 0
    rsi = np.zeros_like(prices)
    rsi[:period] = 100.-100./(1.+rs)
    for i in range(period,len(prices)):
        delta = deltas[i-1]
        upval = max(delta,0)
        downval = -min(delta,0)
        up = (up*(period-1)+upval)/period
        down = (down*(period-1)+downval)/period
        rs = up/down if down!=0 else 0
        rsi[i] = 100.-100./(1.+rs)
    return rsi

def create_features(data):
    df_features = pd.DataFrame(data, columns=['Price'])
    for lag in [1,2,3,7,14,30]:
        df_features[f'Lag_{lag}'] = df_features['Price'].shift(lag)
    df_features['Returns'] = df_features['Price'].pct_change()
    df_features['MA_7'] = df_features['Price'].rolling(7).mean()
    df_features['MA_14'] = df_features['Price'].rolling(14).mean()
    df_features['MA_30'] = df_features['Price'].rolling(30).mean()
    df_features['Volatility'] = df_features['Returns'].rolling(14).std()
    df_features['RSI'] = calculate_rsi(data, 14)
    df_features['Momentum'] = df_features['Price'].diff(14)
    df_features['Rate_of_Change'] = df_features['Price'].pct_change(14)
    df_features['Trend'] = (df_features['MA_7']-df_features['MA_30'])/df_features['MA_30']
    return df_features.dropna()

df_features = create_features(prices)
X = df_features.drop('Price', axis=1).values
y = df_features['Price'].values

scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1,1)).flatten()

split_size = int(0.8*len(X_scaled))
X_train, X_test = X_scaled[:split_size], X_scaled[split_size:]
y_train, y_test = y_scaled[:split_size], y_scaled[split_size:]
y_train_orig, y_test_orig = y[:split_size], y[split_size:]

# ============= TRAIN MODELS =============
with st.spinner("Training models... This may take a minute..."):
    # Ridge
    ridge = Ridge(alpha=10.0).fit(X_train, y_train)
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=200, max_depth=25, min_samples_split=5,
                               min_samples_leaf=2, random_state=42, n_jobs=-1).fit(X_train, y_train)
    
    # Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=7,
                                   min_samples_split=5, random_state=42).fit(X_train, y_train)
    
    # Dense NN
    tf.random.set_seed(42)
    model_dense = tf.keras.Sequential([
        layers.Dense(256, activation="relu", input_shape=(X_train.shape[1],)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="linear")
    ])
    model_dense.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(0.001), metrics=['mae'])
    model_dense.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.15, verbose=0)
    
    # LSTM
    X_train_3d = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_3d = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    model_lstm = tf.keras.Sequential([
        layers.LSTM(128, activation="relu", return_sequences=True, input_shape=(X_train.shape[1],1)),
        layers.Dropout(0.2),
        layers.LSTM(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="linear")
    ])
    model_lstm.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(0.001))
    model_lstm.fit(X_train_3d, y_train, epochs=50, batch_size=32, validation_split=0.15, verbose=0)

st.success("✅ Models trained successfully!")

# ============= PREDICTION FUNCTION =============
def predict_model(model_name, X_input):
    if model_name == "Ridge Regression":
        return ridge.predict(X_input)
    elif model_name == "Random Forest":
        return rf.predict(X_input)
    elif model_name == "Gradient Boosting":
        return gb.predict(X_input)
    elif model_name == "Dense Neural Network":
        return model_dense.predict(X_input, verbose=0).flatten()
    elif model_name == "LSTM":
        return model_lstm.predict(X_input.reshape(1, X_input.shape[1], 1), verbose=0).flatten()
    elif model_name == "Ensemble":
        preds = [
            ridge.predict(X_input),
            rf.predict(X_input),
            gb.predict(X_input),
            model_dense.predict(X_input, verbose=0).flatten(),
            model_lstm.predict(X_input.reshape(1, X_input.shape[1], 1), verbose=0).flatten()
        ]
        return np.mean(preds, axis=0)

# ============= MODEL PERFORMANCE =============
if show_metrics:
    st.divider()
    st.subheader("Model Performance on Test Set")
    
    predictions_ridge = ridge.predict(X_test)
    predictions_rf = rf.predict(X_test)
    predictions_gb = gb.predict(X_test)
    predictions_dense = model_dense.predict(X_test, verbose=0).flatten()
    predictions_lstm = model_lstm.predict(X_test_3d, verbose=0).flatten()
    
    metrics_data = {
        "Model": ["Ridge", "Random Forest", "Gradient Boosting", "Dense NN", "LSTM"],
        "MAE": [
            mean_absolute_error(y_test, predictions_ridge),
            mean_absolute_error(y_test, predictions_rf),
            mean_absolute_error(y_test, predictions_gb),
            mean_absolute_error(y_test, predictions_dense),
            mean_absolute_error(y_test, predictions_lstm)
        ],
        "RMSE": [
            np.sqrt(mean_squared_error(y_test, predictions_ridge)),
            np.sqrt(mean_squared_error(y_test, predictions_rf)),
            np.sqrt(mean_squared_error(y_test, predictions_gb)),
            np.sqrt(mean_squared_error(y_test, predictions_dense)),
            np.sqrt(mean_squared_error(y_test, predictions_lstm))
        ],
        "R² Score": [
            r2_score(y_test, predictions_ridge),
            r2_score(y_test, predictions_rf),
            r2_score(y_test, predictions_gb),
            r2_score(y_test, predictions_dense),
            r2_score(y_test, predictions_lstm)
        ]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df.style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R² Score": "{:.4f}"}), use_container_width=True)

# ============= PREDICTIONS VISUALIZATION =============
if show_predictions:
    st.divider()
    st.subheader("Test Set Predictions vs Actual")
    
    pred_scaled = predict_model(model_choice, X_test)
    pred_actual = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_test_orig, name='Actual Price', mode='lines', line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(y=pred_actual, name=f'{model_choice} Prediction', mode='lines', 
                            line=dict(color='red', width=2, dash='dash')))
    fig.update_layout(title=f'{model_choice} - Test Set Performance', 
                     xaxis_title='Time Period', yaxis_title='Price (USD)', 
                     height=400, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

# ============= FUTURE PREDICTION =============
st.divider()
st.subheader("Predict Bitcoin Price")

col1, col2 = st.columns([2, 1])
with col1:
    future_date = st.date_input("Select a future date", value=datetime.date.today() + datetime.timedelta(days=1))
with col2:
    predict_btn = st.button("Make Prediction", use_container_width=True)

if predict_btn:
    if future_date <= df.index[-1].date():
        st.error("Please select a future date beyond the dataset")
    else:
        last_row = df_features.iloc[-1].copy()
        for lag in [1,2,3,7,14,30]:
            last_row[f'Lag_{lag}'] = prices[-lag]
        recent_prices = prices[-30:]
        last_row['MA_7'] = np.mean(recent_prices[-7:])
        last_row['MA_14'] = np.mean(recent_prices[-14:])
        last_row['MA_30'] = np.mean(recent_prices)
        last_row['Momentum'] = prices[-1] - prices[-14]
        last_row['Rate_of_Change'] = (prices[-1] - prices[-14]) / prices[-14]
        last_row['Returns'] = (prices[-1] - prices[-2]) / prices[-2]
        last_row['Volatility'] = np.std(np.diff(prices[-15:]))
        last_row['RSI'] = calculate_rsi(prices, 14)[-1]
        last_row['Trend'] = (last_row['MA_7'] - last_row['MA_30']) / last_row['MA_30']
        
        X_future = np.array(last_row.drop('Price')).reshape(1, -1)
        X_future_scaled = scaler_X.transform(X_future)
        pred_scaled = predict_model(model_choice, X_future_scaled)
        pred_price = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="prediction-box">
                <h3>Predicted BTC Price</h3>
                <h2>${pred_price:,.2f}</h2>
                <p>on {future_date.strftime('%B %d, %Y')}</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            current_price = prices[-1]
            change = pred_price - current_price
            pct_change = (change / current_price) * 100
            direction = "📈" if change > 0 else "📉"
            st.markdown(f"""
            <div class="metric-card">
                <h3>Expected Change</h3>
                <h2>{direction} ${abs(change):,.2f}</h2>
                <p>({pct_change:+.2f}%)</p>
            </div>
            """, unsafe_allow_html=True)