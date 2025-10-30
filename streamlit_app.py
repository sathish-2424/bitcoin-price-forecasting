import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import datetime
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="Bitcoin Price Forecasting",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="💰"
)

# ===== STYLES =====
st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .prediction-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .error-box {
        background-color: #ffebee;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #f44336;
        color: #c62828;
    }
    </style>
""", unsafe_allow_html=True)

# ===== TITLE =====
col1, col2 = st.columns([3, 1])
with col1:
    st.title("💰 Bitcoin Price Forecasting")
with col2:
    st.metric("Current Date", datetime.date.today().strftime("%b %d, %Y"))

st.divider()

# ===== SIDEBAR =====
with st.sidebar:
    st.header("⚙️ Configuration")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    model_choice = st.selectbox(
        "Select Model",
        ["Ridge Regression", "Random Forest", "Gradient Boosting", "Ensemble"]
    )
    show_metrics = st.checkbox("Show Model Metrics", True)
    show_predictions = st.checkbox("Show Predictions", True)

# ===== LOAD & CLEAN DATA =====
@st.cache_data
def load_and_clean_data(file):
    """Load and validate CSV data"""
    try:
        df = pd.read_csv(file, parse_dates=["Date"], low_memory=False)
        df.columns = [c.strip().lower() for c in df.columns]
        
        # Check for required column (flexible naming)
        price_cols = [c for c in df.columns if "price" in c.lower() or "close" in c.lower()]
        if not price_cols:
            raise ValueError("Missing price/closing price column")
        
        price_col = price_cols[0]
        df = df.rename(columns={price_col: "price"})
        
        # Validate data
        if "date" not in df.columns:
            raise ValueError("Missing 'Date' column")
        
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        
        # Remove invalid rows
        initial_rows = len(df)
        df = df.dropna(subset=["price"])
        
        if len(df) < 30:
            raise ValueError(f"Not enough data: {len(df)} rows (minimum 30 required)")
        
        df = df.sort_values("date").reset_index(drop=True)
        
        return df, price_col
    
    except Exception as e:
        st.error(f"❌ Data Loading Error: {str(e)}")
        st.stop()

# Load dataset
if uploaded_file:
    df, original_price_col = load_and_clean_data(uploaded_file)
    st.sidebar.success(f"✅ Loaded {uploaded_file.name}")
else:
    try:
        df, original_price_col = load_and_clean_data("dataset.csv")
    except:
        st.error("❌ No file uploaded and 'dataset.csv' not found. Please upload a CSV file.")
        st.stop()

# ===== DATA OVERVIEW =====
with st.expander("📊 Dataset Overview", expanded=True):
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = df["price"].iloc[-1]
    price_min = df["price"].min()
    price_max = df["price"].max()
    price_range = price_max - price_min
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Current Price", f"${current_price:,.2f}")
    with col3:
        st.metric("Min Price", f"${price_min:,.2f}")
    with col4:
        st.metric("Max Price", f"${price_max:,.2f}")
    
    st.dataframe(df.tail(10), use_container_width=True)

# ===== FEATURE ENGINEERING =====
def create_features(data):
    """Create lagged features, moving averages, and returns"""
    df = data.copy()
    
    # Lagged features
    df["lag_1"] = df["price"].shift(1)
    df["lag_7"] = df["price"].shift(7)
    df["lag_14"] = df["price"].shift(14)
    
    # Moving averages
    df["ma_7"] = df["price"].rolling(window=7, min_periods=1).mean()
    df["ma_14"] = df["price"].rolling(window=14, min_periods=1).mean()
    df["ma_30"] = df["price"].rolling(window=30, min_periods=1).mean()
    
    # Volatility
    df["volatility_7"] = df["price"].rolling(window=7, min_periods=1).std()
    
    # Returns
    df["returns"] = df["price"].pct_change()
    
    # Remove NaN rows
    df = df.dropna()
    
    return df

df_feat = create_features(df)

# Prepare features and target
X = df_feat.drop(columns=["date", "price"], errors="ignore")
y = df_feat["price"]

# Scaling
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

# Train-test split
split = int(0.8 * len(X_scaled))
X_train, X_test = X_scaled[:split], X_scaled[split:]
y_train, y_test = y_scaled[:split], y_scaled[split:]
y_train_orig, y_test_orig = y.iloc[:split], y.iloc[split:]

# ===== TRAIN MODELS =====
@st.cache_resource
def train_models(X_train, y_train):
    """Train all models and return them"""
    with st.spinner("⚙️ Training models..."):
        ridge = Ridge(alpha=10)
        ridge.fit(X_train, y_train)
        
        rf = RandomForestRegressor(
            n_estimators=150,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        
        gb = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=5,
            random_state=42
        )
        gb.fit(X_train, y_train)
    
    return ridge, rf, gb

ridge, rf, gb = train_models(X_train, y_train)
st.success("✅ Training completed!")

# ===== PREDICTION FUNCTION =====
def predict_model(model_name, X_input):
    """Make predictions using selected model"""
    if model_name == "Ridge Regression":
        return ridge.predict(X_input)
    elif model_name == "Random Forest":
        return rf.predict(X_input)
    elif model_name == "Gradient Boosting":
        return gb.predict(X_input)
    else:  # Ensemble
        preds = np.mean([
            ridge.predict(X_input),
            rf.predict(X_input),
            gb.predict(X_input)
        ], axis=0)
        return preds

# ===== MODEL EVALUATION =====
if show_metrics:
    st.divider()
    st.subheader("📈 Model Evaluation Metrics")
    
    try:
        preds_ridge = ridge.predict(X_test)
        preds_rf = rf.predict(X_test)
        preds_gb = gb.predict(X_test)
        preds_ensemble = np.mean([preds_ridge, preds_rf, preds_gb], axis=0)
        
        # Calculate metrics
        metrics_data = {
            "Model": ["Ridge", "Random Forest", "Gradient Boosting", "Ensemble"],
            "MAE": [
                mean_absolute_error(y_test, preds_ridge),
                mean_absolute_error(y_test, preds_rf),
                mean_absolute_error(y_test, preds_gb),
                mean_absolute_error(y_test, preds_ensemble)
            ],
            "RMSE": [
                np.sqrt(mean_squared_error(y_test, preds_ridge)),
                np.sqrt(mean_squared_error(y_test, preds_rf)),
                np.sqrt(mean_squared_error(y_test, preds_gb)),
                np.sqrt(mean_squared_error(y_test, preds_ensemble))
            ],
            "R²": [
                r2_score(y_test, preds_ridge),
                r2_score(y_test, preds_rf),
                r2_score(y_test, preds_gb),
                r2_score(y_test, preds_ensemble)
            ]
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(
            metrics_df.style.format({
                "MAE": "{:.6f}",
                "RMSE": "{:.6f}",
                "R²": "{:.4f}"
            }).highlight_max(subset=["R²"], color="#d4edda"),
            use_container_width=True
        )
    
    except Exception as e:
        st.error(f"❌ Error calculating metrics: {str(e)}")

# ===== TEST PREDICTIONS VISUALIZATION =====
if show_predictions:
    st.divider()
    st.subheader(f"📉 {model_choice} — Test Predictions vs Actual")
    
    try:
        preds_scaled = predict_model(model_choice, X_test)
        preds_actual = scaler_y.inverse_transform(
            preds_scaled.reshape(-1, 1)
        ).flatten()
        
        # Create interactive plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            y=y_test_orig.values,
            name='Actual Price',
            line=dict(color='#1f77b4', width=2),
            mode='lines'
        ))
        
        fig.add_trace(go.Scatter(
            y=preds_actual,
            name='Predicted Price',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            mode='lines'
        ))
        
        fig.update_layout(
            title=f"{model_choice} — Test Set Predictions",
            xaxis_title="Time Period",
            yaxis_title="Bitcoin Price (USD)",
            height=450,
            hovermode='x unified',
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show prediction error
        mae_test = mean_absolute_error(y_test_orig, preds_actual)
        rmse_test = np.sqrt(mean_squared_error(y_test_orig, preds_actual))
        st.metric("Test MAE", f"${mae_test:,.2f}")
        st.metric("Test RMSE", f"${rmse_test:,.2f}")
    
    except Exception as e:
        st.error(f"❌ Error generating predictions: {str(e)}")

# ===== FUTURE FORECAST =====
st.divider()
st.subheader("🔮 Future Price Forecast")

col1, col2 = st.columns([2, 1])

with col1:
    future_date = st.date_input(
        "Select Future Date",
        value=datetime.date.today() + datetime.timedelta(days=7),
        min_value=datetime.date.today() + datetime.timedelta(days=1)
    )

with col2:
    forecast_btn = st.button("🚀 Predict Future Price", use_container_width=True)

if forecast_btn:
    try:
        last_date = df["date"].max().date() if "date" in df.columns else df.index[-1]
        
        if future_date <= last_date:
            st.warning("⚠️ Please select a date beyond the dataset end date.")
        else:
            # Prepare input for prediction
            last_row = X.iloc[-1].values.reshape(1, -1)
            X_future = scaler_X.transform(last_row)
            
            # Make prediction
            pred_scaled = predict_model(model_choice, X_future)
            pred_price = scaler_y.inverse_transform(
                pred_scaled.reshape(-1, 1)
            ).flatten()[0]
            
            # Calculate change
            current_price = y_test_orig.iloc[-1] if len(y_test_orig) > 0 else y.iloc[-1]
            price_change = pred_price - current_price
            pct_change = (price_change / current_price) * 100
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="prediction-box">
                    <h3>Predicted Bitcoin Price</h3>
                    <h2>${pred_price:,.2f}</h2>
                    <p>on {future_date.strftime('%B %d, %Y')}</p>
                    <hr>
                    <p><strong>Model:</strong> {model_choice}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                direction = "📈" if price_change > 0 else "📉" if price_change < 0 else "➡️"
                color = "#4caf50" if price_change > 0 else "#f44336" if price_change < 0 else "#ff9800"
                
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, {color} 0%, {color} 100%);">
                    <h3>Expected Change</h3>
                    <h2>{direction} ${abs(price_change):,.2f}</h2>
                    <p>({pct_change:+.2f}%)</p>
                    <hr>
                    <p><strong>From:</strong> ${current_price:,.2f}</p>
                </div>
                """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"❌ Prediction Error: {str(e)}")
        st.info("💡 Try ensuring your data has at least 30 days of records.")