import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
import datetime
import plotly.graph_objects as go
import warnings
import os

warnings.filterwarnings("ignore")

# ===== PAGE CONFIGURATION =====
st.set_page_config(
    page_title="BTC-AI | Price Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="⚡"
)

# ===== STYLE CONFIGURATION =====
def load_css():
    st.markdown("""
        <style>
        .main { background-color: #0E1117; }
        .metric-card {
            background-color: #1E1E1E;
            border: 1px solid #333333;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        .metric-up { color: #00CC96; font-weight: bold; }
        .metric-down { color: #EF553B; font-weight: bold; }
        .stTabs [data-baseweb="tab-list"] { gap: 10px; }
        .stTabs [data-baseweb="tab"] {
            background-color: #1E1E1E;
            border-radius: 5px;
            color: white;
            padding: 10px 20px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #F7931A !important;
            color: black !important;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

load_css()

# ===== DATA FUNCTIONS =====
@st.cache_data
def generate_demo_data():
    """Generates realistic demo data if no file is found."""
    dates = pd.date_range(start="2021-01-01", end=datetime.date.today())
    n_days = len(dates)
    # Random walk with drift
    returns = np.random.normal(0.0005, 0.02, n_days)
    price_path = 40000 * np.cumprod(1 + returns)
    return pd.DataFrame({"date": dates, "price": price_path})

@st.cache_data
def load_data(file_path):
    """Loads and cleans data efficiently."""
    try:
        if not os.path.exists(file_path):
            return None
        
        df = pd.read_csv(file_path, low_memory=False)
        
        # Cleanup headers (Yahoo Finance specific)
        if any(x in str(df.iloc[0,0]) for x in ["Ticker", "Info"]):
            df = df.iloc[2:].copy()
            df = df.iloc[:, :2]
        
        # Standardize columns
        df.columns = ["date", "price"]
        
        # Types conversion
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        
        df = df.dropna().sort_values("date").reset_index(drop=True)
        
        if len(df) < 60: return None
        return df
    except:
        return None

@st.cache_data
def feature_engineering(data):
    """Calculates technical indicators."""
    df_temp = data.copy()
    df_temp["lag_1"] = df_temp["price"].shift(1)
    df_temp["lag_7"] = df_temp["price"].shift(7)
    df_temp["ma_7"] = df_temp["price"].rolling(window=7).mean()
    df_temp["ma_30"] = df_temp["price"].rolling(window=30).mean()
    df_temp["volatility"] = df_temp["price"].rolling(window=7).std()
    return df_temp.dropna()

# ===== MODEL FUNCTIONS =====
@st.cache_resource
def train_model(X, y, params):
    """Trains the XGBoost model. Cached to avoid re-training on UI interaction."""
    model = xgb.XGBRegressor(
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
        max_depth=5,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    return model

# ===== SIDEBAR =====
with st.sidebar:
    st.title("⚡ BTC-AI")
    st.caption("XGBoost Recursive Forecasting")
    st.divider()
    
    st.subheader("⚙️ Hyperparameters")
    n_estimators = st.slider("Trees (Estimators)", 50, 500, 200, step=50)
    learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.05, step=0.01)
    
    st.info("ℹ️ Optimization: Model is cached and only retrains when sliders move.")

# ===== MAIN LOGIC =====

# 1. Load Data
df = load_data("dataset.csv")
if df is None:
    df = generate_demo_data()
    if os.path.exists("dataset.csv"):
        st.sidebar.warning("⚠️ Loaded demo data (dataset.csv invalid/empty).")

# 2. Features & Scaling
df_feat = feature_engineering(df)
X_raw = df_feat.drop(columns=["date", "price"], errors="ignore")
y_raw = df_feat["price"]

# Split
split_idx = int(0.95 * len(X_raw))
X_train, X_test = X_raw.iloc[:split_idx], X_raw.iloc[split_idx:]
y_train, y_test = y_raw.iloc[:split_idx], y_raw.iloc[split_idx:]

# Scale (Fit on Train only to prevent leakage)
scaler_X = MinMaxScaler().fit(X_train)
scaler_y = MinMaxScaler().fit(y_train.values.reshape(-1, 1))

X_train_s = scaler_X.transform(X_train)
X_test_s = scaler_X.transform(X_test)
y_train_s = scaler_y.transform(y_train.values.reshape(-1, 1)).flatten()

# 3. Model Training (Cached)
params = {'n_estimators': n_estimators, 'learning_rate': learning_rate}
model = train_model(X_train_s, y_train_s, params)

# ===== TABS =====
tab1, tab2, tab3 = st.tabs(["📊 Market Overview", "🧠 Model Intelligence", "🔮 Future Simulator"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    curr_price = df["price"].iloc[-1]
    pct_change = ((curr_price - df["price"].iloc[-2]) / df["price"].iloc[-2]) * 100
    
    with col1: st.metric("Current Price", f"${curr_price:,.2f}", f"{pct_change:.2f}%")
    with col2: st.metric("All-Time High", f"${df['price'].max():,.2f}")
    with col3: st.metric("Volatility (7D)", f"${df_feat['volatility'].iloc[-1]:,.2f}")
    with col4: st.metric("Data Points", f"{len(df):,}")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["price"], mode='lines', name='Price',
        line=dict(color='#F7931A', width=2), fill='tozeroy', fillcolor='rgba(247, 147, 26, 0.1)'
    ))
    fig.update_layout(title="Historical Price Action", template="plotly_dark", height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    preds_s = model.predict(X_test_s)
    preds = scaler_y.inverse_transform(preds_s.reshape(-1, 1)).flatten()
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Model Accuracy")
        st.write(f"The model explains **{r2*100:.1f}%** of price variance.")
        st.info(f"Mean Absolute Error: **${mae:,.2f}**")
        
    with c2:
        imp = pd.DataFrame({"Feature": X_raw.columns, "Importance": model.feature_importances_}).sort_values("Importance", ascending=True)
        fig_imp = go.Figure(go.Bar(x=imp["Importance"], y=imp["Feature"], orientation='h', marker_color='#00CC96'))
        fig_imp.update_layout(title="Feature Importance", template="plotly_dark", height=300, margin=dict(l=0,r=0,t=30,b=0), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_imp, use_container_width=True)

    fig_val = go.Figure()
    fig_val.add_trace(go.Scatter(y=y_test.values, name="Actual", line=dict(color="#00CC96")))
    fig_val.add_trace(go.Scatter(y=preds, name="Predicted", line=dict(color="#EF553B", dash='dot')))
    fig_val.update_layout(title="Validation Set Performance", template="plotly_dark", height=400, paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_val, use_container_width=True)

with tab3:
    st.markdown("### 🔮 Recursive Pricing Engine")
    
    last_date = df["date"].iloc[-1].date()
    target_date = st.date_input("Target Date", value=last_date + datetime.timedelta(days=14), min_value=last_date + datetime.timedelta(days=1))
    
    if st.button("▶ Run Simulation", type="primary"):
        with st.spinner("Running Recursive Neural Simulation..."):
            
            # OPTIMIZED LOOP LOGIC
            # Only keep the necessary tail to calculate max rolling window (30 days) + buffer
            window_buffer = 40 
            curr_df = df.copy().tail(window_buffer) 
            future_preds = []
            loop_date = last_date + datetime.timedelta(days=1)
            total_days = (target_date - last_date).days
            
            progress = st.progress(0)
            
            for i in range(total_days):
                # 1. Feature Engineering on the small window (fast)
                f_df = feature_engineering(curr_df)
                last_row = f_df.iloc[-1].drop(["date", "price"], errors="ignore")
                
                # 2. Predict
                X_in = scaler_X.transform(last_row.values.reshape(1, -1))
                p_scaled = model.predict(X_in)
                p_price = scaler_y.inverse_transform(p_scaled.reshape(-1, 1))[0][0]
                
                # 3. Append & Shift
                new_row = pd.DataFrame({"date": [pd.to_datetime(loop_date)], "price": [p_price]})
                curr_df = pd.concat([curr_df, new_row], ignore_index=True).tail(window_buffer) # Keep df small!
                
                future_preds.append({"date": loop_date, "price": p_price})
                loop_date += datetime.timedelta(days=1)
                progress.progress((i + 1) / total_days)

            # Results
            res_df = pd.DataFrame(future_preds)
            final_p = res_df["price"].iloc[-1]
            start_p = df["price"].iloc[-1]
            diff = final_p - start_p
            
            # Visualization
            c1, c2 = st.columns(2)
            color_cls = "metric-up" if diff > 0 else "metric-down"
            arrow = "▲" if diff > 0 else "▼"
            
            with c1:
                st.markdown(f"""<div class="metric-card" style="text-align:center">
                    <div style="color:gray">Projected Price</div>
                    <div style="font-size:2em; font-weight:bold">${final_p:,.2f}</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="metric-card" style="text-align:center">
                    <div style="color:gray">Expected Move</div>
                    <div class="{color_cls}" style="font-size:2em">{arrow} {(diff/start_p)*100:.2f}%</div>
                </div>""", unsafe_allow_html=True)

            fig_fut = go.Figure()
            hist = df.tail(60)
            fig_fut.add_trace(go.Scatter(x=hist["date"], y=hist["price"], name="History", line=dict(color="gray")))
            fig_fut.add_trace(go.Scatter(x=res_df["date"], y=res_df["price"], name="Forecast", line=dict(color="#F7931A", width=3)))
            fig_fut.update_layout(title="Projected Trajectory", template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_fut, use_container_width=True)