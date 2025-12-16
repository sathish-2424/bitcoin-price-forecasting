import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import datetime
import plotly.graph_objects as go
import warnings

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
        .topic-badge {
            background-color: #262730;
            color: #F7931A;
            padding: 5px 10px;
            border-radius: 15px;
            border: 1px solid #F7931A;
            font-size: 0.9em;
            margin-right: 10px;
            display: inline-block;
        }
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

# ===== DATA FUNCTIONS (LIVE) =====
@st.cache_data(ttl=3600)
def load_live_data():
    """Fetches live data from Yahoo Finance (BTC-USD)."""
    try:
        ticker = "BTC-USD"
        df = yf.download(ticker, period="5y", interval="1d", progress=False)
        df = df.reset_index()
        
        # Robust Column Handling
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            
        cols_map = {col: col.lower() for col in df.columns}
        df.rename(columns=cols_map, inplace=True)
        
        if 'datetime' in df.columns:
            df.rename(columns={'datetime': 'date'}, inplace=True)
        
        if 'close' not in df.columns:
             return None

        df = df[['date', 'close']].rename(columns={'close': 'price'})
        
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df = df.dropna().sort_values("date").reset_index(drop=True)
        
        return df
    except Exception as e:
        st.error(f"API Connection Error: {e}")
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

def analyze_forecast_topic(start_price, end_price, path_data):
    """Generates a text topic/theme for the forecast."""
    pct_change = ((end_price - start_price) / start_price) * 100
    
    # Trend Topic
    if pct_change > 5: topic = "Strong Bullish Breakout"
    elif pct_change > 1: topic = "Moderate Uptrend"
    elif pct_change > -1: topic = "Consolidation / Neutral"
    elif pct_change > -5: topic = "Moderate Correction"
    else: topic = "Bearish Reversal"
    
    # Volatility Topic
    volatility = np.std(path_data)
    if volatility > (start_price * 0.05): risk = "High Volatility"
    else: risk = "Stable Accumulation"
    
    return topic, risk

# ===== MODEL FUNCTIONS =====
@st.cache_resource
def train_model(X, y):
    """Trains the XGBoost model with optimized defaults."""
    model = xgb.XGBRegressor(
        n_estimators=500,     
        learning_rate=0.01,   
        max_depth=5,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    return model

# ===== MAIN LOGIC =====

# 1. Load Data (LIVE)
with st.spinner("Connecting to live market data..."):
    df = load_live_data()

if df is None or df.empty:
    st.error("Could not fetch data. Please check your internet connection.")
    st.stop()

# 2. Features & Scaling
df_feat = feature_engineering(df)
X_raw = df_feat.drop(columns=["date", "price"], errors="ignore")
y_raw = df_feat["price"]

# Split
split_idx = int(0.95 * len(X_raw))
X_train, X_test = X_raw.iloc[:split_idx], X_raw.iloc[split_idx:]
y_train, y_test = y_raw.iloc[:split_idx], y_raw.iloc[split_idx:]

# Scale
scaler_X = MinMaxScaler().fit(X_train)
scaler_y = MinMaxScaler().fit(y_train.values.reshape(-1, 1))

X_train_s = scaler_X.transform(X_train)
X_test_s = scaler_X.transform(X_test)
y_train_s = scaler_y.transform(y_train.values.reshape(-1, 1)).flatten()

# 3. Model Training
model = train_model(X_train_s, y_train_s)

# ===== DASHBOARD HEADER =====
st.header("Forecast Intelligence")

# ===== TABS =====
tab1, tab2, tab3 = st.tabs(["📊 Market Overview", "🧠 Model Intelligence", "🔮 Future Simulator"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    curr_price = df["price"].iloc[-1]
    prev_price = df["price"].iloc[-2]
    pct_change = ((curr_price - prev_price) / prev_price) * 100
    
    with col1: st.metric("Live Price", f"${curr_price:,.2f}", f"{pct_change:.2f}%")
    with col2: st.metric("All-Time High", f"${df['price'].max():,.2f}")
    with col3: st.metric("Volatility (7D)", f"${df_feat['volatility'].iloc[-1]:,.2f}")
    with col4: st.metric("Data Points", f"{len(df):,}")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["price"], mode='lines', name='Price',
        line=dict(color='#F7931A', width=2), fill='tozeroy', fillcolor='rgba(247, 147, 26, 0.1)'
    ))
    fig.update_layout(title="Live Price Action (BTC-USD)", template="plotly_dark", height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    preds_s = model.predict(X_test_s)
    preds = scaler_y.inverse_transform(preds_s.reshape(-1, 1)).flatten()
    
    # Validation Graph
    fig_val = go.Figure()
    fig_val.add_trace(go.Scatter(y=y_test.values, name="Actual", line=dict(color="#00CC96")))
    fig_val.add_trace(go.Scatter(y=preds, name="Predicted", line=dict(color="#EF553B", dash='dot')))
    fig_val.update_layout(title="Validation Set Performance", template="plotly_dark", height=500, paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_val, use_container_width=True)

with tab3:
    st.markdown("### 🔮 Recursive Pricing Engine")
    
    last_date = df["date"].iloc[-1].date()
    target_date = st.date_input("Target Date", value=last_date + datetime.timedelta(days=14), min_value=last_date + datetime.timedelta(days=1))
    
    if st.button("▶ Run Simulation", type="primary"):
        with st.spinner(f"Simulating market from {last_date} to {target_date}..."):
            
            # Recursive Logic
            window_buffer = 40 
            curr_df = df.copy().tail(window_buffer) 
            future_preds = []
            loop_date = last_date + datetime.timedelta(days=1)
            total_days = (target_date - last_date).days
            
            progress = st.progress(0)
            
            for i in range(total_days):
                try:
                    # 1. Feature Engineering
                    f_df = feature_engineering(curr_df)
                    if f_df.empty: break
                    
                    last_row = f_df.iloc[-1].drop(["date", "price"], errors="ignore")
                    
                    # 2. Predict
                    X_in = scaler_X.transform(last_row.values.reshape(1, -1))
                    p_scaled = model.predict(X_in)
                    p_price = scaler_y.inverse_transform(p_scaled.reshape(-1, 1))[0][0]
                    
                    # 3. Append & Shift
                    new_row = pd.DataFrame({"date": [pd.to_datetime(loop_date)], "price": [p_price]})
                    curr_df = pd.concat([curr_df, new_row], ignore_index=True).tail(window_buffer)
                    
                    future_preds.append({"date": loop_date, "price": p_price})
                    loop_date += datetime.timedelta(days=1)
                    progress.progress((i + 1) / total_days)
                except Exception as e:
                    break

            if future_preds:
                # Results
                res_df = pd.DataFrame(future_preds)
                final_p = res_df["price"].iloc[-1]
                start_p = df["price"].iloc[-1]
                diff = final_p - start_p
                
                # Analyze Topics
                trend_topic, risk_topic = analyze_forecast_topic(start_p, final_p, res_df["price"].values)
                
                # Visualization
                c1, c2 = st.columns(2)
                color_cls = "metric-up" if diff > 0 else "metric-down"
                arrow = "▲" if diff > 0 else "▼"
                
                with c1:
                    st.markdown(f"""<div class="metric-card" style="text-align:center">
                        <div style="color:gray">Projected Price</div>
                        <div style="font-size:2em; font-weight:bold">${final_p:,.2f}</div>
                        <div style="margin-top:10px;">
                            <span class="topic-badge">{trend_topic}</span>
                        </div>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""<div class="metric-card" style="text-align:center">
                        <div style="color:gray">Expected Move</div>
                        <div class="{color_cls}" style="font-size:2em">{arrow} {(diff/start_p)*100:.2f}%</div>
                        <div style="margin-top:10px;">
                            <span class="topic-badge">{risk_topic}</span>
                        </div>
                    </div>""", unsafe_allow_html=True)
                
                st.write("") 

                fig_fut = go.Figure()
                hist = df.tail(60)
                fig_fut.add_trace(go.Scatter(x=hist["date"], y=hist["price"], name="History", line=dict(color="gray")))
                fig_fut.add_trace(go.Scatter(x=res_df["date"], y=res_df["price"], name="Forecast", line=dict(color="#F7931A", width=3)))
                fig_fut.update_layout(title="Projected Trajectory", template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_fut, use_container_width=True)