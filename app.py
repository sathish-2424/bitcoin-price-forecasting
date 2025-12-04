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

# ===== CUSTOM CSS (FINANCIAL DASHBOARD THEME) =====
st.markdown("""
    <style>
    /* Main Background adjustments */
    .main {
        background-color: #0E1117;
    }
    
    /* Custom Card Style */
    .metric-card {
        background-color: #1E1E1E;
        border: 1px solid #333333;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Text Highlights */
    .highlight-text {
        color: #F7931A; /* Bitcoin Orange */
        font-weight: bold;
    }
    
    /* Success/Fail colors for metrics */
    .metric-up { color: #00CC96; font-weight: bold; }
    .metric-down { color: #EF553B; font-weight: bold; }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
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

# ===== SIDEBAR (UPDATED: Removed Upload Section) =====
with st.sidebar:
    st.title("⚡ BTC-AI")
    st.caption("XGBoost Recursive Forecasting")
    st.divider()
    
    # Removed the File Uploader here as requested
    
    st.subheader("⚙️ Model Parameters")
    n_estimators = st.slider("Trees (Estimators)", 50, 500, 200, step=50)
    learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.05, step=0.01)
    
    st.info("ℹ️ System auto-loads 'dataset.csv' or uses live simulation data.")

# ===== DATA LOADING LOGIC =====
@st.cache_data
def load_and_clean_data(file_path):
    try:
        df = pd.read_csv(file_path, low_memory=False)
        
        # Handle metadata rows common in Yahoo Finance exports
        first_val = str(df.iloc[0,0]) if len(df) > 0 else ""
        if "Ticker" in first_val or "Info" in first_val:
            df = df.iloc[2:].copy()
            df = df.iloc[:, :2]
            df.columns = ["date", "price"]
        else:
            df.columns = [c.strip().lower() for c in df.columns]
            date_cols = [c for c in df.columns if "date" in c or "time" in c]
            price_cols = [c for c in df.columns if "price" in c or "close" in c or "value" in c]
            
            if not price_cols or not date_cols:
                raise ValueError("Columns not found")
            
            df = df.rename(columns={date_cols[0]: "date", price_cols[0]: "price"})

        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["price", "date"])
        df = df.sort_values("date").reset_index(drop=True)
        
        if len(df) < 60: raise ValueError("Insufficient data points (<60)")
        return df
    
    except Exception as e:
        return None

# Load Data (Auto-detection)
df = None
if os.path.exists("dataset.csv"):
    df = load_and_clean_data("dataset.csv")

# Demo Data Fallback (If no local file exists)
if df is None:
    dates = pd.date_range(start="2021-01-01", end=datetime.date.today())
    n_days = len(dates)
    # Create a realistic looking trend
    prices = 40000 + np.cumsum(np.random.normal(0, 800, n_days)) 
    df = pd.DataFrame({"date": dates, "price": prices})
    if os.path.exists("dataset.csv"):
        st.sidebar.warning("⚠️ Could not load dataset.csv, using demo data.")

# ===== FEATURE ENGINEERING =====
def create_features(data):
    df_temp = data.copy()
    df_temp["lag_1"] = df_temp["price"].shift(1)
    df_temp["lag_7"] = df_temp["price"].shift(7)
    df_temp["ma_7"] = df_temp["price"].rolling(window=7).mean()
    df_temp["ma_30"] = df_temp["price"].rolling(window=30).mean()
    df_temp["volatility"] = df_temp["price"].rolling(window=7).std()
    return df_temp.dropna()

df_feat = create_features(df)
X = df_feat.drop(columns=["date", "price"], errors="ignore")
y = df_feat["price"]

# Scaling & Split
split_idx = int(0.95 * len(X))
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

scaler_X = MinMaxScaler().fit(X_train)
scaler_y = MinMaxScaler().fit(y_train.values.reshape(-1, 1))

X_train_s = scaler_X.transform(X_train)
X_test_s = scaler_X.transform(X_test)
y_train_s = scaler_y.transform(y_train.values.reshape(-1, 1)).flatten()

# ===== TABS LAYOUT =====
tab1, tab2, tab3 = st.tabs(["📊 Market Overview", "🧠 Model Intelligence", "🔮 Future Simulator"])

# ----- TAB 1: MARKET OVERVIEW -----
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    curr_price = df["price"].iloc[-1]
    prev_price = df["price"].iloc[-2]
    change = curr_price - prev_price
    pct_change = (change / prev_price) * 100
    
    with col1: st.metric("Current Price", f"${curr_price:,.2f}", f"{pct_change:.2f}%")
    with col2: st.metric("All-Time High", f"${df['price'].max():,.2f}")
    with col3: st.metric("Volatility (7D)", f"${df_feat['volatility'].iloc[-1]:,.2f}")
    with col4: st.metric("Data Points", f"{len(df):,}")
    
    st.divider()
    
    # Financial Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["price"],
        mode='lines', name='Close Price',
        line=dict(color='#F7931A', width=2),
        fill='tozeroy', fillcolor='rgba(247, 147, 26, 0.1)'
    ))
    
    fig.update_layout(
        title="Historical Price Action",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=500,
        hovermode="x unified",
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig, use_container_width=True)

# ----- TAB 2: MODEL PERFORMANCE -----
with tab2:
    # Train Model
    model = xgb.XGBRegressor(
        n_estimators=n_estimators, learning_rate=learning_rate, max_depth=5,
        min_child_weight=1, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1
    )
    model.fit(X_train_s, y_train_s)
    
    # Evaluate
    preds_s = model.predict(X_test_s)
    preds = scaler_y.inverse_transform(preds_s.reshape(-1, 1)).flatten()
    
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Model Accuracy")
        st.write(f"This XGBoost model explains **{r2*100:.1f}%** of the price variance on unseen data.")
        st.info(f"Mean Absolute Error: **${mae:,.2f}**")
        
    with c2:
        # Feature Importance
        imp = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
        imp = imp.sort_values("Importance", ascending=True)
        fig_imp = go.Figure(go.Bar(
            x=imp["Importance"], y=imp["Feature"], orientation='h',
            marker_color='#00CC96'
        ))
        fig_imp.update_layout(title="Feature Importance", template="plotly_dark", height=300, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_imp, use_container_width=True)
    
    st.divider()
    
    # Actual vs Predicted Chart
    fig_val = go.Figure()
    fig_val.add_trace(go.Scatter(y=y_test.values, name="Actual", line=dict(color="#00CC96", width=2)))
    fig_val.add_trace(go.Scatter(y=preds, name="Predicted", line=dict(color="#EF553B", width=2, dash='dot')))
    fig_val.update_layout(
        title="Validation: Recent Accuracy Check",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        height=400
    )
    st.plotly_chart(fig_val, use_container_width=True)

# ----- TAB 3: FUTURE SIMULATOR -----
with tab3:
    st.markdown("### 🔮 Recursive Pricing Engine")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("""
        <div class="metric-card">
            <h4>Simulation Controls</h4>
            <p style="font-size: 0.8em; color: gray;">Select a target date. The model will predict one day ahead, re-calculate indicators, and predict the next day recursively.</p>
        </div>
        """, unsafe_allow_html=True)
        
        last_date = df["date"].max().date()
        target_date = st.date_input("Target Forecast Date", 
                                    value=last_date + datetime.timedelta(days=14),
                                    min_value=last_date + datetime.timedelta(days=1))
        
        run_sim = st.button("▶ Run Simulation", use_container_width=True, type="primary")

    if run_sim:
        with st.spinner("Running Recursive Neural Simulation..."):
            # Simulation Logic
            curr_df = df.copy().tail(100).reset_index(drop=True)
            future_preds = []
            loop_date = last_date + datetime.timedelta(days=1)
            
            progress_bar = st.progress(0)
            total_days = (target_date - last_date).days
            day_count = 0
            
            while loop_date <= target_date:
                # Feature calc
                f_df = create_features(curr_df)
                last_row = f_df.iloc[-1].drop(["date", "price"], errors="ignore")
                
                # Predict
                X_in = scaler_X.transform(last_row.values.reshape(1, -1))
                p_scaled = model.predict(X_in)
                p_price = scaler_y.inverse_transform(p_scaled.reshape(-1, 1))[0][0]
                
                # Update
                new_row = pd.DataFrame({"date": [pd.to_datetime(loop_date)], "price": [p_price]})
                curr_df = pd.concat([curr_df, new_row], ignore_index=True)
                future_preds.append({"date": loop_date, "price": p_price})
                
                loop_date += datetime.timedelta(days=1)
                day_count += 1
                progress_bar.progress(min(day_count / total_days, 1.0))
            
            # Results
            res_df = pd.DataFrame(future_preds)
            final_p = res_df["price"].iloc[-1]
            start_p = df["price"].iloc[-1]
            diff = final_p - start_p
            pct_d = (diff / start_p) * 100
            
            with c2:
                # Result Metrics
                rc1, rc2 = st.columns(2)
                color_cls = "metric-up" if diff > 0 else "metric-down"
                arrow = "▲" if diff > 0 else "▼"
                
                with rc1:
                    st.markdown(f"""
                    <div class="metric-card" style="text-align: center;">
                        <div style="font-size: 1.2em; color: gray;">Projected Price</div>
                        <div style="font-size: 2.5em; font-weight: bold; color: white;">${final_p:,.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with rc2:
                     st.markdown(f"""
                    <div class="metric-card" style="text-align: center;">
                        <div style="font-size: 1.2em; color: gray;">Expected Move</div>
                        <div class="{color_cls}" style="font-size: 2.5em;">{arrow} {pct_d:.2f}%</div>
                        <div style="color: gray;">${abs(diff):,.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Forecast Plot
        st.markdown("---")
        fig_fut = go.Figure()
        
        # Historical Tail
        hist = df.tail(90)
        fig_fut.add_trace(go.Scatter(x=hist["date"], y=hist["price"], name="Historical", line=dict(color="gray", width=1)))
        
        # Forecast
        fig_fut.add_trace(go.Scatter(x=res_df["date"], y=res_df["price"], name="Forecast", 
                                    line=dict(color="#F7931A", width=3), mode='lines+markers'))
        
        fig_fut.update_layout(
            title="Projected Trajectory",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode="x unified"
        )
        st.plotly_chart(fig_fut, use_container_width=True)