"""
BTC-AI | Price Intelligence Dashboard
Optimized Streamlit application for Bitcoin price forecasting using XGBoost.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from datetime import date, timedelta
from functools import lru_cache
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore")

# ===== CONSTANTS =====
TICKER = "BTC-USD"
DATA_PERIOD = "5y"
TRAIN_SPLIT_RATIO = 0.95
WINDOW_BUFFER = 60
CACHE_TTL = 3600
FEATURE_COLS = ("lag_1", "lag_7", "ma_7", "ma_30", "volatility")

MODEL_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.01,
    "max_depth": 5,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
}

COLORS = {
    "primary": "#F7931A",
    "success": "#00CC96",
    "danger": "#EF553B",
    "gray": "gray",
    "bg_dark": "#1E1E1E",
    "bg_main": "#0E1117",
    "border": "#333333",
}

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="BTC-AI | Price Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="⚡",
)

# ===== STYLES =====
st.markdown(f"""
<style>
.main {{ background-color: {COLORS['bg_main']}; }}
.metric-card {{
    background-color: {COLORS['bg_dark']};
    border: 1px solid {COLORS['border']};
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    text-align: center;
}}
.metric-up {{ color: {COLORS['success']}; font-weight: bold; }}
.metric-down {{ color: {COLORS['danger']}; font-weight: bold; }}
.topic-badge {{
    background-color: #262730;
    color: {COLORS['primary']};
    padding: 5px 10px;
    border-radius: 15px;
    border: 1px solid {COLORS['primary']};
    font-size: 0.9em;
    display: inline-block;
}}
.stTabs [data-baseweb="tab-list"] {{ gap: 10px; }}
.stTabs [data-baseweb="tab"] {{
    background-color: {COLORS['bg_dark']};
    border-radius: 5px;
    color: white;
    padding: 10px 20px;
}}
.stTabs [aria-selected="true"] {{
    background-color: {COLORS['primary']} !important;
    color: black !important;
    font-weight: bold;
}}
</style>
""", unsafe_allow_html=True)


# ===== SESSION STATE INIT =====
def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "last_target_date": None,
        "simulation_result": None,
        "simulation_target_date": None,
        "simulation_start_price": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ===== UTILITY FUNCTIONS =====
def pct_change(current: float, previous: float) -> float:
    """Calculate percentage change."""
    return ((current - previous) / previous * 100) if previous else 0.0


def chart_layout(title: str, height: int = 500) -> dict:
    """Create chart layout config."""
    return {
        "title": title,
        "template": "plotly_dark",
        "height": height,
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
    }


# ===== DATA FUNCTIONS =====
@st.cache_data(ttl=CACHE_TTL)
def load_data() -> pd.DataFrame | None:
    """Fetch BTC-USD data from Yahoo Finance."""
    try:
        df = yf.download(TICKER, period=DATA_PERIOD, interval="1d", progress=False)
        if df.empty:
            return None

        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        df.columns = df.columns.str.lower()

        # Normalize column names
        if "datetime" in df.columns:
            df.rename(columns={"datetime": "date"}, inplace=True)

        if "close" not in df.columns:
            st.error("Missing 'Close' price in data.")
            return None

        # Select and clean
        result = df[["date", "close"]].rename(columns={"close": "price"})
        result["date"] = pd.to_datetime(result["date"]).dt.tz_localize(None)
        result["price"] = pd.to_numeric(result["price"], errors="coerce")

        return result.dropna().sort_values("date").reset_index(drop=True)

    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return None


@st.cache_data
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical features - vectorized."""
    result = df.copy()
    price = result["price"]
    
    result["lag_1"] = price.shift(1)
    result["lag_7"] = price.shift(7)
    result["ma_7"] = price.rolling(7, min_periods=1).mean()
    result["ma_30"] = price.rolling(30, min_periods=1).mean()
    result["volatility"] = price.rolling(7, min_periods=1).std()
    
    return result.dropna()


def compute_features_inline(df: pd.DataFrame) -> pd.DataFrame:
    """Fast inline feature computation for simulation loop."""
    price = df["price"].values
    n = len(price)
    
    # Pre-allocate arrays
    lag_1 = np.empty(n)
    lag_7 = np.empty(n)
    ma_7 = np.empty(n)
    ma_30 = np.empty(n)
    vol = np.empty(n)
    
    lag_1[0] = np.nan
    lag_1[1:] = price[:-1]
    
    lag_7[:7] = np.nan
    lag_7[7:] = price[:-7]
    
    # Rolling computations using cumsum for efficiency
    cumsum = np.cumsum(np.insert(price, 0, 0))
    
    for i in range(n):
        start_7 = max(0, i - 6)
        start_30 = max(0, i - 29)
        ma_7[i] = (cumsum[i + 1] - cumsum[start_7]) / (i - start_7 + 1)
        ma_30[i] = (cumsum[i + 1] - cumsum[start_30]) / (i - start_30 + 1)
        
        window = price[start_7:i + 1]
        vol[i] = np.std(window) if len(window) > 1 else 0
    
    result = df.copy()
    result["lag_1"] = lag_1
    result["lag_7"] = lag_7
    result["ma_7"] = ma_7
    result["ma_30"] = ma_30
    result["volatility"] = vol
    
    return result.dropna()


def classify_forecast(start: float, end: float, prices: np.ndarray) -> tuple[str, str]:
    """Classify trend and volatility."""
    change = pct_change(end, start)
    
    if change > 5:
        trend = "Strong Bullish Breakout"
    elif change > 1:
        trend = "Moderate Uptrend"
    elif change > -1:
        trend = "Consolidation / Neutral"
    elif change > -5:
        trend = "Moderate Correction"
    else:
        trend = "Bearish Reversal"
    
    vol = np.std(prices)
    risk = "High Volatility" if vol > start * 0.05 else "Stable Accumulation"
    
    return trend, risk


# ===== MODEL FUNCTIONS =====
@st.cache_data
def prepare_data(df_feat: pd.DataFrame) -> dict:
    """Prepare training data and scalers."""
    X = df_feat[list(FEATURE_COLS)].values
    y = df_feat["price"].values
    
    split = int(TRAIN_SPLIT_RATIO * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    scaler_X = MinMaxScaler().fit(X_train)
    scaler_y = MinMaxScaler().fit(y_train.reshape(-1, 1))
    
    return {
        "X_train": scaler_X.transform(X_train),
        "X_test": scaler_X.transform(X_test),
        "y_train": scaler_y.transform(y_train.reshape(-1, 1)).ravel(),
        "y_test": y_test,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
    }


@st.cache_resource
def train_model(_X: np.ndarray, _y: np.ndarray) -> xgb.XGBRegressor:
    """Train XGBoost model."""
    model = xgb.XGBRegressor(**MODEL_PARAMS)
    model.fit(_X, _y)
    return model


# ===== SIMULATION ENGINE =====
def run_simulation(
    df: pd.DataFrame,
    model: xgb.XGBRegressor,
    scaler_X: MinMaxScaler,
    scaler_y: MinMaxScaler,
    target_date: date,
) -> pd.DataFrame | None:
    """Run recursive simulation with volatility noise."""
    last_date = df["date"].iloc[-1].date()
    days = (target_date - last_date).days
    
    if days <= 0:
        st.warning("Target date must be in the future.")
        return None
    
    # Historical volatility & RNG setup
    hist_vol = df["price"].pct_change().std()
    rng = np.random.RandomState(int(target_date.strftime("%Y%m%d")))
    
    # Buffer setup
    buffer = df[["date", "price"]].tail(WINDOW_BUFFER).copy()
    predictions = []
    progress = st.progress(0)
    
    feature_cols = list(FEATURE_COLS)
    
    for i in range(days):
        try:
            # Compute features
            feat_df = compute_features_inline(buffer)
            if feat_df.empty:
                break
            
            # Extract and scale features
            features = feat_df[feature_cols].iloc[-1].values.reshape(1, -1)
            X_scaled = scaler_X.transform(features)
            
            # Predict
            pred_scaled = model.predict(X_scaled)
            pred_price = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
            
            # Add volatility noise
            noise = rng.normal(0, hist_vol * 0.5)
            pred_price *= (1 + noise)
            
            # Next date
            next_date = last_date + timedelta(days=i + 1)
            
            # Update buffer
            new_row = pd.DataFrame({"date": [pd.Timestamp(next_date)], "price": [pred_price]})
            buffer = pd.concat([buffer, new_row], ignore_index=True).tail(WINDOW_BUFFER)
            
            predictions.append({"date": next_date, "price": pred_price})
            progress.progress((i + 1) / days)
            
        except Exception as e:
            st.error(f"Simulation error at day {i + 1}: {e}")
            break
    
    progress.empty()
    return pd.DataFrame(predictions) if predictions else None


# ===== VISUALIZATION =====
def render_price_chart(df: pd.DataFrame) -> None:
    """Render price chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["price"],
        mode="lines", name="Price",
        line=dict(color=COLORS["primary"], width=2),
        fill="tozeroy", fillcolor="rgba(247, 147, 26, 0.1)",
    ))
    fig.update_layout(**chart_layout("Live Price Action (BTC-USD)"))
    st.plotly_chart(fig, use_container_width=True)


def render_validation_chart(actual: np.ndarray, predicted: np.ndarray) -> None:
    """Render validation chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=actual, name="Actual", line=dict(color=COLORS["success"])))
    fig.add_trace(go.Scatter(y=predicted, name="Predicted", line=dict(color=COLORS["danger"], dash="dot")))
    fig.update_layout(**chart_layout("Validation Set Performance"))
    st.plotly_chart(fig, use_container_width=True)


def render_forecast_chart(history: pd.DataFrame, forecast: pd.DataFrame) -> None:
    """Render forecast chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history["date"], y=history["price"], name="History", line=dict(color=COLORS["gray"])))
    fig.add_trace(go.Scatter(x=forecast["date"], y=forecast["price"], name="Forecast", line=dict(color=COLORS["primary"], width=3)))
    fig.update_layout(**chart_layout("Projected Trajectory", 450))
    st.plotly_chart(fig, use_container_width=True)


def render_metric_card(title: str, value: str, badge: str, css_class: str = "") -> None:
    """Render metric card."""
    st.markdown(f"""
    <div class="metric-card">
        <div style="color:gray">{title}</div>
        <div class="{css_class}" style="font-size:2em; font-weight:bold">{value}</div>
        <div style="margin-top:10px"><span class="topic-badge">{badge}</span></div>
    </div>
    """, unsafe_allow_html=True)


# ===== TAB RENDERERS =====
def render_market_tab(df: pd.DataFrame, df_feat: pd.DataFrame) -> None:
    """Render Market Overview tab."""
    curr, prev = df["price"].iloc[-1], df["price"].iloc[-2]
    change = pct_change(curr, prev)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Live Price", f"${curr:,.2f}", f"{change:.2f}%")
    c2.metric("All-Time High", f"${df['price'].max():,.2f}")
    c3.metric("Volatility (7D)", f"${df_feat['volatility'].iloc[-1]:,.2f}")
    c4.metric("Data Points", f"{len(df):,}")
    
    render_price_chart(df)


def render_model_tab(model: xgb.XGBRegressor, data: dict) -> None:
    """Render Model Intelligence tab."""
    preds_scaled = model.predict(data["X_test"])
    preds = data["scaler_y"].inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
    actual = data["y_test"]
    
    mae = mean_absolute_error(actual, preds)
    rmse = np.sqrt(mean_squared_error(actual, preds))
    r2 = r2_score(actual, preds)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"${mae:,.2f}")
    c2.metric("RMSE", f"${rmse:,.2f}")
    c3.metric("R² Score", f"{r2:.4f}")
    
    render_validation_chart(actual, preds)


def render_simulator_tab(df: pd.DataFrame, model: xgb.XGBRegressor, data: dict) -> None:
    """Render Future Simulator tab."""
    st.markdown("### 🔮 Recursive Pricing Engine")
    
    last_date = df["date"].iloc[-1].date()
    target = st.date_input(
        "Target Date",
        value=last_date + timedelta(days=14),
        min_value=last_date + timedelta(days=1),
        max_value=last_date + timedelta(days=365),
        key="target_date",
    )
    
    # Clear results on date change
    if st.session_state.last_target_date != target:
        st.session_state.simulation_result = None
        st.session_state.last_target_date = target
    
    if st.button("▶ Run Simulation", type="primary"):
        with st.spinner(f"Simulating {last_date} → {target}..."):
            result = run_simulation(df, model, data["scaler_X"], data["scaler_y"], target)
            st.session_state.simulation_result = result
            st.session_state.simulation_target_date = target
            st.session_state.simulation_start_price = df["price"].iloc[-1]
    
    # Display results
    result = st.session_state.simulation_result
    if result is not None and not result.empty:
        start = st.session_state.simulation_start_price
        final = result["price"].iloc[-1]
        move = pct_change(final, start)
        trend, risk = classify_forecast(start, final, result["price"].values)
        
        st.info(f"📅 Results for: **{st.session_state.simulation_target_date}**")
        
        c1, c2 = st.columns(2)
        arrow = "▲" if final > start else "▼"
        css = "metric-up" if final > start else "metric-down"
        
        with c1:
            render_metric_card("Projected Price", f"${final:,.2f}", trend)
        with c2:
            render_metric_card("Expected Move", f"{arrow} {move:.2f}%", risk, css)
        
        st.write("")
        render_forecast_chart(df.tail(90), result)


# ===== MAIN =====
def main():
    """Main application."""
    init_session_state()
    
    with st.spinner("Loading market data..."):
        df = load_data()
    
    if df is None or df.empty:
        st.warning("No data available.")
        if st.button("🔄 Reload"):
            st.rerun()
        return
    
    # Prepare model
    df_feat = compute_features(df)
    data = prepare_data(df_feat)
    model = train_model(data["X_train"], data["y_train"])
    
    st.header("⚡ Forecast Intelligence")
    
    tab1, tab2, tab3 = st.tabs(["📊 Market Overview", "🧠 Model Intelligence", "🔮 Future Simulator"])
    
    with tab1:
        render_market_tab(df, df_feat)
    with tab2:
        render_model_tab(model, data)
    with tab3:
        render_simulator_tab(df, model, data)


if __name__ == "__main__":
    main()