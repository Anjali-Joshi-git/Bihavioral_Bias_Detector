import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from pathlib import Path

# =============================
# Page config
# =============================
st.set_page_config(page_title="Behavioral Bias Detector", layout="wide")
st.title("📊 Behavioral Bias Detection Dashboard")

# =============================
# Paths
# =============================
BASE_DIR = Path(__file__).parent
BEHAVIOR_PATH = BASE_DIR / "simulated_investor_behavior.csv"
MARKET_PATH = BASE_DIR / "market_df.csv"
MODEL_PATH = BASE_DIR / "Bias_log_model.pkl"

# =============================
# Robust CSV loader
# =============================
@st.cache_data
def load_behavior_data(path):
    df = pd.read_csv(path)

    # 🚨 FIX: CSV read as single column
    if len(df.columns) == 1 and "," in df.columns[0]:
        df = pd.read_csv(path, sep=",")

    df.columns = df.columns.str.strip()
    return df

@st.cache_data
def load_market_data(path):
    df = pd.read_csv(path)

    if len(df.columns) == 1 and "," in df.columns[0]:
        df = pd.read_csv(path, sep=",")

    df.columns = df.columns.str.strip()
    return df

@st.cache_resource
def load_model(path):
    return joblib.load(path)

behavior_df = load_behavior_data(BEHAVIOR_PATH)
market_df = load_market_data(MARKET_PATH)
model = load_model(MODEL_PATH)

# =============================
# Detect bias column
# =============================
POSSIBLE_TARGETS = [
    "true_bias",
    "bias",
    "bias_label",
    "investor_bias",
    "target"
]

TARGET_COL = next((c for c in POSSIBLE_TARGETS if c in behavior_df.columns), None)

if TARGET_COL is None:
    st.error("❌ No bias label column found.")
    st.write("Available columns:", behavior_df.columns.tolist())
    st.stop()

# =============================
# Sidebar
# =============================
st.sidebar.header("🎛 Controls")

selected_bias = st.sidebar.multiselect(
    "Select Bias Type",
    options=behavior_df[TARGET_COL].unique(),
    default=behavior_df[TARGET_COL].unique()
)

filtered_df = behavior_df[behavior_df[TARGET_COL].isin(selected_bias)]

# =============================
# Market Overview
# =============================
st.subheader("📈 Market Overview")

fig_price = px.line(
    market_df,
    x="date" if "date" in market_df.columns else market_df.columns[0],
    y="close" if "close" in market_df.columns else market_df.columns[1],
    title="Market Price Movement"
)
st.plotly_chart(fig_price, use_container_width=True)

# =============================
# Bias Distribution
# =============================
st.subheader("🧠 Behavioral Bias Distribution")

fig_bias = px.histogram(
    filtered_df,
    x=TARGET_COL,
    color=TARGET_COL
)
st.plotly_chart(fig_bias, use_container_width=True)

# =============================
# Holding Period
# =============================
st.subheader("⏳ Holding Duration by Bias")

fig_hold = px.box(
    filtered_df,
    x=TARGET_COL,
    y="holding_days"
)
st.plotly_chart(fig_hold, use_container_width=True)

# =============================
# PnL Analysis
# =============================
st.subheader("💰 Profit & Loss by Bias")

fig_pnl = px.violin(
    filtered_df,
    x=TARGET_COL,
    y="pnl",
    box=True
)
st.plotly_chart(fig_pnl, use_container_width=True)

# =============================
# Prediction
# =============================
st.subheader("🔮 Bias Prediction")

holding_days = st.slider("Holding Days", 0, 60, 10)
trade_frequency = st.slider("Trade Frequency", 1, 30, 5)
pnl = st.slider("PnL", -0.5, 0.5, 0.05)
volatility = st.slider("Volatility", 0.0, 1.0, 0.25)
drawdown = st.slider("Max Drawdown", 0.0, 1.0, 0.2)
win_rate = st.slider("Win Rate", 0.0, 1.0, 0.55)
position_size = st.slider("Position Size", 1, 100, 10)

input_df = pd.DataFrame([{
    "holding_days": holding_days,
    "trade_frequency": trade_frequency,
    "pnl": pnl,
    "volatility": volatility,
    "drawdown": drawdown,
    "win_rate": win_rate,
    "position_size": position_size
}])

if st.button("Predict Bias"):
    pred = model.predict(input_df)[0]
    st.success(f"🧠 Predicted Investor Bias: **{pred.upper()}**")
