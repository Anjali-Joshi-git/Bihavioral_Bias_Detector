import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from pathlib import Path

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Behavioral Bias Detector",
    layout="wide"
)

st.title("📊 Behavioral Bias Detection Dashboard")
st.caption("Detecting investor irrationalities using behavioral finance & ML")

# -----------------------------
# File paths (cloud + local safe)
# -----------------------------
BASE_DIR = Path(__file__).parent
BEHAVIOR_PATH = BASE_DIR / "simulated_investor_behavior.csv"
MARKET_PATH = BASE_DIR / "market_df.csv"
MODEL_PATH = BASE_DIR / "Bias_log_model.pkl"

# -----------------------------
# Load data & model
# -----------------------------
@st.cache_data
def load_data():
    behavior_df = pd.read_csv(BEHAVIOR_PATH)
    market_df = pd.read_csv(MARKET_PATH)

    # Clean column names (important)
    behavior_df.columns = behavior_df.columns.str.strip()
    market_df.columns = market_df.columns.str.strip()

    return behavior_df, market_df

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

behavior_df, market_df = load_data()
model = load_model()

# -----------------------------
# Detect target / bias column safely
# -----------------------------
POSSIBLE_TARGETS = ["true_bias", "bias", "bias_label", "investor_bias", "target"]

TARGET_COL = next(
    (col for col in POSSIBLE_TARGETS if col in behavior_df.columns),
    None
)

if TARGET_COL is None:
    st.error(
        "❌ No bias label column found.\n\n"
        "Expected one of: " + ", ".join(POSSIBLE_TARGETS)
    )
    st.write("Available columns:", behavior_df.columns.tolist())
    st.stop()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("🎛 Controls")

selected_bias = st.sidebar.multiselect(
    "Select Bias Type",
    options=behavior_df[TARGET_COL].unique(),
    default=behavior_df[TARGET_COL].unique()
)

filtered_df = behavior_df[behavior_df[TARGET_COL].isin(selected_bias)]

# -----------------------------
# Section 1: Market Overview
# -----------------------------
st.subheader("📈 Market Overview")

fig_price = px.line(
    market_df,
    x="Date",
    y="Close",
    title="Market Price Movement"
)
st.plotly_chart(fig_price, use_container_width=True)

# -----------------------------
# Section 2: Behavioral Distribution
# -----------------------------
st.subheader("🧠 Behavioral Bias Distribution")

fig_bias = px.histogram(
    filtered_df,
    x=TARGET_COL,
    color=TARGET_COL,
    title="Investor Bias Distribution"
)
st.plotly_chart(fig_bias, use_container_width=True)

# -----------------------------
# Section 3: Holding Behavior
# -----------------------------
st.subheader("⏳ Holding Duration by Bias")

fig_hold = px.box(
    filtered_df,
    x=TARGET_COL,
    y="holding_days",
    color=TARGET_COL,
    title="Holding Duration Patterns"
)
st.plotly_chart(fig_hold, use_container_width=True)

# -----------------------------
# Section 4: PnL Analysis
# -----------------------------
st.subheader("💰 Profit & Loss Analysis")

fig_pnl = px.violin(
    filtered_df,
    x=TARGET_COL,
    y="pnl",
    color=TARGET_COL,
    box=True,
    title="PnL Distribution by Bias"
)
st.plotly_chart(fig_pnl, use_container_width=True)

# -----------------------------
# Section 5: Live Prediction Demo
# -----------------------------
st.subheader("🔮 Bias Prediction Demo")

col1, col2, col3 = st.columns(3)

with col1:
    holding_days = st.slider("Holding Days", 0, 60, 10)
    volatility = st.slider("Volatility", 0.0, 1.0, 0.25)

with col2:
    trade_frequency = st.slider("Trade Frequency", 1, 30, 5)
    drawdown = st.slider("Max Drawdown", 0.0, 1.0, 0.2)

with col3:
    pnl = st.slider("PnL", -0.5, 0.5, 0.05)
    win_rate = st.slider("Win Rate", 0.0, 1.0, 0.55)
    position_size = st.slider("Position Size", 1, 100, 10)

# -----------------------------
# Build input (MUST match training)
# -----------------------------
FEATURES = [
    "holding_days",
    "trade_frequency",
    "pnl",
    "volatility",
    "drawdown",
    "win_rate",
    "position_size"
]

input_df = pd.DataFrame([{
    "holding_days": holding_days,
    "trade_frequency": trade_frequency,
    "pnl": pnl,
    "volatility": volatility,
    "drawdown": drawdown,
    "win_rate": win_rate,
    "position_size": position_size
}])[FEATURES]

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Bias"):
    prediction = model.predict(input_df)[0]
    st.success(f"🧠 Predicted Investor Bias: **{prediction.upper()}**")
