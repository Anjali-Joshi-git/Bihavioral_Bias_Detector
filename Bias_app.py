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

BASE_DIR = Path(__file__).parent

# =============================
# Ultra-robust CSV loader
# =============================
@st.cache_data
def load_csv_safe(path):
    df = pd.read_csv(path)

    # 🚨 Case: entire CSV loaded as ONE column
    if len(df.columns) == 1 and "," in df.columns[0]:
        col = df.columns[0]

        # Split header
        headers = col.split(",")

        # Split rows
        data = df[col].astype(str).str.split(",", expand=True)

        data.columns = headers
        df = data

    df.columns = df.columns.str.strip()
    return df


@st.cache_resource
def load_model(path):
    return joblib.load(path)

# =============================
# Load data
# =============================
behavior_df = load_csv_safe(BASE_DIR / "simulated_investor_behavior.csv")
market_df = load_csv_safe(BASE_DIR / "market_df.csv")
model = load_model(BASE_DIR / "Bias_log_model.pkl")

# =============================
# Bias column detection
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
    st.write("Available columns:")
    st.code(list(behavior_df.columns))
    st.stop()

# =============================
# Sidebar
# =============================
st.sidebar.header("🎛 Filters")

selected_bias = st.sidebar.multiselect(
    "Select Bias",
    behavior_df[TARGET_COL].unique(),
    default=behavior_df[TARGET_COL].unique()
)

filtered_df = behavior_df[behavior_df[TARGET_COL].isin(selected_bias)]

# =============================
# Bias Distribution
# =============================
st.subheader("🧠 Bias Distribution")

fig_bias = px.histogram(
    filtered_df,
    x=TARGET_COL,
    color=TARGET_COL
)
st.plotly_chart(fig_bias, use_container_width=True)

# =============================
# Holding Period
# =============================
st.subheader("⏳ Holding Days by Bias")

fig_hold = px.box(
    filtered_df,
    x=TARGET_COL,
    y="holding_days"
)
st.plotly_chart(fig_hold, use_container_width=True)

# =============================
# PnL Analysis
# =============================
st.subheader("💰 PnL by Bias")

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
st.subheader("🔮 Predict Investor Bias")

holding_days = st.slider("Holding Days", 0, 60, 10)
trade_frequency = st.slider("Trade Frequency", 1, 30, 5)
pnl = st.slider("PnL", -1.0, 1.0, 0.1)
volatility = st.slider("Volatility", 0.0, 1.0, 0.3)
drawdown = st.slider("Drawdown", 0.0, 1.0, 0.2)
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
    prediction = model.predict(input_df)[0]
    st.success(f"🧠 Predicted Bias: **{prediction.upper()}**")
