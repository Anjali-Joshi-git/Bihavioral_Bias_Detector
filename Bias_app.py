import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Behavioral Bias Detector",
    layout="wide"
)

st.title("📊 Behavioral Bias Detection Dashboard")
st.caption("Detecting investor irrationalities using Behavioral Finance & ML")

# -----------------------------
# Load data & model
# -----------------------------
@st.cache_data
def load_data():
    behavior_df = pd.read_csv("simulated_investor_behavior.csv")
    market_df = pd.read_csv("market_df.csv")
    return behavior_df, market_df

@st.cache_resource
def load_model():
    return joblib.load("Bias_log_model.pkl")

behavior_df, market_df = load_data()
model = load_model()

# -----------------------------
# IMPORTANT: Feature schema
# Must match training exactly
# -----------------------------
FEATURE_COLUMNS = [
    "holding_days",
    "trade_frequency",
    "pnl",
    "avg_return",
    "volatility",
    "action_imbalance",
    "trend_following"
]

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Controls")

selected_bias = st.sidebar.multiselect(
    "Select Investor Bias",
    options=behavior_df["true_bias"].unique(),
    default=behavior_df["true_bias"].unique()
)

filtered_df = behavior_df[behavior_df["true_bias"].isin(selected_bias)]

# -----------------------------
# Market Overview
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
# Behavioral Distribution
# -----------------------------
st.subheader("🧠 Behavioral Bias Distribution")

fig_bias = px.histogram(
    filtered_df,
    x="true_bias",
    color="true_bias",
    title="Distribution of Investor Biases"
)
st.plotly_chart(fig_bias, use_container_width=True)

# -----------------------------
# Holding Duration Analysis
# -----------------------------
st.subheader("⏳ Holding Duration by Bias")

fig_hold = px.box(
    filtered_df,
    x="true_bias",
    y="holding_days",
    color="true_bias",
    title="Holding Duration Patterns"
)
st.plotly_chart(fig_hold, use_container_width=True)

# -----------------------------
# PnL Analysis
# -----------------------------
st.subheader("💰 Profit & Loss by Bias")

fig_pnl = px.violin(
    filtered_df,
    x="true_bias",
    y="pnl",
    color="true_bias",
    box=True,
    title="PnL Distribution Across Biases"
)
st.plotly_chart(fig_pnl, use_container_width=True)

# -----------------------------
# Live Bias Prediction
# -----------------------------
st.subheader("🔮 Live Bias Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    holding_days = st.slider("Holding Days", 0, 30, 5)

with col2:
    trade_frequency = st.slider("Trade Frequency", 1, 20, 5)

with col3:
    pnl = st.slider("PnL", -0.2, 0.2, 0.01)

# -----------------------------
# Build inference input
# -----------------------------
input_df = pd.DataFrame([{
    "holding_days": holding_days,
    "trade_frequency": trade_frequency,
    "pnl": pnl,
    "avg_return": 0.0,        # neutral defaults
    "volatility": 0.0,
    "action_imbalance": 0.0,
    "trend_following": 0.0
}])

# Enforce correct feature order
input_df = input_df[FEATURE_COLUMNS]

if st.button("Predict Investor Bias"):
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    st.success(f"🧠 Predicted Investor Bias: **{prediction.upper()}**")

    prob_df = pd.DataFrame({
        "Bias": model.classes_,
        "Probability": probabilities
    })

    fig_prob = px.bar(
        prob_df,
        x="Bias",
        y="Probability",
        title="Prediction Confidence"
    )

    st.plotly_chart(fig_prob, use_container_width=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Built by Anjali Joshi | Behavioral Bias Detection Project")
