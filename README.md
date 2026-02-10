<a href="https://bihavioralbiasdetector-greefspdfqlxal8b7mrrds.streamlit.app/" target="_blank" style="color: #FF4B4B; text-decoration: none; font-weight: bold;">
    🚀 Try the Live Dashboard Now
</a>

# Behavioral Bias Detection in Financial Markets

## Overview
This project implements a data-driven framework to detect behavioral biases in financial markets using simulated investor actions derived from historical price data. The approach integrates behavioral finance principles with interpretable machine learning to identify structured irrational patterns under noisy and uncertain conditions.

The emphasis of this project is on interpretability, robustness, and realistic behavioral modeling rather than maximizing predictive accuracy.

---

## Problem Definition
Traditional financial models assume rational decision-making, which fails to capture systematic behavioral distortions observed in real markets. This project investigates whether behavioral biases can be detected using engineered features derived solely from observable market data.

---

## Objectives
- Simulate investor actions influenced by behavioral finance biases
- Engineer behavior-aware features from historical price data
- Train interpretable classification models for bias detection
- Evaluate model robustness under noisy conditions
- Avoid information leakage and unrealistic assumptions

---

## Dataset
**Source:** Yahoo Finance  
**Type:** Historical equity time-series data

### Raw Features
- Open, High, Low, Close prices
- Trading Volume

### Preprocessing
- Conservative handling of missing values
- Computation of returns and rolling statistics
- Strict use of historical windows only
- Explicit prevention of data leakage

---

## Behavioral Simulation
Investor actions (**BUY / SELL / HOLD**) are generated using probabilistic rules aligned with established behavioral finance theory.

### Modeled Biases
- **Herd Behavior:** Trend-following with longer holding periods
- **Loss Aversion:** Rapid exit following negative returns
- **Overconfidence:** Increased trading frequency and reduced holding duration

Controlled noise is injected to simulate human inconsistency and prevent deterministic patterns.

---

## Feature Engineering
Engineered features include:
- Holding duration metrics
- Trade frequency measures
- Rolling returns and volatility
- Profit and loss sensitivity indicators
- Action imbalance ratios

All features are computed using past information only.

---

## Modeling Approach

### Logistic Regression (Primary Model)
- Selected for interpretability
- Enables coefficient-level behavioral analysis
- Demonstrates stable performance under noise

### Random Forest (Benchmark Model)
- Used for performance comparison
- Achieved near-perfect accuracy
- Excluded from final selection due to overfitting risk and limited interpretability

---

## Evaluation
Models were evaluated using:
- Confusion Matrix
- Precision, Recall, and Macro F1-score
- Multi-class ROC-AUC
- Noise stress testing for robustness validation

---

## Results
- Logistic Regression maintained consistent performance under increasing noise
- Random Forest performance degraded under noise injection
- Interpretable models provided meaningful behavioral insights despite lower raw accuracy

---

## Project Structure
├── data/ # Market data

├── eda/ # Exploratory analysis

├── simulation/ # Behavioral simulation logic

├── features/ # Feature engineering

├── models/ # Training and evaluation

├── visualizations/ # Interpretability and performance plots

└── README.md

---

## Limitations
- Investor behavior is simulated rather than sourced from real transaction data
- Bias definitions are theory-driven
- Results are not intended for live trading deployment

---

## Future Work
- Integration of real investor transaction data
- Reinforcement learning–based behavioral agents
- Bias-aware portfolio optimization
- Market regime–specific behavioral modeling

---

## Author
Anjali Joshi
