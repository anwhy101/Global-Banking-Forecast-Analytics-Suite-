
import streamlit as st
import pandas as pd
from PIL import Image
import os

# --------------------------------------------------
# Page Config
# --------------------------------------------------

st.set_page_config(
    page_title="Global Banking Forecast Analytics Suite",
    layout="wide"
)

# --------------------------------------------------
# Load Data
# --------------------------------------------------

metrics = pd.read_csv(
    "outputs/all_stocks_metrics.csv"
)

forecast_df = pd.read_csv(
    "outputs/future_forecasts.csv"
)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

stock_map = {
    "JPMorgan Chase (US)": "JPM",
    "Citigroup (US)": "C",
    "Deutsche Bank (Germany)": "DBK.DE",
    "Nomura (Japan)": "NMR",
    "HDFC Bank (India)": "HDFCBANK.NS",
    "Axis Bank (India)": "AXISBANK.NS"
}

selected_bank = st.sidebar.selectbox(
    "Select Bank",
    list(stock_map.keys())
)

st.sidebar.markdown("---")

st.sidebar.markdown("""
## 📌 Project Scope

### Countries
- 🇺🇸 USA
- 🇩🇪 Germany
- 🇯🇵 Japan
- 🇮🇳 India

### Models
- ARIMA
- LSTM
- Hybrid ARIMA-LSTM

### Validation
- Walk-Forward Validation

### Metrics
- RMSE
- Volatility
- Sharpe Ratio
- Forecast Interval
""")

ticker = stock_map[selected_bank]

# --------------------------------------------------
# Bank Data
# --------------------------------------------------

stock_metrics = metrics[
    metrics["Ticker"] == ticker
]

forecast_row = forecast_df[
    forecast_df["Ticker"] == ticker
].iloc[0]

best_model = stock_metrics.loc[
    stock_metrics["RMSE"].idxmin()
]

# --------------------------------------------------
# Title
# --------------------------------------------------

st.title(
    "🌍 Global Banking Forecast Analytics Suite"
)

st.markdown("""
Forecasting and benchmarking international banking stocks using:

- ARIMA
- LSTM
- Hybrid ARIMA-LSTM
- Walk-Forward Validation
- Risk Analytics
- Forecast Uncertainty Analysis
""")

# --------------------------------------------------
# Metric Guide
# --------------------------------------------------

with st.expander("📚 Understanding These Metrics"):

    st.markdown("""
### Current Price
The latest market price of the stock.

### 30-Day Forecast
Expected stock price after 30 trading days based on the forecasting model.

### Volatility (%)
Measures how much a stock's price fluctuates.

- Lower = More stable
- Higher = More risky

### Sharpe Ratio
Measures return relative to risk.

- Higher = Better risk-adjusted performance

### RMSE
Forecast error metric.

- Lower RMSE = More accurate model

### 95% Forecast Range
Range where future prices are expected to lie with high confidence.

- Narrow range = Higher certainty
- Wide range = Higher uncertainty
""")

# --------------------------------------------------
# Executive Summary
# --------------------------------------------------

st.header(selected_bank)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Current Price",
        f"{forecast_row['Current_Price']}"
    )
    st.caption("Latest market price.")

expected_return = forecast_row["Expected_Return_%"]

if expected_return > 2:
    signal = "🟢 Bullish"

elif expected_return > -2:
    signal = "🟡 Neutral"

else:
    signal = "🔴 Bearish"

with col2:
    st.metric(
        "Investment Signal",
        signal
    )
    st.caption(
        f"Expected Return: {expected_return:.2f}%"
    )

with col3:
    st.metric(
        "Volatility %",
        f"{forecast_row['Volatility_%']}"
    )
    st.caption("Measures stock risk.")

with col4:
    st.metric(
        "Sharpe Ratio",
        f"{forecast_row['Sharpe_Ratio']}"
    )
    st.caption("Risk-adjusted return.")

with col5:
    st.metric(
        "Best Model",
        best_model["Model"]
    )
    st.caption("Lowest forecasting error.")

# --------------------------------------------------
# Risk Classification
# --------------------------------------------------

vol = forecast_row["Volatility_%"]

if vol < 25:
    risk = "🟢 Low Risk"
elif vol < 35:
    risk = "🟡 Moderate Risk"
else:
    risk = "🔴 High Risk"

st.write(f"### Risk Classification: {risk}")

# --------------------------------------------------
# Forecast Outlook
# --------------------------------------------------

st.subheader("📈 Forecast Outlook")

st.info(
    f"""
95% Forecast Range

Lower Bound: {forecast_row['Forecast_Lower_95']}

Upper Bound: {forecast_row['Forecast_Upper_95']}

Interpretation:
Future prices are expected to lie within this range with high confidence.
"""
)

# --------------------------------------------------
# Sharpe Interpretation
# --------------------------------------------------

sharpe = forecast_row["Sharpe_Ratio"]

if sharpe > 1:
    st.success(
        "Excellent risk-adjusted performance."
    )
elif sharpe > 0.5:
    st.info(
        "Reasonable risk-adjusted performance."
    )
else:
    st.warning(
        "Relatively weak risk-adjusted performance."
    )

# --------------------------------------------------
# Model Performance
# --------------------------------------------------

st.subheader("📊 Model Performance")

st.dataframe(
    stock_metrics,
    use_container_width=True
)

# --------------------------------------------------
# Forecast Plot
# --------------------------------------------------

plot_file = (
    ticker.replace(".", "_")
    + "_comparison.png"
)

plot_path = os.path.join(
    "outputs",
    "plots",
    plot_file
)

st.subheader("📉 Forecast Comparison")

if os.path.exists(plot_path):

    image = Image.open(plot_path)

    st.image(
        image,
        use_container_width=True
    )

else:

    st.warning(
        "Plot not found."
    )

# --------------------------------------------------
# Key Insights
# --------------------------------------------------

st.subheader("🧠 Key Insights")

arima_rmse = stock_metrics[
    stock_metrics["Model"] == "ARIMA"
]["RMSE"].values[0]

lstm_rmse = stock_metrics[
    stock_metrics["Model"] == "LSTM"
]["RMSE"].values[0]

hybrid_rmse = stock_metrics[
    stock_metrics["Model"] == "Hybrid"
]["RMSE"].values[0]

st.write(f"ARIMA RMSE : {arima_rmse:.4f}")
st.write(f"LSTM RMSE : {lstm_rmse:.4f}")
st.write(f"Hybrid RMSE : {hybrid_rmse:.4f}")

if arima_rmse < lstm_rmse:

    st.success(
        "ARIMA outperformed LSTM under walk-forward validation."
    )

else:

    st.success(
        "LSTM outperformed ARIMA under walk-forward validation."
    )

# --------------------------------------------------
# Global Ranking
# --------------------------------------------------

st.header("🌎 Global Banking Ranking")

ranking = forecast_df.copy()

ranking["Risk_Level"] = ranking["Volatility_%"].apply(
    lambda x:
    "Low"
    if x < 25
    else "Moderate"
    if x < 35
    else "High"
)

ranking = ranking.sort_values(
    by="Sharpe_Ratio",
    ascending=False
)

ranking.insert(
    0,
    "Rank",
    range(1, len(ranking)+1)
)

st.dataframe(
    ranking[
        [
            "Rank",
            "Bank",
            "Sharpe_Ratio",
            "Volatility_%",
            "Risk_Level"
        ]
    ],
    use_container_width=True
)

# --------------------------------------------------
# Top Performer
# --------------------------------------------------

top_bank = ranking.iloc[0]

st.success(
    f"""
🏆 Top Ranked Bank: {top_bank['Bank']}

Sharpe Ratio: {top_bank['Sharpe_Ratio']}

Volatility: {top_bank['Volatility_%']}%
"""
)

# --------------------------------------------------
# Global Benchmark
# --------------------------------------------------

st.header("🏆 Best Model By Bank")

best_models = (
    metrics
    .sort_values("RMSE")
    .groupby("Ticker")
    .first()
    .reset_index()
)
st.header("🎯 Project Impact")

st.info(
    """
This platform compares global banking stocks
using forecasting models, volatility analysis,
Sharpe ratio benchmarking, and risk classification.

The goal is to help users understand risk-adjusted
investment opportunities across international banks.
"""
)
with open(
    "outputs/future_forecasts.csv",
    "rb"
) as file:

    st.download_button(
        label="📥 Download Forecast Report",
        data=file,
        file_name="future_forecasts.csv",
        mime="text/csv"
    )

st.dataframe(
    best_models[
        [
            "Ticker",
            "Model",
            "RMSE"
        ]
    ],
    use_container_width=True
)

