import pandas as pd
import numpy as np
import yfinance as yf
import warnings
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

STOCKS = {
    "JPM": "JPMorgan Chase",
    "C": "Citigroup",
    "DBK.DE": "Deutsche Bank",
    "NMR": "Nomura",
    "HDFCBANK.NS": "HDFC Bank",
    "AXISBANK.NS": "Axis Bank"
}

results = []

for ticker, name in STOCKS.items():

    print(f"\n{'='*50}")
    print(f"Processing {name} ({ticker})")
    print(f"{'='*50}")

    try:

        df = yf.download(
            ticker,
            start="2015-01-01",
            auto_adjust=True,
            progress=False
        )

        if df.empty:
            print(f"Skipping {ticker} - no data found")
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close = df["Close"].dropna()

        if len(close) < 100:
            print(f"Skipping {ticker} - insufficient data")
            continue

        # -----------------------------
        # Current Price
        # -----------------------------
        current_price = float(close.iloc[-1])

        # -----------------------------
        # Risk Metrics
        # -----------------------------
        returns = close.pct_change().dropna()

        annual_volatility = (
            returns.std() * np.sqrt(252) * 100
        )

        annual_return = (
            returns.mean() * 252 * 100
        )

        sharpe_ratio = (
            annual_return / annual_volatility
            if annual_volatility != 0
            else 0
        )

        # -----------------------------
        # ARIMA Forecast
        # -----------------------------
        model = ARIMA(close, order=(2, 1, 1))
        fitted = model.fit()

        forecast_obj = fitted.get_forecast(steps=30)

        future = forecast_obj.predicted_mean

        conf_int = forecast_obj.conf_int(alpha=0.05)

        lower = conf_int.iloc[:, 0]
        upper = conf_int.iloc[:, 1]

        forecast_price = float(future.iloc[-1])

        expected_return = (
            (forecast_price - current_price)
            / current_price
        ) * 100

        # -----------------------------
        # Save Results
        # -----------------------------
        results.append({
            "Ticker": ticker,
            "Bank": name,

            "Current_Price": round(current_price, 2),
            "Forecast_30D": round(forecast_price, 2),

            "Expected_Return_%": round(expected_return, 2),

            "Forecast_Lower_95": round(float(lower.iloc[-1]), 2),
            "Forecast_Upper_95": round(float(upper.iloc[-1]), 2),

            "Volatility_%": round(float(annual_volatility), 2),
            "Sharpe_Ratio": round(float(sharpe_ratio), 2)
        })

        print(f"Current Price     : {current_price:.2f}")
        print(f"Forecast Price    : {forecast_price:.2f}")
        print(f"Expected Return   : {expected_return:.2f}%")
        print(
            f"95% Forecast Range: "
            f"{float(lower.iloc[-1]):.2f} "
            f"to "
            f"{float(upper.iloc[-1]):.2f}"
        )
        print(f"Volatility        : {annual_volatility:.2f}%")
        print(f"Sharpe Ratio      : {sharpe_ratio:.2f}")

    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        continue

# -----------------------------------
# Create Final DataFrame
# -----------------------------------
forecast_df = pd.DataFrame(results)

forecast_df = forecast_df.sort_values(
    by="Expected_Return_%",
    ascending=False
)

forecast_df.to_csv(
    "outputs/future_forecasts.csv",
    index=False
)

print("\n")
print("=" * 80)
print("GLOBAL BANK FORECAST ANALYTICS")
print("=" * 80)

print(forecast_df.to_string(index=False))

print("\nSaved: outputs/future_forecasts.csv")