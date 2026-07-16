import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import yfinance as yf
import os

# ── Stocks to analyze ──────────────────────────────────────────────
STOCKS = {
    "JPM"         : "JPMorgan Chase (US)",
    "C"           : "Citigroup (US)",
    "DBK.DE"      : "Deutsche Bank (Germany)",
    "NMR"         : "Nomura (Japan)",
    "HDFCBANK.NS" : "HDFC Bank (India)",
    "AXISBANK.NS" : "Axis Bank (India)",
}

START = "2015-01-01"
END   = "2024-12-31"

# ── Output folders ─────────────────────────────────────────────────
os.makedirs("outputs/plots",       exist_ok=True)
os.makedirs("outputs/predictions", exist_ok=True)

all_metrics = []

# ══════════════════════════════════════════════════════════════════
def run_arima(series, train_size):
    series  = pd.Series(series.values.astype(float))
    train   = series[:train_size]
    test    = series[train_size:]
    history = list(train)
    predictions = []
    print(f"    ARIMA walk-forward ({len(test)} steps)...")
    for i in range(len(test)):
        model  = ARIMA(history, order=(2, 1, 1))
        result = model.fit()
        yhat   = result.forecast(steps=1)[0]
        predictions.append(yhat)
        history.append(float(test.iloc[i]))
        if (i + 1) % 100 == 0:
            print(f"      Step {i+1}/{len(test)}...")
    return test, np.array(predictions)


def run_lstm(series, train_size, window=60, epochs=10):
    series = pd.Series(series.values.astype(float))
    data   = series.values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i-window:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)

    split   = train_size - window
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        LSTM(50, input_shape=(window, 1)),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    print(f"    LSTM training ({epochs} epochs)...")
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)

    predictions = scaler.inverse_transform(model.predict(X_test, verbose=0))
    actual      = scaler.inverse_transform(y_test)
    return actual.flatten(), predictions.flatten()


def get_metrics(actual, pred, model_name, ticker):
    mae  = mean_absolute_error(actual, pred)
    rmse = mean_squared_error(actual, pred) ** 0.5
    return {
        "Ticker" : ticker,
        "Model"  : model_name,
        "MAE"    : round(mae, 4),
        "RMSE"   : round(rmse, 4),
    }


# ══════════════════════════════════════════════════════════════════
for ticker, name in STOCKS.items():
    print(f"\n{'='*55}")
    print(f"  Processing: {name} ({ticker})")
    print(f"{'='*55}")

    # Download
    raw = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)
    if raw.empty or len(raw) < 200:
        print(f"  Skipping {ticker} — insufficient data.")
        continue

    # Flatten MultiIndex columns if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Clean series
    series = raw["Close"].dropna().squeeze()
    series = pd.Series(series.values.astype(float))

    train_size = int(len(series) * 0.8)

    # ── ARIMA ──
    test_arima, pred_arima = run_arima(series, train_size)
    m_arima = get_metrics(test_arima.values, pred_arima, "ARIMA", ticker)
    print(f"    ARIMA  MAE: {m_arima['MAE']}  RMSE: {m_arima['RMSE']}")

    # ── LSTM ──
    actual_lstm, pred_lstm = run_lstm(series, train_size)
    m_lstm = get_metrics(actual_lstm, pred_lstm, "LSTM", ticker)
    print(f"    LSTM   MAE: {m_lstm['MAE']}  RMSE: {m_lstm['RMSE']}")

    # ── Align lengths ──
    n            = min(len(pred_arima), len(pred_lstm))
    pred_arima_  = pred_arima[-n:]
    pred_lstm_   = pred_lstm[-n:]
    actual_      = actual_lstm[-n:]

    # ── Hybrid (50/50) ──
    hybrid    = (0.5 * pred_arima_) + (0.5 * pred_lstm_)
    m_hybrid  = get_metrics(actual_, hybrid, "Hybrid", ticker)
    print(f"    Hybrid MAE: {m_hybrid['MAE']}  RMSE: {m_hybrid['RMSE']}")

    all_metrics.extend([m_arima, m_lstm, m_hybrid])

    # ── Save predictions ──
    pd.DataFrame({
        "Actual"            : actual_,
        "ARIMA_Prediction"  : pred_arima_,
        "LSTM_Prediction"   : pred_lstm_,
        "Hybrid_Prediction" : hybrid,
    }).to_csv(f"outputs/predictions/{ticker.replace('.', '_')}_predictions.csv", index=False)

    # ── Plot ──
    plt.figure(figsize=(14, 5))
    plt.plot(actual_,     label="Actual",            color="black", linewidth=1.5)
    plt.plot(pred_arima_, label="ARIMA",             color="blue",  linewidth=1, linestyle="--")
    plt.plot(pred_lstm_,  label="LSTM",              color="green", linewidth=1)
    plt.plot(hybrid,      label="Hybrid ARIMA-LSTM", color="red",   linewidth=1.5)
    plt.title(f"{name} ({ticker}) — Model Comparison")
    plt.xlabel("Test Period (Days)")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"outputs/plots/{ticker.replace('.', '_')}_comparison.png", dpi=150)
    plt.close()
    print(f"    Plot saved.")

# ── Combined metrics table ─────────────────────────────────────────
metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv("outputs/all_stocks_metrics.csv", index=False)

print(f"\n{'='*55}")
print("ALL STOCKS COMPLETE")
print(metrics_df.to_string(index=False))
print(f"{'='*55}")
print("Saved: outputs/all_stocks_metrics.csv")