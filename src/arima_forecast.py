import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# Load data
df = pd.read_csv("data/JPM.csv")

# Clean
df = df.iloc[1:].copy()
df.rename(columns={"Price": "Date"}, inplace=True)
df["Date"] = pd.to_datetime(df["Date"])

for col in ["Close", "High", "Low", "Open", "Volume"]:
    df[col] = pd.to_numeric(df[col])

df = df.set_index("Date")
series = df["Close"]

# Train-Test Split (same 80/20 as LSTM)
train_size = int(len(series) * 0.8)
train = series[:train_size]
test  = series[train_size:]

print(f"Training size : {len(train)}")
print(f"Test size     : {len(test)}")
print(f"Starting walk-forward validation — this will take ~15-20 mins...")

# Walk-Forward Validation
history     = list(train)
predictions = []

for i in range(len(test)):
    # Fit ARIMA on all data seen so far
    model  = ARIMA(history, order=(2, 1, 1))
    result = model.fit()

    # Forecast 1 step ahead
    yhat = result.forecast(steps=1)[0]
    predictions.append(yhat)

    # Add actual value to history (expanding window)
    history.append(test.iloc[i])

    # Progress update every 50 steps
    if (i + 1) % 50 == 0:
        print(f"  Step {i+1}/{len(test)} done...")

print("Walk-forward validation complete.")

# Metrics
actual = test.values
mae  = mean_absolute_error(actual, predictions)
rmse = mean_squared_error(actual, predictions) ** 0.5

print(f"\nARIMA Walk-Forward MAE : {mae}")
print(f"ARIMA Walk-Forward RMSE: {rmse}")

# Save
pd.DataFrame({
    "Actual"           : actual,
    "ARIMA_Prediction" : predictions
}).to_csv("outputs/arima_predictions.csv", index=False)

print("ARIMA predictions saved.")