import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load predictions
arima_df  = pd.read_csv("outputs/arima_predictions.csv")
lstm_df   = pd.read_csv("outputs/lstm_predictions.csv")

# Align lengths
n = len(lstm_df)
arima_df = arima_df.tail(n).reset_index(drop=True)

actual     = lstm_df["Actual"]
arima_pred = arima_df["ARIMA_Prediction"]
lstm_pred  = lstm_df["LSTM_Prediction"]

# Weighted hybrid (LSTM gets 80% weight, ARIMA gets 20%)
w_arima = 0.5
w_lstm  = 0.5
hybrid_pred = (w_arima * arima_pred) + (w_lstm * lstm_pred)

# Metrics
mae  = mean_absolute_error(actual, hybrid_pred)
rmse = mean_squared_error(actual, hybrid_pred) ** 0.5

print("Hybrid MAE :", mae)
print("Hybrid RMSE:", rmse)

# Save
pd.DataFrame({
    "Actual"            : actual,
    "Hybrid_Prediction" : hybrid_pred
}).to_csv("outputs/hybrid_predictions.csv", index=False)

print("Hybrid predictions saved.")