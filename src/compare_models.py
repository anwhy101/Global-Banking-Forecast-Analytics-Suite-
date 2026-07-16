import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load all predictions
arima_df = pd.read_csv("outputs/arima_predictions.csv")
lstm_df  = pd.read_csv("outputs/lstm_predictions.csv")
hybrid_df = pd.read_csv("outputs/hybrid_predictions.csv")

# Align all to same length (LSTM is shortest due to 60-day window)
n = len(lstm_df)
arima_df = arima_df.tail(n).reset_index(drop=True)

actual       = lstm_df["Actual"]
arima_pred   = arima_df["ARIMA_Prediction"]
lstm_pred    = lstm_df["LSTM_Prediction"]
hybrid_pred  = hybrid_df["Hybrid_Prediction"]

# Metrics
def get_metrics(actual, pred, name):
    mae  = mean_absolute_error(actual, pred)
    rmse = mean_squared_error(actual, pred) ** 0.5
    return {"Model": name, "MAE": round(mae, 4), "RMSE": round(rmse, 4)}

metrics = pd.DataFrame([
    get_metrics(actual, arima_pred,  "ARIMA"),
    get_metrics(actual, lstm_pred,   "LSTM"),
    get_metrics(actual, hybrid_pred, "Hybrid ARIMA-LSTM"),
])

print("\n===== Model Comparison =====")
print(metrics.to_string(index=False))

metrics.to_csv("outputs/model_comparison.csv", index=False)
print("\nMetrics saved to outputs/model_comparison.csv")

# Plot
plt.figure(figsize=(14, 6))
plt.plot(actual.values,      label="Actual",             color="black", linewidth=1.5)
plt.plot(arima_pred.values,  label="ARIMA",              color="blue",  linewidth=1, linestyle="--")
plt.plot(lstm_pred.values,   label="LSTM",               color="green", linewidth=1)
plt.plot(hybrid_pred.values, label="Hybrid ARIMA-LSTM",  color="red",   linewidth=1.5)

plt.title("JPM Stock Price Forecast — Model Comparison", fontsize=14)
plt.xlabel("Test Period (Days)")
plt.ylabel("Price (USD)")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/model_comparison.png", dpi=150)
plt.show()
print("Plot saved to outputs/model_comparison.png")