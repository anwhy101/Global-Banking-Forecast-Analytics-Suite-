import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Load data
df = pd.read_csv("data/JPM.csv")

# Clean
df = df.iloc[1:].copy()
df.rename(columns={"Price": "Date"}, inplace=True)

df["Date"] = pd.to_datetime(df["Date"])

for col in ["Close", "High", "Low", "Open", "Volume"]:
    df[col] = pd.to_numeric(df[col])

series = df["Close"]

orders = [
    (1,1,0),
    (0,1,1),
    (1,1,1),
    (2,1,1)
]

for order in orders:
    model = ARIMA(series, order=order)
    result = model.fit()

    print(
        f"ARIMA{order} | AIC = {result.aic:.2f}"
    )