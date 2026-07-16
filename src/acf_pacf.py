import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import (
    plot_acf,
    plot_pacf
)

# Load data
df = pd.read_csv("data/JPM.csv")

# Clean
df = df.iloc[1:].copy()
df.rename(columns={"Price": "Date"}, inplace=True)

df["Date"] = pd.to_datetime(df["Date"])

for col in ["Close", "High", "Low", "Open", "Volume"]:
    df[col] = pd.to_numeric(df[col])

# First differencing
df["Close_Diff"] = df["Close"].diff()

df.dropna(inplace=True)

# ACF
plt.figure(figsize=(10,5))
plot_acf(df["Close_Diff"], lags=40)
plt.show()

# PACF
plt.figure(figsize=(10,5))
plot_pacf(df["Close_Diff"], lags=40)
plt.show()