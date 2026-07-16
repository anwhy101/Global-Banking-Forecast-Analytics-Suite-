import pandas as pd
from statsmodels.tsa.stattools import adfuller

# Load data
df = pd.read_csv("data/JPM.csv")

# Clean
df = df.iloc[1:].copy()
df.rename(columns={"Price": "Date"}, inplace=True)

df["Date"] = pd.to_datetime(df["Date"])

for col in ["Close", "High", "Low", "Open", "Volume"]:
    df[col] = pd.to_numeric(df[col])

# ADF Test
result = adfuller(df["Close"])

print("ADF Statistic:", result[0])
print("p-value:", result[1])