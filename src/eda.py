import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/JPM.csv")

# Clean
df = df.iloc[1:].copy()
df.rename(columns={"Price": "Date"}, inplace=True)

df["Date"] = pd.to_datetime(df["Date"])

for col in ["Close", "High", "Low", "Open", "Volume"]:
    df[col] = pd.to_numeric(df[col])

# Plot Closing Price
plt.figure(figsize=(12,6))

plt.plot(df["Date"], df["Close"])

plt.title("JPM Stock Closing Price (2015-2025)")
plt.xlabel("Date")
plt.ylabel("Price")

plt.grid(True)

plt.show()