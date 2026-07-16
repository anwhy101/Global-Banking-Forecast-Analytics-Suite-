print("Program Started")
import yfinance as yf
import pandas as pd
import os

# Bank stocks
stocks = {
    "JPM": "JPM",
    "DB": "DB",
    "CITI": "C",
    "ICICI": "ICICIBANK.NS",
    "AXIS": "AXISBANK.NS"
}

os.makedirs("../data", exist_ok=True)

for name, ticker in stocks.items():

    print(f"Downloading {name}...")

    data = yf.download(
        ticker,
        start="2015-01-01",
        end="2026-01-01"
    )

    data.to_csv(f"../data/{name}.csv")

    print(f"{name} saved successfully!")

print("\nAll data downloaded!")