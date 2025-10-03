# ChatGPT Code For VWAP Reclaim

import yfinance as yf
import pandas as pd

# Download intraday data
data = yf.download("AAPL", period="5d", interval="5m")

# Drop adj close if present
if "Adj Close" in data.columns:
    data = data.drop(columns=["Adj Close"])

# VWAP calculation
data["VWAP"] = (data["Close"] * data["Volume"]).cumsum() / data["Volume"].cumsum()

# Generate signals
data["Above_VWAP"] = data["Close"] > data["VWAP"]

# Trade signals (1 = long, 0 = flat)
data["Signal"] = data["Above_VWAP"].astype(int)

# Only count entry on the *reclaim* (when it switches from 0 → 1)
data["Entry"] = (data["Signal"].shift(1) == 0) & (data["Signal"] == 1)

# Position = 1 when above VWAP, 0 otherwise
data["Position"] = data["Signal"]

# Returns
data["Market_Returns"] = data["Close"].pct_change()
data["Strategy_Returns"] = data["Position"].shift() * data["Market_Returns"]

print("Cumulative Strategy Return:", data["Strategy_Returns"].cumsum().iloc[-1])

import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plt.plot(data.index, data["Close"], label="Close Price", alpha=0.6)
plt.plot(data.index, data["VWAP"], label="VWAP", alpha=0.8, linestyle="--")
plt.legend()
plt.title("AAPL VWAP Reclaim Strategy")
plt.show()
