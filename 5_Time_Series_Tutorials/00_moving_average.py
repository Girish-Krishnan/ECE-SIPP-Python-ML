import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate a noisy sine wave time series
rng = pd.date_range(start="2020-01-01", periods=200, freq="D")
data = np.sin(np.linspace(0, 8 * np.pi, len(rng))) + np.random.normal(scale=0.5, size=len(rng))
series = pd.Series(data, index=rng)

# Compute different moving averages
sma20 = series.rolling(window=20).mean()
sma50 = series.rolling(window=50).mean()
ema20 = series.ewm(span=20, adjust=False).mean()

# Plot the series and moving averages
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(series, label="Noisy series", alpha=0.6)
plt.plot(sma20, label="20-day SMA")
plt.plot(sma50, label="50-day SMA")
plt.title("Simple Moving Averages")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(series, label="Noisy series", alpha=0.6)
plt.plot(ema20, label="20-day EMA", color="tab:orange")
plt.title("Exponential Moving Average")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()

plt.tight_layout()
plt.savefig("moving_average.png")
plt.show()
