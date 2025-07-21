import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate a noisy sine wave time series
rng = pd.date_range(start="2020-01-01", periods=200, freq="D")
data = np.sin(np.linspace(0, 8 * np.pi, len(rng))) + np.random.normal(scale=0.5, size=len(rng))
series = pd.Series(data, index=rng)

# Compute 20-day simple moving average
rolling = series.rolling(window=20).mean()

# Plot the original series and the moving average
plt.figure(figsize=(8, 4))
plt.plot(series, label="Noisy series")
plt.plot(rolling, label="20-day SMA", linewidth=2)
plt.title("Simple Moving Average")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.savefig("moving_average.png")
plt.show()
