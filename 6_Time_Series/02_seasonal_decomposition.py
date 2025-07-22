import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the monthly CO2 dataset
co2 = sm.datasets.co2.load_pandas().data["co2"].resample("MS").mean().ffill()

# Perform additive seasonal decomposition
result = seasonal_decompose(co2, model="additive", period=12)

# Plot the decomposed components
fig = result.plot()
fig.set_size_inches(10, 8)
plt.suptitle("Seasonal Decomposition of CO2", y=1.02)
plt.tight_layout()
plt.savefig("seasonal_decomposition.png")
plt.show()
