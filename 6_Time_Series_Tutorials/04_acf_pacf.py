import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load monthly CO2 concentration dataset
co2 = sm.datasets.co2.load_pandas().data["co2"].resample("MS").mean().ffill()

fig, axes = plt.subplots(2, 1, figsize=(10, 8))
plot_acf(co2, lags=40, ax=axes[0])
plot_pacf(co2, lags=40, ax=axes[1])
axes[0].set_title('Autocorrelation')
axes[1].set_title('Partial Autocorrelation')
plt.tight_layout()
plt.savefig('acf_pacf.png')
plt.show()
