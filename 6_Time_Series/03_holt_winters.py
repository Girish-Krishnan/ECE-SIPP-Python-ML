import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load monthly CO2 concentration dataset
co2 = sm.datasets.co2.load_pandas().data["co2"].resample("MS").mean().ffill()

# Split into train and test sets (last 24 months for testing)
train = co2.iloc[:-24]
test = co2.iloc[-24:]

# Fit additive Holt-Winters model with trend and seasonality
model = ExponentialSmoothing(train, seasonal='add', trend='add', seasonal_periods=12)
fit = model.fit()

# Forecast the next 24 months
forecast = fit.forecast(24)

plt.figure(figsize=(10, 6))
plt.plot(co2, label='Observed')
plt.plot(forecast.index, forecast, color='red', label='Holt-Winters Forecast')
plt.axvspan(test.index[0], test.index[-1], color='gray', alpha=0.1, label='Forecast Horizon')
plt.title('Holt-Winters Forecast of CO2')
plt.xlabel('Date')
plt.ylabel('ppm')
plt.legend()
plt.tight_layout()
plt.savefig('holt_winters_forecast.png')
plt.show()
