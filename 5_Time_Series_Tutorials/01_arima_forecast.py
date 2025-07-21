import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

# Load monthly CO2 concentration dataset
co2 = sm.datasets.co2.load_pandas().data
co2 = co2['co2'].resample('MS').mean().ffill()

# Split into train and test
train = co2[:-12]

# Fit ARIMA(1,1,1)
model = ARIMA(train, order=(1, 1, 1))
model_fit = model.fit()

# Forecast the next 12 months
forecast = model_fit.forecast(steps=12)

# Plot
plt.figure(figsize=(8, 4))
plt.plot(co2, label='CO2')
plt.plot(forecast.index, forecast, color='red', label='Forecast')
plt.title('ARIMA Forecast of CO2')
plt.xlabel('Date')
plt.ylabel('ppm')
plt.legend()
plt.tight_layout()
plt.savefig('arima_forecast.png')
plt.show()
