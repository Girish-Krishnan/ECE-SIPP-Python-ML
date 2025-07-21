import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

# Load monthly CO2 concentration dataset
co2 = sm.datasets.co2.load_pandas().data
co2 = co2["co2"].resample("MS").mean().ffill()

# Split into train (all but last 24 months) and test sets
train = co2.iloc[:-24]
test = co2.iloc[-24:]

# Fit ARIMA(1,1,1) on the training data
model = ARIMA(train, order=(1, 1, 1))
model_fit = model.fit()

# Forecast the next 24 months with confidence intervals
pred = model_fit.get_forecast(steps=24)
pred_mean = pred.predicted_mean
pred_ci = pred.conf_int()

# Plot the full series and forecast
plt.figure(figsize=(10, 6))
plt.plot(co2, label="Observed")
plt.plot(pred_mean.index, pred_mean, color="red", label="Forecast")
plt.fill_between(
    pred_ci.index,
    pred_ci.iloc[:, 0],
    pred_ci.iloc[:, 1],
    color="red",
    alpha=0.3,
    label="95% CI",
)
plt.axvspan(test.index[0], test.index[-1], color="gray", alpha=0.1, label="Forecast Horizon")
plt.title("ARIMA(1,1,1) Forecast of CO2")
plt.xlabel("Date")
plt.ylabel("ppm")
plt.legend()
plt.tight_layout()
plt.savefig("arima_forecast.png")
plt.show()
