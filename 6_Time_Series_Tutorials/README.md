# Time Series Analysis Tutorials

There are two main ways to run the code in this directory:

## Option 1: Jupyter Notebook

Click on the badge below to open the Jupyter Notebook in your browser, in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Girish-Krishnan/ECE-SIPP-Python-ML/blob/main/6_Time_Series_Tutorials/time_series_tutorials.ipynb)

This option does not require you to install anything locally. The notebook contains step-by-step explanations and runnable code cells with interactive plots.

## Option 2: Python Scripts

Install the required packages with pip:

```bash
pip install pandas matplotlib statsmodels
```

### 0. Moving averages
`python 00_moving_average.py`

Generates a noisy sine wave and visualizes both simple and exponential moving averages.

### 1. ARIMA forecasting
`python 01_arima_forecast.py`

Loads the monthly CO2 dataset, fits an ARIMA(1,1,1) model and plots a two-year forecast with confidence intervals.

### 2. Seasonal decomposition
`python 02_seasonal_decomposition.py`

Breaks down the CO2 series into trend, seasonal, and residual components with a 4-panel plot.

### 3. Holt-Winters forecasting
`python 03_holt_winters.py`

Uses Holt-Winters exponential smoothing to forecast the next two years of CO2 concentrations.

### 4. ACF and PACF plots
`python 04_acf_pacf.py`

Displays autocorrelation and partial autocorrelation plots for the CO2 series.
