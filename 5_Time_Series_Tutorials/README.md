# Time Series Analysis Tutorials

There are two main ways to run the code in this directory:

## Option 1: Jupyter Notebook

Click on the badge below to open the Jupyter Notebook in your browser, in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Girish-Krishnan/ECE-SIPP-Python-ML/blob/main/5_Time_Series_Tutorials/time_series_tutorials.ipynb)

This option does not require you to install anything locally. The notebook contains step-by-step explanations and runnable code cells with interactive plots.

## Option 2: Python Scripts

Install the required packages with pip:

```bash
pip install pandas matplotlib statsmodels
```

### 0. Simple moving average
`python 00_moving_average.py`

Generates a noisy sine wave, computes a rolling mean and displays the result.

### 1. ARIMA forecasting
`python 01_arima_forecast.py`

Loads the monthly CO2 dataset, fits a simple ARIMA(1,1,1) model and plots a one-year forecast.
