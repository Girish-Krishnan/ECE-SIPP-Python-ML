# Foundations: pandas data analysis

There are two main ways to run the code in this directory:

## Option 1: Jupyter Notebook

Click on the badge below to open the notebook in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Girish-Krishnan/ECE-SIPP-Python-ML/blob/main/0_Foundations/foundations.ipynb)

The notebook contains step-by-step explanations for beginners. No local installation is required when using Colab.

## Option 2: Python Scripts

Each script corresponds to a section in the notebook. Install the required libraries if needed:

```bash
pip install pandas matplotlib seaborn scikit-learn
```

### 0. pandas basics
`python 00_pandas_basics.py`

Creating DataFrames and computing basic statistics.

### 1. Exploratory data analysis (Iris dataset)
`python 01_eda_plotting.py`

Load the Iris dataset and create simple visualizations. The script now also
generates a pair plot and correlation heatmap for all four features.

### 2. Digits dataset EDA
`python 02_digits_eda.py`

Inspect the digits dataset, plot the distribution of target classes, and display
example images from the built-in digits dataset.

### 3. Handling missing data
`python 03_missing_data.py`

Examples of filling and dropping missing values.

Feel free to modify these examples or use them as a starting point for your own experiments!
