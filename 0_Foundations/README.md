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

### 1. Exploratory data analysis
`python 01_eda_plotting.py`

Load the Iris dataset and create simple visualizations.

### 2. CSV input and output
`python 02_csv_io.py`

Write a DataFrame to a CSV file and read it back.

### 3. DataFrame merging and grouping
`python 03_pandas_merge_groupby.py`

Load the Titanic dataset from an online CSV, merge with a lookup table and compute grouped statistics.

### 4. Advanced data visualization
`python 04_advanced_visualization.py`

Visualize relationships in the tips dataset using pair plots and a correlation heatmap.

Feel free to modify these examples or use them as a starting point for your own experiments!
