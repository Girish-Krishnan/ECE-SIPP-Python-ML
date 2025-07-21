# Foundations: NumPy and pandas

There are two main ways to run the code in this directory:

## Option 1: Jupyter Notebook

Click on the badge below to open the notebook in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Girish-Krishnan/ECE-SIPP-Python-ML/blob/main/0_Foundations/foundations.ipynb)

The notebook contains step-by-step explanations for beginners. No local installation is required when using Colab.

## Option 2: Python Scripts

Each script corresponds to a section in the notebook. Install the required libraries if needed:

```bash
pip install numpy pandas matplotlib scikit-learn
```

### 0. NumPy basics
`python 00_numpy_basics.py`

Create arrays of different shapes and data types.

### 1. Array math
`python 01_numpy_math.py`

Element-wise operations and vectorized functions.

### 2. Indexing and slicing
`python 02_numpy_indexing.py`

Accessing data efficiently with slices and boolean masks.

### 3. Broadcasting
`python 03_numpy_broadcasting.py`

Automatic expansion of array shapes for arithmetic.

### 4. Random numbers & statistics
`python 04_random_statistics.py`

Drawing random samples and computing summary statistics.

### 5. Polynomial fitting
`python 05_polynomial_fit.py`

Fit a noisy quadratic curve and visualize the result.

### 6. Saving and loading
`python 06_numpy_io.py`

Persist arrays using `npy`, `npz` and text formats.

### 7. Vectorization speed comparison
`python 07_vectorization_speed.py`

Demonstrate the performance benefits of vectorized code.

### 8. pandas basics
`python 08_pandas_basics.py`

Creating DataFrames and computing basic statistics.

### 9. Exploratory data analysis
`python 09_eda_plotting.py`

Load the Iris dataset and create simple visualizations.

### 10. CSV input and output
`python 10_csv_io.py`

Write a DataFrame to a CSV file and read it back.

### 11. DataFrame merging and grouping
`python 11_pandas_merge_groupby.py`

Load the Titanic dataset from an online CSV, merge with a lookup table and
compute grouped statistics.

Feel free to modify these examples or use them as a starting point for your own experiments!
