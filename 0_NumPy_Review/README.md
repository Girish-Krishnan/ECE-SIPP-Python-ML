# NumPy Tutorials

There are two main ways to run the code in this directory:

## Option 1: Jupyter Notebook

Click on the badge below to open the Jupyter Notebook in your browser, in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Girish-Krishnan/ECE-SIPP-Python-ML/blob/main/0_NumPy_Review/numpy_review.ipynb)

This option does not require you to install anything locally on your machine. The notebook contains explanations and code examples that you can run interactively.

## Option 2: Python Scripts

You can also run the examples as standalone Python scripts. Each script corresponds to a section in the Jupyter Notebook. Firstly, ensure you have NumPy and Matplotlib installed:

```bash
pip install numpy matplotlib
```

### 0. Create arrays
`python 00_create_arrays.py`

Demonstrates several ways to create NumPy arrays including zeros and identity matrices.

### 1. Array math
`python 01_array_math.py`

Shows element-wise arithmetic, vectorized functions and basic aggregations.

### 2. Indexing and slicing
`python 02_indexing_slicing.py`

Covers slicing, boolean masks and fancy indexing to access data efficiently.

### 3. Broadcasting
`python 03_broadcasting.py`

Illustrates how operations automatically expand array shapes without copies.

### 4. Random numbers & statistics
`python 04_random_and_statistics.py`

Uses NumPy's random module to draw samples and compute statistics like mean, standard deviation and simple histograms.
Displays a histogram of the samples.

### 5. Linear algebra
`python 05_linear_algebra.py`

Performs matrix multiplication, computes a determinant and finds the inverse of a matrix using `numpy.linalg`.

### 6. Polynomial fitting
`python 06_polynomial_fit.py`

Generates noisy quadratic data, fits a polynomial with `numpy.polyfit` and evaluates the fitted curve.
Shows a scatter plot of the data alongside the fitted curve.

### 7. Saving and loading
`python 07_saving_loading.py`

Shows how to persist arrays using `np.save`, `np.load` and `savetxt`.

### 8. Fourier transforms
`python 08_fourier_transform.py`

Computes the discrete Fourier transform of a synthetic signal using `numpy.fft`.
Plots the original signal and its frequency spectrum.

### 9. Vectorization speed comparison
`python 09_vectorization_speed.py`

Shows how vectorized operations in NumPy can drastically outperform equivalent Python loops.

Feel free to modify the scripts or use them as starting points for your own experiments with NumPy!
