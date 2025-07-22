# Supervised Machine Learning Tutorials

There are two main ways to run the code in this directory:

## Option 1: Jupyter Notebook

Click on the badge below to open the Jupyter Notebook in your browser, in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Girish-Krishnan/ECE-SIPP-Python-ML/blob/main/1_Supervised_ML/supervised_ml_tutorials.ipynb)

This option does not require you to install anything locally. The notebook contains step-by-step explanations and runnable code cells.

## Option 2: Python Scripts

You can also run the examples as standalone Python scripts. Firstly, ensure you have scikit-learn and Matplotlib installed:

```bash
pip install scikit-learn matplotlib
```

### 0. Linear regression from scratch and scikit-learn
`python 00_linear_regression.py`

Computes the closed-form solution for a synthetic dataset and compares it with `LinearRegression`.

### 1. k-NN classification on digits
`python 02_knn_digits.py`

Implements a simple k-NN classifier before showing `KNeighborsClassifier`.

### 2. Perceptron with gradient descent
`python 03_perceptron.py`

Trains a perceptron from scratch on a synthetic dataset and compares with scikit-learn.

### 3. Logistic regression on Iris
`python 04_logistic_regression.py`

Uses gradient descent to fit a logistic regression model and then uses scikit-learn's implementation.

Feel free to modify these examples or use them as starting points for experimenting with scikit-learn!
