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

### 0. Linear regression on synthetic data
`python 00_linear_regression.py`

Generates a noisy 1D dataset and fits a linear regression model, printing the learned coefficient and intercept.

### 1. Logistic regression on Iris
`python 01_logistic_regression.py`

Splits the Iris dataset into train and test sets and reports the accuracy of a logistic regression classifier.

### 2. k-NN classification on digits
`python 02_knn_digits.py`

Trains a 3-nearest-neighbor classifier on the handwritten digits dataset and prints the test accuracy.

### 3. Decision tree classifier
`python 03_decision_tree.py`

Fits a shallow decision tree on the Iris dataset and shows the classification report on the test split.

Feel free to modify these examples or use them as starting points for experimenting with scikit-learn!
