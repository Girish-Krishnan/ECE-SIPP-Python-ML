# Basic Machine Learning Tutorials

This folder contains small scikit-learn examples
showing common algorithms. Each script can be run with `python` followed
by the file name.

Firstly, ensure you have scikit-learn and Matplotlib installed:

```bash
pip install scikit-learn matplotlib
```

### 0. Linear regression on synthetic data
`python 00_linear_regression.py`

Generates a noisy 1D dataset and fits a linear regression model,
printing the learned coefficient and intercept.

### 1. Logistic regression on Iris
`python 01_logistic_regression.py`

Splits the Iris dataset into train and test sets and reports the
accuracy of a logistic regression classifier.

### 2. k-NN classification on digits
`python 02_knn_digits.py`

Trains a 3-nearest-neighbor classifier on the handwritten digits dataset
and prints the test accuracy.

### 3. Decision tree classifier
`python 03_decision_tree.py`

Fits a shallow decision tree on the Iris dataset and shows the
classification report on the test split.

### 4. k-means clustering
`python 04_kmeans_clustering.py`

Performs k-means clustering with three clusters on the Iris features and
prints the cluster counts and centers.

### 5. Principal component analysis
`python 05_pca_digits.py`

Reduces the dimensionality of the digits dataset to two components using
PCA and prints the explained variance ratio.

Feel free to modify these examples or use them as starting points for
experimenting with scikit-learn!
