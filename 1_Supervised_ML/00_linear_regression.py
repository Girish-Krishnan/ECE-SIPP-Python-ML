import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression


def closed_form_lr(X, y):
    X_b = np.c_[np.ones((len(X), 1)), X]
    theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    return theta[1:], theta[0]


def main():
    X, y, coef = make_regression(n_samples=100, n_features=1, noise=10.0,
                                 coef=True, random_state=42)
    w, b = closed_form_lr(X, y)
    print("True coefficient:", coef)
    print("Closed-form coefficient:", w[0])
    print("Intercept:", b)

    model = LinearRegression()
    model.fit(X, y)
    print("scikit-learn coefficient:", model.coef_[0])
    print("scikit-learn intercept:", model.intercept_)


if __name__ == "__main__":
    main()
