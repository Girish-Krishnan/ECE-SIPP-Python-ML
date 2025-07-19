import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression


def main():
    X, y, coef = make_regression(n_samples=100, n_features=1, noise=10.0,
                                 coef=True, random_state=42)
    model = LinearRegression()
    model.fit(X, y)
    print("True coefficient:", coef)
    print("Learned coefficient:", model.coef_[0])
    print("Intercept:", model.intercept_)


if __name__ == "__main__":
    main()
