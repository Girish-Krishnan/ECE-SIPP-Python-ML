import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def closed_form_poly(X, y, degree=2):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    X_b = np.c_[np.ones((len(X_poly), 1)), X_poly]
    theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    return theta, poly


def main():
    X, y, _ = make_regression(n_samples=100, n_features=1, noise=15.0,
                              coef=True, random_state=42)
    theta, poly = closed_form_poly(X, y, degree=2)
    print("Closed-form coefficients:", theta[1:])
    print("Intercept:", theta[0])

    model = make_pipeline(PolynomialFeatures(degree=2, include_bias=False),
                          LinearRegression())
    model.fit(X, y)
    lr = model.named_steps["linearregression"]
    print("scikit-learn coefficients:", lr.coef_)
    print("scikit-learn intercept:", lr.intercept_)


if __name__ == "__main__":
    main()
