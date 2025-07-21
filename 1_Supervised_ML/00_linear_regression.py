import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def main():
    X, y, coef = make_regression(n_samples=100, n_features=1, noise=10.0,
                                 coef=True, random_state=42)
    model = LinearRegression()
    model.fit(X, y)
    print("True coefficient:", coef)
    print("Learned coefficient:", model.coef_[0])
    print("Intercept:", model.intercept_)

    # Visualization of the fitted line
    x_grid = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_pred = model.predict(x_grid)

    plt.scatter(X, y, color="blue", label="Data")
    plt.plot(x_grid, y_pred, color="red", label="Fit")
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
