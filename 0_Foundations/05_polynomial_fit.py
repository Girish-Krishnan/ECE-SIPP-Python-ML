"""Polynomial curve fitting example."""
import numpy as np
import matplotlib.pyplot as plt


def main():
    # Create noisy quadratic data
    rng = np.random.default_rng(0)
    x = np.linspace(-3, 3, 20)
    y = 0.5 * x**2 - x + 2 + rng.normal(scale=1.0, size=x.shape)

    # Fit a second degree polynomial
    coeffs = np.polyfit(x, y, deg=2)
    print("coefficients:", coeffs)

    # Evaluate the fitted polynomial
    p = np.poly1d(coeffs)
    y_fit = p(x)

    # Show first few fitted values
    print("\nfirst 5 fitted values:", y_fit[:5])

    # Plot the data and fitted curve
    plt.scatter(x, y, label="data")
    plt.plot(x, y_fit, color="red", label="fit")
    plt.title("Polynomial fit")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
