import numpy as np


def main():
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])

    # Element-wise operations
    print("x + y =", x + y)
    print("x * y =", x * y)

    # Vectorized functions
    print("sin(x) =", np.sin(x))

    # Aggregations
    print("mean of y =", y.mean())


if __name__ == "__main__":
    main()
