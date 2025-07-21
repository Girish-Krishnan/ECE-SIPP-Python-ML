"""Demonstrates basic NumPy array creation."""
import numpy as np


def main():
    # Create arrays from Python lists
    a = np.array([1, 2, 3])
    print("1D array:", a)

    # Create a 2D array of zeros
    b = np.zeros((2, 3))
    print("\n2D zeros array:\n", b)

    # Create a 3x3 identity matrix
    c = np.eye(3)
    print("\nIdentity matrix:\n", c)

    # More array creation tricks
    d = np.arange(12).reshape(3, 4)
    print("\nReshaped array:\n", d)

    e = np.full((2, 2), 7)
    print("\nConstant array:\n", e)


if __name__ == "__main__":
    main()
