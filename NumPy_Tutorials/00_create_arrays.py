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


if __name__ == "__main__":
    main()
