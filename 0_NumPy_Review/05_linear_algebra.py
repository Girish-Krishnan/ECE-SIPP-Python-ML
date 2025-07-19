import numpy as np


def main():
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    print("A =\n", A)
    print("B =\n", B)

    # Matrix multiplication
    C = A @ B
    print("\nA @ B =\n", C)

    # Determinant and inverse
    det = np.linalg.det(A)
    inv = np.linalg.inv(A)
    print("\ndet(A) =", det)
    print("inv(A) =\n", inv)


if __name__ == "__main__":
    main()
