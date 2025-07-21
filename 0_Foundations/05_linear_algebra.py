"""Linear algebra operations."""
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

    # Eigen decomposition
    w, v = np.linalg.eig(A)
    print("\neigenvalues =", w)
    print("eigenvectors =\n", v)

    # Solve a linear system A x = b
    b_vec = np.array([5, 6])
    x = np.linalg.solve(A, b_vec)
    print("\nsolution to A x = b where b=[5,6]:", x)


if __name__ == "__main__":
    main()
