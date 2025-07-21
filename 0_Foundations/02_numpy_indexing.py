"""Indexing and slicing examples."""
import numpy as np


def main():
    a = np.arange(10)
    print("a =", a)

    # Basic slicing
    print("a[2:5] =", a[2:5])

    # Negative step slicing
    print("reverse =", a[::-1])

    # Boolean masking
    mask = a % 2 == 0
    print("even elements =", a[mask])

    # Fancy indexing
    idx = [1, 3, 5]
    print("selected indices =", a[idx])

    # Adding a new axis
    col_vec = a[:, np.newaxis]
    print("\ncolumn vector shape:", col_vec.shape)


if __name__ == "__main__":
    main()
