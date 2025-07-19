import numpy as np


def main():
    a = np.arange(10)
    print("a =", a)

    # Basic slicing
    print("a[2:5] =", a[2:5])

    # Boolean masking
    mask = a % 2 == 0
    print("even elements =", a[mask])

    # Fancy indexing
    idx = [1, 3, 5]
    print("selected indices =", a[idx])


if __name__ == "__main__":
    main()
