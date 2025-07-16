import numpy as np


def main():
    x = np.arange(3)
    print("x =", x)

    # Add a scalar (broadcast)
    print("x + 5 =", x + 5)

    # Add a 2D column vector to a row vector
    a = x.reshape(3, 1)
    b = np.array([10, 20, 30])
    print("\na =\n", a)
    print("b =", b)
    print("a + b =\n", a + b)


if __name__ == "__main__":
    main()
