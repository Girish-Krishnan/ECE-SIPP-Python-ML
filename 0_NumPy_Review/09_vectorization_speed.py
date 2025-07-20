import numpy as np
import time


def main():
    n = 1000000
    data = np.arange(n)

    # Sum using Python loop
    start = time.time()
    total = 0
    for value in data:
        total += value
    loop_time = time.time() - start

    # Sum using vectorized operation
    start = time.time()
    vector_total = np.sum(data)
    vector_time = time.time() - start

    print("loop sum =", total, "took", loop_time, "seconds")
    print("vectorized sum =", vector_total, "took", vector_time, "seconds")


if __name__ == "__main__":
    main()
