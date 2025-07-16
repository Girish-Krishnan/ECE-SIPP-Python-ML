import numpy as np


def main():
    # Random samples from a normal distribution
    samples = np.random.randn(1000)
    print("first five samples:", samples[:5])

    # Compute basic statistics
    print("mean =", samples.mean())
    print("std =", samples.std())

    # Histogram (counts per bin)
    hist, bins = np.histogram(samples, bins=5)
    print("\nhistogram:")
    for b_left, b_right, count in zip(bins[:-1], bins[1:], hist):
        print(f"{b_left: .2f} to {b_right: .2f}: {count}")


if __name__ == "__main__":
    main()
