import numpy as np
import matplotlib.pyplot as plt


def main():
    # Random samples from a normal distribution
    samples = np.random.randn(1000)
    print("first five samples:", samples[:5])

    # Compute basic statistics
    print("mean =", samples.mean())
    print("std =", samples.std())

    print("25th percentile =", np.percentile(samples, 25))

    # Histogram (counts per bin)
    hist, bins = np.histogram(samples, bins=5)
    print("\nhistogram:")
    for b_left, b_right, count in zip(bins[:-1], bins[1:], hist):
        print(f"{b_left: .2f} to {b_right: .2f}: {count}")

    # Visualize the distribution
    plt.hist(samples, bins=30, density=True, alpha=0.7)
    plt.title("Histogram of random samples")
    plt.xlabel("value")
    plt.ylabel("density")
    plt.show()


if __name__ == "__main__":
    main()
