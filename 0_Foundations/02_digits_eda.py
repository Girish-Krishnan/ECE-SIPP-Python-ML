"""EDA for the scikit-learn digits dataset."""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, fetch_openml


def main() -> None:
    digits = load_digits()
    df = pd.DataFrame(digits.data)
    df["target"] = digits.target
    print("First five rows:\n", df.head(), "\n")
    print("Summary statistics:\n", df.describe(), "\n")
    df["target"].value_counts().plot.bar()
    plt.title("Digit class distribution")
    plt.xlabel("digit")
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()

    # Show a few example images from the scikit-learn dataset
    fig, axes = plt.subplots(2, 5, figsize=(8, 4))
    for ax, image, label in zip(axes.ravel(), digits.images, digits.target):
        ax.imshow(image, cmap="gray_r")
        ax.set_title(label)
        ax.axis("off")
    plt.suptitle("Sample digits")
    plt.tight_layout()
    plt.show()

    # Fetch the MNIST dataset and plot one example of each digit class
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    images = mnist.data.reshape(-1, 28, 28)
    labels = mnist.target.astype(int)

    fig, axes = plt.subplots(2, 5, figsize=(8, 4))
    for digit in range(10):
        idx = (labels == digit).nonzero()[0][0]
        ax = axes[digit // 5, digit % 5]
        ax.imshow(images[idx], cmap="gray_r")
        ax.set_title(digit)
        ax.axis("off")
    plt.suptitle("Example MNIST digits")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
