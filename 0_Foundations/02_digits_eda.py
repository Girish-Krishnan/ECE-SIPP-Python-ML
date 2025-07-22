"""EDA for the scikit-learn digits dataset."""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits


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



if __name__ == "__main__":
    main()
