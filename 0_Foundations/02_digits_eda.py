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


if __name__ == "__main__":
    main()
