"""Advanced data visualization with pandas and seaborn."""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main() -> None:
    # Load the built-in tips dataset from seaborn
    tips = sns.load_dataset("tips")
    print("First five rows:\n", tips.head(), "\n")

    # Pairplot showing relationships between variables
    sns.pairplot(tips, hue="sex")
    plt.suptitle("Pairplot of tips dataset", y=1.02)
    plt.show()

    # Correlation heatmap of numeric columns
    corr = tips.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation heatmap")
    plt.show()


if __name__ == "__main__":
    main()
