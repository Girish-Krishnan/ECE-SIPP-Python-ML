"""Simple exploratory data analysis with pandas and Matplotlib."""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets


def main():
    # Load the Iris dataset from scikit-learn and put it in a DataFrame
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target

    # Display the first few rows
    print("First five rows:\n", df.head(), "\n")

    # Summary statistics
    print("Summary statistics:\n", df.describe(), "\n")

    # Scatter plot of two features
    df.plot.scatter(x="sepal length (cm)", y="petal length (cm)", c="target", cmap="viridis")
    plt.title("Iris feature scatter plot")
    plt.show()

    # Histogram of petal widths
    df["petal width (cm)"].hist(bins=20)
    plt.title("Petal width distribution")
    plt.xlabel("width (cm)")
    plt.ylabel("count")
    plt.show()

    # Pairplot of all numerical features colored by class
    sns.pairplot(df, hue="target")
    plt.suptitle("Iris pair plot", y=1.02)
    plt.show()

    # Correlation heatmap
    corr = df.drop(columns=["target"]).corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Feature correlation heatmap")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
