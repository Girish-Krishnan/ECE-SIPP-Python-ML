"""Simple exploratory data analysis with pandas and Matplotlib."""
import pandas as pd
import matplotlib.pyplot as plt
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


if __name__ == "__main__":
    main()
