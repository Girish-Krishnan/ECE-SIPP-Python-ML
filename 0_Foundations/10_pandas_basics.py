"""Introductory pandas DataFrame operations."""
import pandas as pd


def main():
    # Create a DataFrame from a Python dictionary
    data = {
        "name": ["Alice", "Bob", "Charlie", "David"],
        "age": [25, 30, 35, 40],
        "score": [85.5, 92.0, 88.0, 95.5],
    }
    df = pd.DataFrame(data)
    print("DataFrame:\n", df, "\n")

    # Basic selection and summary statistics
    print("Names column:\n", df["name"])
    print("Average age:", df["age"].mean())
    print("Describe scores:\n", df["score"].describe())

    # Simple visualization of the scores
    df.plot.bar(x="name", y="score", title="Participant scores", legend=False)
    import matplotlib.pyplot as plt
    plt.xlabel("name")
    plt.ylabel("score")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
