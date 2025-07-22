"""Demonstrate handling of missing data in pandas."""
import pandas as pd
import numpy as np


def main() -> None:
    df = pd.DataFrame({
        "A": [1, 2, np.nan, 4],
        "B": [5, np.nan, np.nan, 8],
    })
    print("Original:\n", df, "\n")
    print("Fill missing values with column means:\n", df.fillna(df.mean(numeric_only=True)))
    print("\nDrop rows with any missing values:\n", df.dropna())


if __name__ == "__main__":
    main()
