"""DataFrame merging and grouping example using the Titanic dataset."""
import pandas as pd


def main() -> None:
    # Load Titanic passenger data from seaborn's GitHub repository
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
    titanic = pd.read_csv(url)
    print("First five rows:\n", titanic.head(), "\n")

    # Create a lookup table mapping passenger class to a simpler category
    class_map = pd.DataFrame(
        {
            "class": ["First", "Second", "Third"],
            "luxury_level": ["high", "medium", "low"],
        }
    )

    # Merge the lookup table with the passenger data
    merged = titanic.merge(class_map, on="class")

    # Group by the luxury level and compute average fare and survival rate
    summary = merged.groupby("luxury_level").agg({"fare": "mean", "survived": "mean"})
    print("Summary by luxury level:\n", summary)


if __name__ == "__main__":
    main()
