"""Write and read pandas DataFrames using CSV files."""
import pandas as pd


def main():
    # Create a simple DataFrame
    data = {
        "city": ["San Diego", "Los Angeles", "San Francisco"],
        "population": [1.4, 3.9, 0.88],
    }
    df = pd.DataFrame(data)
    print("Original DataFrame:\n", df, "\n")

    # Write the DataFrame to CSV
    csv_path = "cities.csv"
    df.to_csv(csv_path, index=False)
    print(f"Data written to {csv_path}")

    # Read the file back in
    loaded = pd.read_csv(csv_path)
    print("\nLoaded from CSV:\n", loaded)


if __name__ == "__main__":
    main()
