"""
This module processes a wine dataset by loading it into a DataFrame and saving
it as a CSV file.

Steps:
1. Define the column names for the dataset.
2. Load the dataset into a pandas DataFrame.
3. Save the DataFrame to a CSV file.

"""

import pandas as pd

# Define the column names for the dataset
columns = [
    "Class", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium",
    "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins",
    "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"
]


def reformat_wine_data(input_file_path: str,
                       output_file_path: str,
                       column_names: list[str]) -> None:
    """
    Load the wine dataset from a file and save it as a CSV.

    Parameters:
    input_file (str): Path to the input data file.
    output_file (str): Path to the output CSV file.
    column_names (list[str]): List of column names for the dataset.
    """
    # Load the wine data file using the specified column names
    df: pd.DataFrame = pd.read_csv(input_file_path,
                                   header=None,
                                   names=column_names)

    # Save the DataFrame to a CSV file
    df.to_csv(output_file_path, index=False)


# File paths
input_data_path = './data/raw/wine.data'
output_data_path = './data/processed/wine_data.csv'

# Load the dataset and save it as a CSV file
reformat_wine_data(input_data_path, output_data_path, columns)
