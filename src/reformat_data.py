
# Import packages
import pandas as pd

columns = [
    "Class", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium",
    "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins",
    "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"
]

# Import Dataset
# Load the wine.data file using the specified column names
df = pd.read_csv('./data/raw/wine.data', header=None, names=columns)

# Save the DataFrame to a CSV file
df.to_csv('./data/processed/wine_data.csv', index=False)
