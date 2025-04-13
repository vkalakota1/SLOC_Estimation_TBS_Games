import pandas as pd
from scipy import stats

# Path to the original dataset
data_path = "../data/game_data.csv"
df = pd.read_csv(data_path)

# Print a summary of the original data
print("Original Data Summary:")
print(df.describe())

# List of predictor columns (excluding the 'Game' column)
predictors = ['NRUL', 'MGOP', 'NPLY', 'ANIM', 'MAP', 'UNU', 'NP']

# Normalize the predictors using standard scaling:
# (subtract the mean and divide by the standard deviation)
df[predictors] = (df[predictors] - df[predictors].mean()) / df[predictors].std()

# Remove outliers using z-score filtering: keep rows where all predictors have |z| < 3
df_clean = df[(abs(stats.zscore(df[predictors])) < 3).all(axis=1)]

# Save the cleaned dataset to a new CSV file
processed_data_path = "../data/game_data_processed.csv"
df_clean.to_csv(processed_data_path, index=False)
print("\nPreprocessing complete. Cleaned data saved to 'game_data_processed.csv'.")
