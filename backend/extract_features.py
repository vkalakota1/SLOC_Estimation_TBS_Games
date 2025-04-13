import pandas as pd

# Path to the preprocessed dataset
processed_data_path = "../data/game_data_processed.csv"
df = pd.read_csv(processed_data_path)

# Display the extracted features for each game
print("Extracted Features from Processed Data:\n")
for index, row in df.iterrows():
    print("Game:", row['Game'])
    print("  NRUL:", row['NRUL'])
    print("  MGOP:", row['MGOP'])
    print("  NPLY:", row['NPLY'])
    print("  ANIM:", row['ANIM'])
    print("  MAP:", row['MAP'])
    print("  UNU:", row['UNU'])
    print("  NP:", row['NP'])
    print("  GF:", row['GF'])
    print("-" * 40)
