import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Load Processed Data
df = pd.read_csv("../data/game_data_processed.csv")

# Drop 'Game' Column (Non-Numeric)
df = df.drop(columns=['Game'])

X = df.drop(columns=['SLOC'])
y = df['SLOC']

# Train Model
model = LinearRegression()
model.fit(X, y)

# Save Model
with open('saved_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model Saved Successfully To 'saved_model.pkl'")
