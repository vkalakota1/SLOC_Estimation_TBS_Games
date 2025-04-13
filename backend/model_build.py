import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import KFold
import numpy as np

# Set the path to the preprocessed dataset (this should be your real data)
data_path = "../data/game_data_processed.csv"
df = pd.read_csv(data_path)

# Check that the dataset has sufficient observations
if df.shape[0] < 15:
    raise ValueError("Insufficient data. Please ensure the dataset has at least 20 observations for a robust model.")

# Confirm that 'SLOC' exists, which is our response variable (actual SLOC counts)
if 'SLOC' not in df.columns:
    raise ValueError("Column 'SLOC' not found in dataset. Please update game_data.csv with real SLOC values.")

# Define predictor columns – using all predictors in this model
predictors = ['NRUL', 'MGOP', 'NPLY', 'ANIM', 'MAP', 'UNU', 'NP']

# Set up X and y for regression (X = predictors, y = SLOC)
X = df[predictors]
y = df['SLOC']

# Add constant term to include the intercept
X = sm.add_constant(X)

# Build the full Ordinary Least Squares (OLS) regression model
model = sm.OLS(y, X).fit()

print("------ REGRESSION MODEL SUMMARY ------")
print(model.summary())

# The summary now gives you real statistical values:
# - R-squared indicates how much variance in SLOC is explained by predictors.
# - Coefficients and p-values show the significance of each predictor.
# Note: A sufficiently large dataset should now yield valid degrees of freedom (Df Residuals > 0).

# --- Forward Stepwise Selection --- #
# If you want to implement forward stepwise selection for predictor inclusion, you can
# use the following simple iterative approach. (This is a prototype implementation for real data.)

def forward_stepwise_selection(X, y, significance_level=0.05):
    """Perform forward stepwise regression to select predictors."""
    initial_features = []
    remaining_features = list(X.columns)
    selected_features = []
    best_model = None

    while remaining_features:
        best_p_value = float('inf')
        best_feature = None
        for feature in remaining_features:
            try:
                features_to_test = selected_features + [feature]
                X_test = X[features_to_test]
                model_test = sm.OLS(y, sm.add_constant(X_test)).fit()
                p_value = model_test.pvalues[feature]
                if p_value < best_p_value:
                    best_p_value = p_value
                    best_feature = feature
            except Exception as e:
                print(f"Error testing feature {feature}: {e}")

        if best_feature is not None and best_p_value < significance_level:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
            best_model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()
            print(f"Added feature '{best_feature}' with p-value {best_p_value:.4f}")
        else:
            break

    return best_model, selected_features

print("\n------ FORWARD STEPWISE SELECTION ------")
# Run forward stepwise selection on the predictors
best_model, selected_features = forward_stepwise_selection(X, y)
if best_model is not None:
    print("\nSelected Features:", selected_features)
    print(best_model.summary())
else:
    print("No predictors met the significance threshold.")

# --- Model Validation: K-Fold Cross-Validation --- #
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mmre_list = []

print("\n------ K-FOLD CROSS-VALIDATION ------")
for fold, (train_index, test_index) in enumerate(kf.split(X), start=1):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    fold_model = sm.OLS(y_train, X_train).fit()
    predictions = fold_model.predict(X_test)
    
    # Calculate Mean Magnitude of Relative Error (MMRE)
    relative_errors = np.abs((y_test - predictions) / y_test)
    fold_mmre = np.mean(relative_errors)
    mmre_list.append(fold_mmre)
    print(f"Fold {fold}: MMRE = {fold_mmre:.4f}")

average_mmre = np.mean(mmre_list)
print(f"\nAverage MMRE across folds: {average_mmre:.4f}")
