import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# --- 1. Load data ---
df = pd.read_csv("C:/Users/brije/Downloads/Housing.csv")

# --- 2. Preprocess categorical variables ---
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_cols:
    df[col] = df[col].apply(lambda x: 1 if x == 'yes' else 0)

furnishing_dummies = pd.get_dummies(df['furnishingstatus'], prefix='furnishing', drop_first=True, dtype=int)
df = pd.concat([df, furnishing_dummies], axis=1)
df.drop('furnishingstatus', axis=1, inplace=True)

# --- 3. Split features and target ---
X = df.drop('price', axis=1)
y = df['price']

# --- 4. Scale both features and target ---
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

# --- 5. Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# --- 6. Hyperparameter tuning (for RBF SVR) ---
params = {
    'C': [10, 50, 100],
    'gamma': ['scale', 0.01, 0.1, 1],
    'epsilon': [0.1, 0.2, 0.5]
}

grid = GridSearchCV(SVR(kernel='rbf'), params, cv=3, scoring='r2', n_jobs=-1)
grid.fit(X_train, y_train)

best_svr = grid.best_estimator_
print(f"Best parameters: {grid.best_params_}")

# --- 7. Evaluation ---
y_pred_scaled = best_svr.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

mse = mean_squared_error(y_test_orig, y_pred)
r2 = r2_score(y_test_orig, y_pred)

print("\n--- Tuned RBF SVR Model Evaluation ---")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²): {r2:.2f}")

# --- 8. Plot actual vs predicted ---
plt.figure(figsize=(8, 5))
plt.scatter(y_test_orig, y_pred, color='blue', alpha=0.6)
plt.plot([y_test_orig.min(), y_test_orig.max()],
         [y_test_orig.min(), y_test_orig.max()],
         'r--', lw=2)
plt.title('Actual vs Predicted House Prices')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.grid(True)
plt.show()