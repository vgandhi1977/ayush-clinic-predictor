import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# Load cleaned dataset
df = pd.read_csv("ndap_data/AYUSH_Merged_Cleaned.csv")

# Define features and target
features = ['TotalPopulation', 'TotalPopulationUrban', 'TotalPopulationRural',
            'LandArea', 'LandAreaUrban', 'LandAreaRural', 'NumberOfHouseholds']
target = 'Total_AYUSH'

X = df[features]
y = df[target]

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"R² score: {r2_score(y_test, y_pred):.2f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")

# Save model
joblib.dump(model, "ayush_model.pkl")
print("✅ Model saved as ayush_model.pkl")

