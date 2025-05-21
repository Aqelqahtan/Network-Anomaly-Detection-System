import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from utils import prepare_features

# Load training data
df = pd.read_csv("data/train.csv")
X_scaled, y, scaler = prepare_features(df)

# Train a supervised classifier
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
model.fit(X_scaled, y)

# Save model and scaler
joblib.dump(model, "model/random_forest.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("✅ Supervised Random Forest model trained and saved.")
