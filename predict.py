import pandas as pd
import joblib
from utils import prepare_features
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Load model and scaler
model = joblib.load("model/random_forest.pkl")
scaler = joblib.load("model/scaler.pkl")

# Load test data
df = pd.read_csv("data/test.csv")
X_scaled, y_true, _ = prepare_features(df)

# Predict using classifier
y_pred = model.predict(X_scaled)
df["predicted_anomaly"] = y_pred

# Save results
df.to_csv("data/predicted.csv", index=False)

# Class distribution check
print("\n✅ Class distribution in test set:")
print(df["anomaly"].value_counts())

# Evaluation
print("\n✅ Evaluation Metrics (Supervised):")
print(f"Accuracy Score : {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision Score: {precision_score(y_true, y_pred):.4f}")
print(f"Recall Score   : {recall_score(y_true, y_pred):.4f}")
print(f"F1 Score       : {f1_score(y_true, y_pred):.4f}")
print("\nDetailed classification report:")
print(classification_report(y_true, y_pred, target_names=["Anomaly", "Normal"]))
