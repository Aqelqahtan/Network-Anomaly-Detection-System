import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load predicted results
df = pd.read_csv("data/predicted.csv")

# Compute confusion matrix
y_true = df["anomaly"]
y_pred = df["predicted_anomaly"]
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

# Plot
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=["Anomaly", "Normal"],
            yticklabels=["Anomaly", "Normal"])
plt.title("Confusion Matrix (Supervised Model)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()
