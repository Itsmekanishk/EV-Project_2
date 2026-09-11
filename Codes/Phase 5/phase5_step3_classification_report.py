import numpy as np
from sklearn.metrics import classification_report

# =========================
# 1. Load Data
# =========================
y_test = np.load("y_test.npy")
y_pred = np.load("lstm_predictions.npy")

# =========================
# 2. Classification Report
# =========================
labels = ["Normal", "DoS", "DDoS", "Malformed"]

report = classification_report(y_test, y_pred, target_names=labels)

print("\n📊 Classification Report:\n")
print(report)

# =========================
# 3. Save Report
# =========================
with open("lstm_classification_report.txt", "w") as f:
    f.write(report)

print("✅ Report saved as lstm_classification_report.txt")