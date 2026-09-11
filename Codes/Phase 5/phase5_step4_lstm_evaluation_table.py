import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# Load data
y_test = np.load("y_test.npy")
y_pred = np.load("lstm_predictions.npy")

# Labels
classes = ["Normal", "DoS", "DDoS", "Malformed"]

# Calculate metrics
precision = precision_score(y_test, y_pred, average=None)
recall = recall_score(y_test, y_pred, average=None)
f1 = f1_score(y_test, y_pred, average=None)

# Create DataFrame
df = pd.DataFrame({
    "Class": classes,
    "Precision": precision,
    "Recall": recall,
    "F1-Score": f1
})

print("\n📊 LSTM Evaluation Table:\n")
print(df)

# Save as CSV
df.to_csv("lstm_evaluation_table.csv", index=False)

print("✅ Table saved as lstm_evaluation_table.csv")