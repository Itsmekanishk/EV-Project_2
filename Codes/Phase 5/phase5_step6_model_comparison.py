import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

# =========================
# 1. Load Predictions
# =========================
y_test = np.load("y_test.npy")

y_pred_lstm = np.load("lstm_predictions.npy")
y_pred_knn = np.load("knn_predictions.npy")
y_pred_svm = np.load("svm_predictions.npy")

# =========================
# 2. Calculate Metrics
# =========================
models = ["LSTM", "KNN", "SVM"]

accuracy = [
    accuracy_score(y_test, y_pred_lstm),
    accuracy_score(y_test, y_pred_knn),
    accuracy_score(y_test, y_pred_svm)
]

f1 = [
    f1_score(y_test, y_pred_lstm, average='weighted'),
    f1_score(y_test, y_pred_knn, average='weighted'),
    f1_score(y_test, y_pred_svm, average='weighted')
]

# =========================
# 3. Create Table
# =========================
df = pd.DataFrame({
    "Model": models,
    "Accuracy": accuracy,
    "F1-Score": f1
})

print("\n📊 Model Comparison Table:\n")
print(df)

df.to_csv("model_comparison.csv", index=False)

# =========================
# 4. Plot Graph
# =========================
plt.figure(figsize=(6,5))

x = range(len(models))

plt.bar(x, accuracy)
plt.xticks(x, models)
plt.ylabel("Accuracy")
plt.title("Model Comparison")

plt.show()

print("✅ Comparison complete")