import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================
# Load Data
# ==========================

X_test = pd.read_csv("C:/Users/ADMIN/Desktop/BTech/Codes/Phase 2/X_test.csv")
y_test = pd.read_csv("C:/Users/ADMIN/Desktop/BTech/Codes/Phase 2/y_test.csv").values.ravel()

knn = joblib.load("knn_model.pkl")
svm = joblib.load("svm_model.pkl")

# ==========================
# Predictions
# ==========================

knn_pred = knn.predict(X_test)
svm_pred = svm.predict(X_test)

# ==========================
# Classification Reports
# ==========================

print("\n===== KNN Report =====")
print(classification_report(y_test, knn_pred, zero_division=0))

print("\n===== SVM Report =====")
print(classification_report(y_test, svm_pred, zero_division=0))

# ==========================
# Confusion Matrices
# ==========================

labels = sorted(set(y_test))

plt.figure()
sns.heatmap(confusion_matrix(y_test, knn_pred),
            annot=True, fmt='d')
plt.title("KNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

plt.figure()
sns.heatmap(confusion_matrix(y_test, svm_pred),
            annot=True, fmt='d')
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
