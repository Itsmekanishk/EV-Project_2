import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
X_test = pd.read_csv("X_test_full.csv")
y_test = pd.read_csv("y_test_full.csv").values.ravel()

knn = joblib.load("knn_model_full.pkl")
svm = joblib.load("svm_model_full.pkl")

knn_pred = knn.predict(X_test)
svm_pred = svm.predict(X_test)

# ==========================
# Confusion Matrix - KNN
# ==========================
plt.figure()
sns.heatmap(confusion_matrix(y_test, knn_pred),
            annot=True, fmt='d')
plt.title("KNN Confusion Matrix (Full Dataset)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==========================
# Confusion Matrix - SVM
# ==========================
plt.figure()
sns.heatmap(confusion_matrix(y_test, svm_pred),
            annot=True, fmt='d')
plt.title("SVM Confusion Matrix (Full Dataset)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
