import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ==============================
# Load Full Dataset
# ==============================

X_train = pd.read_csv("X_train_full.csv")
X_test = pd.read_csv("X_test_full.csv")
y_train = pd.read_csv("y_train_full.csv").values.ravel()
y_test = pd.read_csv("y_test_full.csv").values.ravel()

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# ==============================
# 1️⃣ KNN Model
# ==============================

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)

print("\n===== KNN Results =====")
print("Accuracy:", accuracy_score(y_test, knn_pred))
print(classification_report(y_test, knn_pred))

joblib.dump(knn, "knn_model_full.pkl")

# ==============================
# 2️⃣ SVM Model
# ==============================

svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)

svm_pred = svm.predict(X_test)

print("\n===== SVM Results =====")
print("Accuracy:", accuracy_score(y_test, svm_pred))
print(classification_report(y_test, svm_pred))

joblib.dump(svm, "svm_model_full.pkl")

print("\n✅ Full baseline models trained and saved.")
