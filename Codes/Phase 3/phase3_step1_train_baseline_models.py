import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ===============================
# 1️⃣ Load Phase 2 processed data
# ===============================

X_train = pd.read_csv("C:/Users/ADMIN/Desktop/BTech/Codes/Phase 2/X_train.csv")
X_test = pd.read_csv("C:/Users/ADMIN/Desktop/BTech/Codes/Phase 2/X_test.csv")
y_train = pd.read_csv("C:/Users/ADMIN/Desktop/BTech/Codes/Phase 2/y_train.csv").values.ravel()
y_test = pd.read_csv("C:/Users/ADMIN/Desktop/BTech/Codes/Phase 2/y_test.csv").values.ravel()

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# ===============================
# 2️⃣ Train KNN
# ===============================

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)

print("\n===== KNN Results =====")
print("Accuracy:", accuracy_score(y_test, knn_pred))
print(classification_report(y_test, knn_pred))

joblib.dump(knn, "knn_model.pkl")

# ===============================
# 3️⃣ Train SVM
# ===============================

svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)

svm_pred = svm.predict(X_test)

print("\n===== SVM Results =====")
print("Accuracy:", accuracy_score(y_test, svm_pred))
print(classification_report(y_test, svm_pred))

joblib.dump(svm, "svm_model.pkl")

print("\n✅ Baseline models trained and saved.")
