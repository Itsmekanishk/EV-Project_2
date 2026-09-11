import numpy as np
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# =========================
# 1. Load Full Dataset
# =========================
X = np.load("C:/Users/ADMIN/Desktop/BTech/Codes/Phase 3/X_sequences.npy")
y = np.load("C:/Users/ADMIN/Desktop/BTech/Codes/Phase 3/y_sequences.npy")

print("Full dataset shape:", X.shape)

# =========================
# 2. Train-Test Split (same as Step 1)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# =========================
# 3. Convert Sequence → Tabular
# =========================
X_train_tabular = X_train[:, -1, :]
X_test_tabular = X_test[:, -1, :]

print("Converted train shape:", X_train_tabular.shape)
print("Converted test shape:", X_test_tabular.shape)

# =========================
# 4. KNN Model
# =========================
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_tabular, y_train)

y_pred_knn = knn.predict(X_test_tabular)

print("\n📊 KNN Classification Report:\n")
print(classification_report(y_test, y_pred_knn))

# =========================
# 5. SVM Model
# =========================
svm = SVC()
svm.fit(X_train_tabular, y_train)

y_pred_svm = svm.predict(X_test_tabular)

print("\n📊 SVM Classification Report:\n")
print(classification_report(y_test, y_pred_svm))

# =========================
# 6. Save Predictions
# =========================
np.save("knn_predictions.npy", y_pred_knn)
np.save("svm_predictions.npy", y_pred_svm)

print("✅ KNN & SVM evaluation completed correctly (no data leakage)")