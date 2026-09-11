import numpy as np
from sklearn.model_selection import train_test_split

# =========================
# 1. Load Sequence Data
# =========================
X = np.load("C:/Users/ADMIN/Desktop/BTech/Codes/Phase 3/X_sequences.npy")
y = np.load("C:/Users/ADMIN/Desktop/BTech/Codes/Phase 3/y_sequences.npy")   # IMPORTANT: you must have this

print("Full dataset shape:", X.shape)

# =========================
# 2. Train-Test Split (70-30)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y   # VERY IMPORTANT for balanced classes
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# =========================
# 3. Save Test Data
# =========================
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print("✅ X_test.npy and y_test.npy saved")