import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Phase 1 final dataset
df = pd.read_csv("C:/Users/ADMIN/Desktop/BTech/Codes/Phase 1/phase1_ev_attack_dataset.csv")

print("Original shape:", df.shape)
print(df['label'].value_counts())

# Drop non-numerical columns if needed
df = df.drop(columns=["timestamp", "src_id"])

X = df.drop("label", axis=1)
y = df["label"]

# Train-Test split (Stratified this time)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save
pd.DataFrame(X_train).to_csv("X_train_full.csv", index=False)
pd.DataFrame(X_test).to_csv("X_test_full.csv", index=False)
pd.DataFrame(y_train).to_csv("y_train_full.csv", index=False)
pd.DataFrame(y_test).to_csv("y_test_full.csv", index=False)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Dataset ready for proper ML training.")
