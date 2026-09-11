import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load feature-engineered dataset
df = pd.read_csv("phase2_feature_engineered_dataset.csv")

# Separate features and label
X = df[['mean_requests_per_sec', 'avg_packet_size', 'iat_variance', 'burst_duration']]
y = df['label']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Train-test split WITHOUT stratification (due to small sample size)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df,
    y,
    test_size=0.30,
    random_state=42
)

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)

print("\nTraining label distribution:")
print(y_train.value_counts())

print("\nTesting label distribution:")
print(y_test.value_counts())

# Save datasets
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("\nPhase 2 completed successfully.")
print("Saved: X_train.csv, X_test.csv, y_train.csv, y_test.csv")
