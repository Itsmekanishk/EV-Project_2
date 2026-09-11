import pandas as pd

# Load cleaned dataset
df = pd.read_csv("phase2_cleaned_ev_dataset.csv")

# -----------------------------
# Feature Engineering per src_id
# -----------------------------
feature_df = df.groupby('src_id').agg(
    mean_requests_per_sec=('request_rate', 'mean'),
    avg_packet_size=('packet_size', 'mean'),
    iat_variance=('time_gap', 'var'),
    burst_duration=('request_rate', lambda x: (x > 50).sum()),
    label=('label', 'max')  # src-level label
).reset_index()

print("Feature-engineered dataset shape:")
print(feature_df.shape)

print("\nFeature preview:")
print(feature_df.head())

# Save engineered features
feature_df.to_csv("phase2_feature_engineered_dataset.csv", index=False)

print("\nFeature-engineered dataset saved as: phase2_feature_engineered_dataset.csv")
