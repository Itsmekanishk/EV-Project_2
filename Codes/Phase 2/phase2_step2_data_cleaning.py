import pandas as pd

# Load dataset
df = pd.read_csv("C:/Users/ADMIN/Desktop/BTech/Codes/Phase 1/phase1_ev_attack_dataset.csv")

print("Original Shape:", df.shape)

# Step 2.1: Remove invalid packet sizes
df = df[df['packet_size'] > 0]

# Step 2.2: Remove invalid time gaps
df = df[df['time_gap'] > 0]

print("Shape After Cleaning:", df.shape)

# Step 2.3: Verify no invalid values remain
print("\nPacket Size Min:", df['packet_size'].min())
print("Time Gap Min:", df['time_gap'].min())

# Step 2.4: Label distribution after cleaning
print("\nLabel Distribution After Cleaning:")
print(df['label'].value_counts())

# Save cleaned dataset (IMPORTANT)
df.to_csv("phase2_cleaned_ev_dataset.csv", index=False)

print("\nCleaned dataset saved as: phase2_cleaned_ev_dataset.csv")
