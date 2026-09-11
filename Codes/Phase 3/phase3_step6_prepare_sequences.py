import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ==========================
# 1️⃣ Load Dataset
# ==========================

df = pd.read_csv("C:/Users/ADMIN/Desktop/BTech/Codes/Phase 1/phase1_ev_attack_dataset.csv")

# Sort by timestamp
df = df.sort_values(by="timestamp").reset_index(drop=True)

# Drop non-numerical columns
df = df.drop(columns=["src_id"])

# Separate features & label
X = df.drop("label", axis=1)
y = df["label"].values

# ==========================
# 2️⃣ Normalize Features
# ==========================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================
# 3️⃣ Create Sequences
# ==========================

SEQ_LEN = 10

X_sequences = []
y_sequences = []

for i in range(len(X_scaled) - SEQ_LEN):
    X_sequences.append(X_scaled[i:i+SEQ_LEN])
    y_sequences.append(y[i+SEQ_LEN-1])  # label of last item

X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)

print("Sequence shape:", X_sequences.shape)
print("Labels shape:", y_sequences.shape)

# Save for training
np.save("X_sequences.npy", X_sequences)
np.save("y_sequences.npy", y_sequences)

print("✅ Sequential dataset ready for LSTM.")
