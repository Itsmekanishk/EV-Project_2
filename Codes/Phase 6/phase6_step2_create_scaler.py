import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
df = pd.read_csv("C:/Users/ADMIN/Desktop/BTech/Codes/Phase 1/phase1_ev_attack_dataset.csv")

features = ["request_rate", "packet_size", "time_gap"]

X = df[features]

# Create scaler
scaler = StandardScaler()
scaler.fit(X)

# Save scaler
joblib.dump(scaler, "../Phase 6/scaler.pkl")

print("✅ Scaler created and saved as scaler.pkl")