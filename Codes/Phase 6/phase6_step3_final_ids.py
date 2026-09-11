import pandas as pd
import time
import torch
import numpy as np
import torch.nn.functional as F
from collections import deque
import joblib

# ================== LOAD DATA ==================
df = pd.read_csv("C:/Users/ADMIN/Desktop/BTech/Codes/Phase 1/phase1_ev_attack_dataset.csv")

# Shuffle for realistic traffic
df = df.sample(frac=1).reset_index(drop=True)

features = ["request_rate", "packet_size", "time_gap"]

# ================== LOAD SCALER ==================
scaler = joblib.load("scaler.pkl")

# ================== LSTM MODEL ==================
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=1, num_classes=4):
        super(LSTMModel, self).__init__()
        
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# Initialize model
model = LSTMModel()

# Load trained weights
model.load_state_dict(torch.load("C:/Users/ADMIN/Desktop/BTech/Codes/Phase 3/lstm_model.pth"))

model.eval()

# ================== LABEL MAP ==================
labels_map = {
    0: "Normal",
    1: "DoS",
    2: "DDoS",
    3: "Malformed"
}

# ================== STREAM FUNCTION ==================
def stream_data(delay=0.5):
    for i in range(len(df)):
        sample = df.iloc[i][features].values
        yield sample
        time.sleep(delay)

# ================== SLIDING WINDOW ==================
window_size = 10
sequence = deque(maxlen=window_size)

# ================== ATTACK COUNTER ==================
attack_count = {
    "Normal": 0,
    "DoS": 0,
    "DDoS": 0,
    "Malformed": 0
}

# ================== REAL-TIME IDS ==================
stream = stream_data()

print("🚀 Real-Time Intrusion Detection System Started...\n")

for data in stream:

    # ================== APPLY SCALING ==================
    data_df = pd.DataFrame([data], columns=features)
    data_scaled = scaler.transform(data_df)[0]

    # Add dummy feature (to match 4 features)
    data_scaled = list(data_scaled)
    data_scaled.append(0)

    sequence.append(data_scaled)

    # Wait until sequence is full
    if len(sequence) < window_size:
        continue

    # Convert to tensor
    seq_array = np.array(sequence)
    seq_tensor = torch.tensor(seq_array, dtype=torch.float32).unsqueeze(0)

    # Prediction
    with torch.no_grad():
        output = model(seq_tensor)
        probs = F.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    label = labels_map[pred]
    attack_count[label] += 1

    # ================== ALERT SYSTEM ==================
    if label != "Normal":
        print(f"⚠️ ALERT: {label} detected | Confidence: {confidence:.2f}")
    else:
        print(f"✅ Normal traffic | Confidence: {confidence:.2f}")

    # ================== SUMMARY ==================
    total = sum(attack_count.values())
    if total % 20 == 0:
        print("\n📊 Detection Summary:")
        for k, v in attack_count.items():
            print(f"{k}: {v}")
        print("-" * 40)