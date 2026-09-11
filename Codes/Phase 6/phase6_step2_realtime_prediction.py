import pandas as pd
import time
import torch
import numpy as np
from collections import deque

# ================== LOAD DATA ==================
df = pd.read_csv("C:/Users/ADMIN/Desktop/BTech/Codes/Phase 1/phase1_ev_attack_dataset.csv")

features = ["request_rate", "packet_size", "time_gap"]

# ================== LSTM MODEL ==================
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=1, num_classes=4):
        super(LSTMModel, self).__init__()
        
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # take last timestep
        out = self.fc(out)
        return out


# Initialize model
model = LSTMModel()

# Load trained weights (state_dict)
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

# ================== REAL-TIME DETECTION ==================
stream = stream_data()

print("🚀 Real-Time Detection Started...\n")

for data in stream:

    # Convert to list and add dummy feature (to match 4 features)
    data = list(data)
    data.append(0)

    sequence.append(data)

    # Wait until we have full sequence
    if len(sequence) < window_size:
        continue

    # Convert to tensor
    seq_array = np.array(sequence)
    seq_tensor = torch.tensor(seq_array, dtype=torch.float32).unsqueeze(0)

    # Prediction
    with torch.no_grad():
        output = model(seq_tensor)
        pred = torch.argmax(output, dim=1).item()

    # Print result
    print(f"[PREDICTION] {labels_map[pred]}")