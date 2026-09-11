import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# 1. Define LSTM Model (same as Phase 3)
# =========================
class LSTMModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=1, num_classes=4):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# =========================
# 2. Load Model Weights
# =========================
MODEL_PATH = "C:/Users/ADMIN/Desktop/BTech/Codes/Phase 3/lstm_model.pth"

model = LSTMModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

print("✅ Model loaded correctly")

# =========================
# 3. Load Test Data
# =========================
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

print("Test shape:", X_test.shape)

# =========================
# 4. Predictions
# =========================
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)

y_pred = predicted.numpy()

print("✅ Predictions done")

# =========================
# 5. Confusion Matrix
# =========================
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:\n", cm)

# =========================
# 6. Plot
# =========================
labels = ["Normal", "DoS", "DDoS", "Malformed"]

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels,
            yticklabels=labels)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("LSTM Confusion Matrix")
plt.show()

# =========================
# 7. Save Outputs
# =========================
np.save("lstm_predictions.npy", y_pred)
np.save("lstm_confusion_matrix.npy", cm)

print("✅ Outputs saved")