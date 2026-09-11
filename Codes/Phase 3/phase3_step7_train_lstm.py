import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ==========================
# 1️⃣ Load sequences
# ==========================

X = np.load("X_sequences.npy")
y = np.load("y_sequences.npy")

print("Full sequence dataset:", X.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train),
                          batch_size=64,
                          shuffle=True)

test_loader = DataLoader(TensorDataset(X_test, y_test),
                         batch_size=64)

# ==========================
# 2️⃣ Define LSTM Model
# ==========================

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

model = LSTMClassifier(input_size=4,
                       hidden_size=64,
                       num_classes=4)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ==========================
# 3️⃣ Training Loop
# ==========================

EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

# ==========================
# 4️⃣ Evaluation
# ==========================

model.eval()
all_preds = []
all_true = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.numpy())
        all_true.extend(batch_y.numpy())

print("\n===== LSTM Results =====")
print("Accuracy:", accuracy_score(all_true, all_preds))
print(classification_report(all_true, all_preds))

torch.save(model.state_dict(), "lstm_model.pth")
print("✅ LSTM model trained and saved.")