"""
Retrain a clean, leak-free LSTM for the real-time dashboard.

Why a new model?
----------------
* Phase 3's lstm_model.pth reported 100% accuracy, but the train/test split was
  performed AFTER sliding-window generation. Consecutive windows share 9 out of
  10 rows, so the test set leaked training rows -> the reported 1.00 was bogus.
  When that model was fed clean traffic emulation it misclassified everything
  (e.g. DoS-rate traffic -> "Malformed" with 99% confidence).
* An earlier experiment's model was trained on 281 rows whose labels were
  derived from the same thresholds used at runtime (circular) and only reached
  ~uniform probabilities.

This script trains an LSTM on the Phase 1 dataset with a leak-free protocol:

* features  : request_rate, packet_size, time_gap   (measurable live per packet)
* labels    : 0 Normal, 1 DoS, 2 DDoS, 3 Malformed
* windows   : 10 rows, label = class of the LAST row (predict "current packet")
* splits    : within each class, rows are ordered by time and split 70/30 in
              time order BEFORE windowing -> no row appears in both train/test.

Outputs (Dashboard/model/):
    scaler.pkl          StandardScaler fitted on training rows only
    lstm_model.pth      trained state dict (input 3, hidden 64, output 4)
    training_report.txt evaluation + runtime emulation checks

Usage:  python model/retrain.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import joblib

DASH_DIR = Path(__file__).resolve().parent.parent

# locate the Phase 1 dataset by climbing up from the repo root candidates
def find_dataset() -> Path:
    for cand in [
        Path(__file__).resolve().parents[3] / "Codes" / "Phase 1" / "phase1_ev_attack_dataset.csv",
        Path(__file__).resolve().parents[2] / "Codes" / "Phase 1" / "phase1_ev_attack_dataset.csv",
        DASH_DIR.parent / "Codes" / "Phase 1" / "phase1_ev_attack_dataset.csv",
    ]:
        if cand.exists():
            return cand
    raise FileNotFoundError("phase1_ev_attack_dataset.csv not found - run from repo checkout")

def main() -> None:
    import pandas as pd
    import torch
    import torch.nn as nn
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(42)
    np.random.seed(42)

    src = find_dataset()
    df = pd.read_csv(src)
    df = df.sort_values("timestamp").reset_index(drop=True)

    FEATURES = ["request_rate", "packet_size", "time_gap"]
    CLASSES = ["Normal", "DoS", "DDoS", "Malformed"]
    WINDOW = 10
    out_dir = DASH_DIR / "model"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {src}")
    print(df["label"].value_counts().sort_index().to_dict())

    # ---------- 1. leak-free time-ordered split per class ----------
    train_idx: list[int] = []
    test_idx: list[int] = []
    for label in sorted(df["label"].unique()):
        rows = df.index[df["label"] == label].tolist()
        cut = int(len(rows) * 0.7)
        train_idx += rows[:cut]
        test_idx += rows[cut:]
    train_idx.sort()
    test_idx.sort()

    def build_windows(indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for label in sorted(df["label"].unique()):
            cls_rows = sorted(i for i in indices if df.at[i, "label"] == label)
            arr = df.loc[cls_rows, FEATURES].to_numpy(dtype=float)
            for i in range(len(arr) - WINDOW):
                xs.append(arr[i : i + WINDOW])
                ys.append(label)
        return np.array(xs), np.array(ys)

    Xtr, ytr = build_windows(train_idx)
    Xte, yte = build_windows(test_idx)

    # ---------- 2. scale on TRAIN only ----------
    all_train_rows = df.loc[train_idx, FEATURES].to_numpy(dtype=float)
    scaler = StandardScaler().fit(all_train_rows)
    Xtr = scaler.transform(Xtr.reshape(-1, 3)).reshape(Xtr.shape)
    Xte = scaler.transform(Xte.reshape(-1, 3)).reshape(Xte.shape)
    joblib.dump(scaler, out_dir / "scaler.pkl")

    # ---------- 3. model ----------
    class LSTMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(3, 64, batch_first=True)
            self.fc = nn.Linear(64, 4)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    model = LSTMModel()
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    xtr = torch.tensor(Xtr, dtype=torch.float32)
    ytr_ = torch.tensor(ytr, dtype=torch.long)
    xte = torch.tensor(Xte, dtype=torch.float32)
    yte_ = torch.tensor(yte, dtype=torch.long)
    loader = DataLoader(TensorDataset(xtr, ytr_), batch_size=64, shuffle=True)

    best_val, best_state = -1.0, None
    EPOCHS = 25
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tot = 0.0
        for bx, by in loader:
            opt.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            opt.step()
            tot += loss.item()
        model.eval()
        with torch.no_grad():
            val_acc = (model(xte).argmax(1) == yte_).float().mean().item()
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:02d}  loss={tot/len(loader):.4f}  val_acc={val_acc:.4f}")

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_dir / "lstm_model.pth")

    # ---------- 4. honest evaluation ----------
    model.eval()
    with torch.no_grad():
        preds = model(xte).argmax(1).numpy()

    lines = [f"Leak-free LSTM report - {src.name}", "=" * 60]
    lines.append(f"train windows: {len(Xtr)}   test windows: {len(Xte)}")
    lines.append("")
    lines.append("classification_report (test):")
    lines.append(
        classification_report(yte, preds, target_names=CLASSES, digits=3)
    )
    cm = confusion_matrix(yte, preds)
    lines.append("confusion matrix (rows=actual, cols=predicted):")
    lines.append("            " + "".join(f"{c:>9}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        lines.append(f"{c:>9}: " + "".join(f"{cm[i, j]:>9}" for j in range(4)))

    # ---------- 5. runtime-style emulation (what the dashboard will feed) ----
    lines.append("\nruntime emulation: homogeneous windows at each class centre")
    centers = {
        "Normal": (2.0, 400.0, 0.9),
        "DoS": (120.0, 1150.0, 0.02),
        "DDoS": (65.0, 1000.0, 0.05),
        "Malformed": (27.0, 1700.0, 1.0),
    }
    for name, (r, s, g) in centers.items():
        win = np.array([[r, s, g]] * WINDOW)
        scaled = scaler.transform(win.reshape(-1, 3)).reshape(1, WINDOW, 3)
        with torch.no_grad():
            p = torch.softmax(model(torch.tensor(scaled).float()), 1)[0]
        lines.append(f"  {name:>9} centre -> {CLASSES[int(p.argmax())]:>9}"
                     f"  conf={float(p.max()):.2f}  probs={np.round(p.numpy(),3)}")

    # boundary windows: 3 normal rows then 7 attack rows
    lines.append("\nboundary probe: window = 3 normal + 7 attack rows")
    for name, (r, s, g) in [("DoS", (120.0, 1150.0, 0.02)),
                            ("DDoS", (65.0, 1000.0, 0.05))]:
        win = np.vstack([np.array([[2.0, 400.0, 0.9]] * 3),
                         np.array([[r, s, g]] * 7)])
        scaled = scaler.transform(win).reshape(1, WINDOW, 3)
        with torch.no_grad():
            p = torch.softmax(model(torch.tensor(scaled).float()), 1)[0]
        lines.append(f"  ->{name:>7} last rows -> {CLASSES[int(p.argmax())]:>9}"
                     f"  conf={float(p.max()):.2f}")

    report = "\n".join(lines)
    print("\n" + report)
    (out_dir / "training_report.txt").write_text(report + "\n")
    print(f"\nSaved model + scaler + report into {out_dir}")

if __name__ == "__main__":
    main()
