# Real-Time IDS Dashboard — EV Charging Network

Hybrid Intrusion Detection System for EV charging communication.
Detects **DoS** and **DDoS** floods in real time with an **LSTM model +
rule engine + stability logic**, and visualises everything on a live
Streamlit dashboard.

```
packets ──► features (request_rate, packet_size, time_gap)
         ──► sliding window of 10 ──► LSTM verdict + confidence
         ──► 1-second rule pass (packet rate + source diversity)
         ──► stability counters ──► Normal / DoS / DDoS  ──► dashboard
```

---

## Quick start

```bash
# everything below uses the Anaconda Python
/opt/anaconda3/bin/python -m streamlit run dashboard.py
```

The sidebar defaults to **Scenario demo**: a scripted
`Normal → DoS → Normal → DDoS → Normal` cycle (~50 s). No root, no extra
terminals, deterministic — ideal for exams/defence demos.

| Traffic source | What it does | Root needed? |
|---|---|---|
| Scenario demo | built-in scripted attacks | no |
| Live capture | scapy `sniff()` on a chosen interface | macOS: yes (sudo) |
| Replay pcap | replays an offline `.pcap`/`.pcapng` | no |

### Live demo (three terminals)

```bash
# A — dashboard (macOS: sudo needed for capture)
sudo /opt/anaconda3/bin/python -m streamlit run dashboard.py
#    sidebar -> "Live capture" -> Apply source

# B — victim web server
/opt/anaconda3/bin/python -m http.server 8000 --bind 127.0.0.1

# C — attacker (single source = DoS, many sources = DDoS)
/opt/anaconda3/bin/python dos_generator.py --mode dos --pps 130 --payload 1000 --duration 15
/opt/anaconda3/bin/python dos_generator.py --mode ddos --pps 80 --sources 8 --payload 1000 --duration 15
```

> `--payload 1000` sends ~1 KB messages so live traffic matches the
> EV-charging packet sizes the model was trained on. `--payload 0` = tiny HTTP GETs.

---

## Tests

| Command | Verifies |
|---|---|
| `python selftest.py` | full detection lifecycle with a fake clock: Normal → DoS → Normal → DDoS → Normal, plus alert log |
| `python ui_test.py` | dashboard renders headlessly (Streamlit AppTest), sidebar mode switching works, live-capture error is handled gracefully |

Both exit 0 on success.

---

## Model

`model/lstm_model.pth` + `model/scaler.pkl` are a **leak-free retrain** of the
Phase-1 dataset (`Codes/Phase 1/phase1_ev_attack_dataset.csv`).

Why a new model was needed:

* Phase 3's `lstm_model.pth` reported **100% accuracy** but its train/test
  split was performed *after* sliding-window creation — neighbouring windows
  share 9/10 rows, so the test set leaked training rows. Fed with clean
  traffic emulation it misclassified everything (DoS → "Malformed" @ 99%).
* An earlier experiment's model (trained on 281 self-labelled rows) had
  outputs barely better than a coin flip.

The retrained model (window 10, features `request_rate/packet_size/time_gap`,
classes Normal/DoS/DDoS/Malformed) is evaluated leak-free: classes are split
in time order **before** windowing.

* test accuracy **0.999** (see `model/training_report.txt`)
* runtime emulation of each class centre: correct, confidence ≈ 1.00

Retrain any time with:

```bash
cd Dashboard/model
/opt/anaconda3/bin/python retrain.py
```

### Why the engine is "hybrid" and not "pure ML"

Anomaly traffic is raised when the **rules confirm a flood AND either the
LSTM agrees or the rate is clearly extreme** (threshold × 1.6). The final
`DoS` vs `DDoS` label comes from **source diversity** (≤2 sources = DoS,
more = DDoS) because a model fed only rate/size/gap cannot see IP counts —
this matches the project write-up exactly ("ML detects, IP count classifies").

Every second the dashboard shows both signals independently (model verdict +
confidence, ML↔status agreement), so the ML contribution stays visible and
auditable during a demo.

---

## Tuning

| Control | Where | Default |
|---|---|---|
| Attack threshold | sidebar slider (pkt/s) | 30 |
| Confirm / recovery | `ids_backend.py` `CONFIRM_TICKS` / `RECOVER_TICKS` | 3 / 3 |
| Demo speed | sidebar (×, applies on Apply) | 1.0 |
| ML vote bar | `ids_backend.py` `ML_MIN_CONF` | 0.50 |

Tune the threshold down for quiet lab networks or up in noisy ones.

---

## Files

```
Dashboard/
├── dashboard.py        # Streamlit UI (metrics, charts, alerts, controls)
├── ids_backend.py      # detection engine (ML + rules + stability, sources)
├── dos_generator.py    # paced DoS/DDoS attacker (127.0.0.x multi-source)
├── selftest.py         # deterministic lifecycle test
├── ui_test.py          # headless UI test (AppTest)
├── model/
│   ├── retrain.py          # leak-free LSTM retraining script
│   ├── lstm_model.pth      # trained weights (3 features → 4 classes)
│   ├── scaler.pkl          # StandardScaler
│   └── training_report.txt # evaluation + runtime emulation report
└── requirements.txt    # python dependencies
```

Dependencies (all present in `/opt/anaconda3`): streamlit ≥ 1.37, plotly,
pandas, numpy, scikit-learn, joblib, torch, scapy.

---

## Troubleshooting

* **Live capture does nothing / "Permission denied: /dev/bpf" (macOS)** —
  packet capture needs root: start Streamlit with `sudo`, or use Scenario /
  Replay modes which need no capture privileges.
* **Dashboard shows only Normal on a busy network** — raise the attack
  intensity (`--pps 200+`) or lower the sidebar threshold.
* **Model says "rule-only"** — model files are missing; run
  `model/retrain.py`.
* **Attacks not detected with tiny HTTP GET floods** — payloads of a few
  hundred bytes sit outside the model's training distribution; add
  `--payload 1000`.
