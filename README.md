# 🚗 EV-Charging Network Intrusion Detection System (IDS)

**Final Year Project** — Real-time detection of **DoS** and **DDoS** attacks on
Electric Vehicle (EV) charging communication networks using **Machine Learning
(LSTM)** combined with **rule-based detection**, visualised on a **live
Streamlit dashboard**.

---

## 1️⃣ Status Update for Project Teacher / Guide

> Below is a ready-to-send summary of where the project stands. You can copy
> and paste it to your project guide.

---

**Dear Sir/Ma'am,**

My final year project is an **Intrusion Detection System (IDS) for Electric
Vehicle charging networks** that detects cyber-attacks such as **DoS (Denial
of Service)** and **DDoS (Distributed Denial of Service)** on EV charging
communication.

As discussed, I am building this project in a **step-by-step (phased) manner**.
The complete project is **divided into 6 parts**, and it is **not yet
completed** — I am currently working on **Part 2**. The division is as follows:

| Part | Phase | Work covered | Status |
|------|-------|--------------|--------|
| Part 1 | **Phase 1** | EV network **dataset generation** — normal EV traffic + simulated DoS, DDoS & malformed attacks | ✅ Completed |
| **Part 2** | **Phase 2** | **Data cleaning, inspection & feature engineering** of the Phase-1 dataset, scaling and train/test split | 🔄 **In progress (currently working on this)** |
| Part 3 | **Phase 3** | Building the ML detection models — **LSTM** (sequence model) with baseline comparison (KNN, SVM) | ⏳ Pending |
| Part 4 | **Phase 4** | **Explainable AI** (SHAP / LIME) to explain why the model flags traffic as an attack | ⏳ Pending |
| Part 5 | **Phase 5** | **Model evaluation & comparison** — confusion matrix, precision/recall/F1, model selection | ⏳ Pending |
| Part 6 | **Phase 6** | **Real-time IDS** — live packet capture, detection engine, and a **dashboard** to view attacks | ⏳ Pending |

I am following this structure so that each part can be reviewed and validated
before moving to the next. I will share the outputs of **Part 2** with you as
soon as it is complete.

Thank you.

---

*[Your Name]  ·  [Roll No]  ·  [Class / Batch]  ·  [Guide Name]*

---

## 2️⃣ Project Overview — What the System Does

Electric vehicles and charging stations communicate over a network. An
attacker can flood this communication with a huge number of requests, making
the charging network slow or completely unavailable (DoS), or send the flood
from many different sources at the same time (DDoS).

This project builds an **automatic detection system** that:

1. Watches network traffic continuously.
2. Extracts meaningful **features** from every packet.
3. Uses an **LSTM neural network** + **rule engine** to decide whether the
   traffic is *Normal*, *DoS* or *DDoS*.
4. Raises an **alert** and shows everything on a **live dashboard**.

## 3️⃣ How the Project is Divided (6 Parts)

The project is intentionally split into 6 parts so progress can be reviewed
stage by stage:

| Phase | Purpose | Main outputs / folder |
|-------|---------|-----------------------|
| **Phase 1** | Generate a labelled EV-charging traffic dataset: normal traffic from many EV chargers + scripted DoS, DDoS and malformed attack traffic | `Codes/Phase 1/` → `phase1_ev_attack_dataset.csv` |
| **Phase 2** | Inspect, clean and engineer features from the raw dataset; scale features; create train/test splits | `Codes/Phase 2/` → cleaned + feature-engineered CSVs, `X_train/X_test/y_train/y_test` |
| **Phase 3** | Train detection models on sequential windows of traffic: **LSTM** (primary) plus KNN / SVM baselines | `Codes/Phase 3/` → trained `.pth` / `.pkl` models |
| **Phase 4** | Explainable AI — SHAP / LIME explanations of model decisions | `Codes/Phase 4/` |
| **Phase 5** | Evaluate and compare models (confusion matrices, precision / recall / F1) | `Codes/Phase 5/` → reports & comparison CSV |
| **Phase 6** | Real-time pipeline: live packet capture → feature extraction → model prediction → alerts + dashboard | `Codes/Phase 6/`, `Dashboard/` |

> The written phase reports (`.docx`) are kept in the **`Phases/`** folder.

## 4️⃣ Complete Working — System Architecture

```
EV traffic (normal + attacks)
        │  (packets over the network)
        ▼
┌─────────────────────┐
│  1. Traffic Source  │   Scenario demo │ live scapy capture │ pcap replay
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  2. Feature Extract │   per packet: request_rate, packet_size, time_gap
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  3. LSTM Model      │   sliding window of 10 packets → 4-class prediction
│     (ML engine)     │   (Normal / DoS / DDoS / Malformed) + confidence
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  4. Rule Engine     │   1-second packet rate + unique source IP count
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  5. Hybrid Decision │   ML vote + rule vote + stability counters
│     + Stability     │   (attack must persist several seconds)
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  6. Alert & Status  │   Normal / DoS / DDoS  +  alert feed
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  7. Dashboard       │   live charts: packet rate, IP count, timeline,
└─────────────────────┘   ML confidence, alerts, top sources
```

### 4.1 How an attack is detected (step by step)

1. **Capture** — every IP packet is captured (demo scenario, live network, or
   a saved `.pcap` file).
2. **Feature extraction** — for each packet we compute:
   - `request_rate` — packets observed per second (instantaneous rate),
   - `packet_size` — size of the packet in bytes,
   - `time_gap` — time since the previous packet.
3. **ML prediction** — the last **10 packets** form a sliding window that is
   passed to the trained **LSTM**. The LSTM outputs one of
   **Normal / DoS / DDoS / Malformed** with a confidence score. A majority
   (stability) filter smooths the verdict.
4. **Rule check** — every second the engine measures the packet rate and the
   number of **unique source IPs**:
   - rate ≥ threshold **and** ≤ 2 sources → *DoS*,
   - rate ≥ threshold **and** > 2 sources → *DDoS*.
5. **Hybrid decision** — an attack is confirmed when the rules fire **and**
   either the LSTM agrees or the rate is clearly extreme. This reduces false
   alarms. The final **DoS vs DDoS** label uses source-IP diversity (a model
   cannot "see" IP counts — the project design is *ML detects, IP count
   classifies*).
6. **Stability logic** — the alarm is only raised after the suspicious state
   persists for a few consecutive seconds, and cleared only after traffic
   stays normal for a few seconds (avoids flickering alerts).
7. **Alert + dashboard** — transitions push coloured alerts
   (`⚠️ DoS`, `🚨 DDoS`, `✅ returned to Normal`) and the dashboard updates
   live every second.

### 4.2 The ML model

- Trained on the Phase-1 dataset (leak-free evaluation — see
  `Dashboard/model/training_report.txt`).
- **Architecture:** LSTM, input features = 3, hidden size = 64, output = 4
  classes, sliding window of 10 packets.
- **Result:** ~0.999 test accuracy across Normal / DoS / DDoS / Malformed
  (runtime class-centre emulation ≈ 100% confidence).
- Retrain script: `Dashboard/model/retrain.py`.

### 4.3 The dashboard (Phase 6 visual output)

The Streamlit dashboard shows, refreshed every second:

- 🟢/🟠/🔴 **status banner** (Normal / DoS / DDoS),
- metric cards: packet rate, unique IPs, throughput, total packets, ML
  verdict, ML↔status agreement,
- **packet-rate chart** with coloured bands marking detected attacks,
- unique source-IP chart,
- **LSTM confidence** over time, coloured by predicted class,
- a **session attack timeline**, LSTM class-vote donut and top source IPs,
- a colour-coded **alert feed**.

## 5️⃣ How to Run the Code

> **Environment:** use the Anaconda Python which has all libraries
> (`/opt/anaconda3/bin/python` on this machine — streamlit, torch, scapy,
> sklearn, plotly, pandas). Dependencies are listed in
> `Dashboard/requirements.txt`.

### Option A — Quick demo (no root, no extra terminals) — recommended

```bash
cd Dashboard
/opt/anaconda3/bin/python -m streamlit run dashboard.py
```

The dashboard auto-starts the **Scenario demo**, which loops through a
scripted attack timeline every ~50 seconds:

```
Normal 8 s → ⚠️ DoS 12 s → Normal 8 s → 🚨 DDoS 12 s → Normal 8 s
```

Watch the status pill, the alert feed and the coloured bands appear live.
This mode needs no special permissions and is perfect for demonstrating the
full pipeline.

### Option B — Run the automated tests

```bash
cd Dashboard
/opt/anaconda3/bin/python selftest.py   # full detection lifecycle (Normal→DoS→Normal→DDoS→Normal)
/opt/anaconda3/bin/python ui_test.py    # headless dashboard UI test
```

Both print ✅ per check and exit `0` when everything passes.

### Option C — Live capture demo (attacks a real local server)

Uses three terminals. Packet capture needs **root on macOS** (Windows with
Npcap usually works without it).

**Terminal A — dashboard (live capture):**
```bash
cd Dashboard
sudo /opt/anaconda3/bin/python -m streamlit run dashboard.py
# sidebar → Traffic source = "Live capture" → Apply source
```

**Terminal B — the target web server:**
```bash
/opt/anaconda3/bin/python -m http.server 8000 --bind 127.0.0.1
```

**Terminal C — launch the attacks:**
```bash
cd Dashboard
# single source = DoS
/opt/anaconda3/bin/python dos_generator.py --mode dos --pps 130 --payload 1000 --duration 15
# many sources (127.0.0.1 - 127.0.0.8) = DDoS
/opt/anaconda3/bin/python dos_generator.py --mode ddos --pps 80 --sources 8 --payload 1000 --duration 15
```

### Option D — Replay a saved capture

In the dashboard sidebar choose **Traffic source → "Replay pcap"**, point it at
any `.pcap` / `.pcapng` file (e.g. the bundled `traffic_capture.pcapng`) and
press **Apply source**.

### Retraining the model (Phase 3 work)

```bash
cd Dashboard/model
/opt/anaconda3/bin/python retrain.py    # writes lstm_model.pth + scaler.pkl + report
```

### Running the individual phase scripts

Every phase under `Codes/Phase 1 … 6` has numbered `step` / `phase` scripts
that run top-to-bottom (1 → 2 → 3 …) to reproduce the pipeline.

⚠️ **Note:** the Phase 2–6 scripts were written earlier with machine-specific
Windows paths (e.g. `C:/Users/ADMIN/Desktop/BTech/...`) hard-coded inside
them. Before running any phase script on a different machine, replace those
paths at the top of the script with the location of this project on your
computer, then run the script from its own folder, e.g.:

```bash
cd "Codes/Phase 2"
/opt/anaconda3/bin/python phase2_step1_data_inspection.py
```

The Phase-1 generators and everything in `Dashboard/` are already
path-independent and run anywhere.

## 6️⃣ Folder Structure

```
EV-Project-main/
├── README.md                     ← this file
├── RUN.txt                       ← quick-run cheat sheet
├── traffic_capture.pcapng        ← sample real capture (for replay mode)
├── IDS_Project_Explanation.txt   ← short system explanation
│
├── Codes/                        ← code for the 6 phases
│   ├── Phase 1/                  ← dataset generation (normal + attacks)
│   ├── Phase 2/                  ← cleaning, feature engineering, split
│   ├── Phase 3/                  ← model training (LSTM, KNN, SVM)
│   ├── Phase 4/                  ← explainable AI (SHAP / LIME)
│   ├── Phase 5/                  ← evaluation & model comparison
│   └── Phase 6/                  ← real-time data stream & prediction
│
├── Phases/                       ← written phase reports (.docx)
│
└── Dashboard/                    ← final real-time system + UI
    ├── dashboard.py              ← Streamlit dashboard (live charts/alerts)
    ├── ids_backend.py            ← detection engine (ML + rules + stability)
    ├── dos_generator.py          ← DoS / DDoS attack generator (for demos)
    ├── selftest.py               ← automated engine test
    ├── ui_test.py                ← automated UI test
    ├── model/                    ← retrained LSTM + scaler + training report
    └── README.md                 ← dashboard-specific documentation
```

## 7️⃣ Tools & Libraries

| Tool | Used for |
|------|----------|
| Python 3 | main language |
| Scapy | packet capture / network traffic handling |
| Pandas / NumPy | dataset processing |
| scikit-learn | preprocessing, scaling, KNN / SVM baselines, metrics |
| PyTorch | LSTM model (train + real-time inference) |
| Streamlit | live dashboard |
| Plotly | interactive charts |
| Joblib | saving scaler / models |
| Wireshark (optional) | visual packet inspection during capture |
