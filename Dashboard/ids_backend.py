"""
Real-Time IDS Engine  (hybrid: LSTM ML + rule-based detection + stability)

Pipeline (matches the project architecture):
    packets -> per-packet features (request_rate, packet_size, time_gap)
            -> sliding window of 10 -> LSTM verdict + confidence
            -> 1-second rule pass (packet rate + source diversity)
            -> stability counters -> final status (Normal / DoS / DDoS)

The engine is UI-agnostic and thread-safe. Three traffic sources:
    * scenario : built-in scripted demo (Normal -> DoS -> Normal -> DDoS)
    * live     : scapy sniff() on a network interface (requires root)
    * replay   : offline pcap/pcapng replay

Model: Dashboard/model/lstm_model.pth + scaler.pkl (leak-free retrain of the
Phase-1 dataset). If the model files or torch are missing, the engine falls
back to rule-only detection and says so through info["ml"].
"""
from __future__ import annotations

import threading
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np

DASH_DIR = Path(__file__).resolve().parent
MODEL_DIR = DASH_DIR / "model"

CLASS_NAMES = ["Normal", "DoS", "DDoS", "Malformed"]
HISTORY_SECONDS = 600          # how much history the dashboard keeps
CONFIRM_TICKS = 3              # consecutive attack seconds before alarm
RECOVER_TICKS = 3              # consecutive clean seconds before clearing
ML_MIN_CONF = 0.50             # min confidence for the ML attack vote
EXTREME_FACTOR = 1.6           # pps >= threshold*factor -> rules alarm alone


def _load_ml():
    """Return (model, scaler, error_message). Either may be None."""
    try:
        import joblib  # noqa: F401
        import torch  # noqa: F401
    except Exception as exc:  # pragma: no cover - env dependent
        return None, None, f"torch/joblib not installed ({exc})"

    pth = MODEL_DIR / "lstm_model.pth"
    scf = MODEL_DIR / "scaler.pkl"
    if not (pth.exists() and scf.exists()):
        return None, None, "model files missing - run  python model/retrain.py"

    try:
        import torch
        import torch.nn as nn
        import joblib

        class LSTMModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(input_size=3, hidden_size=64,
                                    batch_first=True)
                self.fc = nn.Linear(64, 4)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])

        model = LSTMModel()
        model.load_state_dict(
            torch.load(pth, map_location="cpu", weights_only=True)
        )
        model.eval()
        return model, joblib.load(scf), None
    except Exception as exc:  # pragma: no cover - env dependent
        return None, None, f"failed to load model ({type(exc).__name__}: {exc})"


class IDSEngine:
    """Thread-safe real-time detection engine."""

    # scenario layout: (kind, seconds at 1x) — ~50 s full cycle
    SCENARIO = [
        ("normal", 8), ("dos", 12), ("normal", 8),
        ("ddos", 12), ("normal", 8),
    ]
    EV_IPS = ["192.168.50.11", "192.168.50.12", "192.168.50.13",
              "192.168.50.14", "192.168.50.15"]
    BOT_IPS = ["10.0.0.21", "10.0.0.22", "10.0.0.23", "10.0.0.24",
               "10.0.0.25", "10.0.0.26", "10.0.0.27", "10.0.0.28"]

    def __init__(self, clock=time.monotonic, sleeper=time.sleep):
        self._clock = clock
        self._sleep = sleeper
        self._lock = threading.RLock()

        self.model, self.scaler, self.ml_error = _load_ml()
        self.threshold_pps = 30.0

        # streaming state
        self.status = "Normal"
        self._atk_run = 0
        self._nrm_run = 0
        self._mixed_run = 0
        self._low_sec = 0                     # consecutive quiet seconds
        self._last_ts = None
        self._arrivals: deque = deque()       # arrival times of last ~1 s
        self._sizes: deque = deque()          # aligned with arrivals
        self._srcs: deque = deque()           # aligned with arrivals
        self._src_second: Counter = Counter()
        self._src_total: Counter = Counter()
        self._feat_window = deque(maxlen=10)  # (rate, size, gap) per packet
        self._raw_ml = deque(maxlen=3)        # recent LSTM labels
        self._last_ml = (0, 1.0)              # consensus (label_idx, conf)
        self._ml_total = [0, 0, 0, 0]

        # per-second history
        self.t_hist: list = []
        self.pps_hist: list = []
        self.ips_hist: list = []
        self.bps_hist: list = []
        self.status_hist: list = []
        self.ml_hist: list = []
        self.conf_hist: list = []
        self.regime_hist: list = []           # Normal / DoS / DDoS / Suspicious
        self.alerts: list = []

        self.total_packets = 0
        self.session_start = self._clock()

        self.mode = None
        self.mode_label = "idle"
        self.paused = False
        self.running = False
        self._stop_evt = threading.Event()
        self._threads: list = []
        self.info = {
            "ml": ("LSTM loaded" if self.model is not None
                   else f"rule-only ({self.ml_error})"),
            "source": "idle",
        }

    # ------------------------------------------------------------- lifecycle

    def start(self, mode: str, **params):
        """(Re)start the engine: scenario | live | replay."""
        with self._lock:
            self.running = False
            self._stop_evt.set()
            threads = list(self._threads)
            self._threads = []
        for th in threads:           # join outside the lock
            th.join(timeout=2)

        with self._lock:
            self.reset()
            self._stop_evt = threading.Event()
            self.paused = False
            self.running = True
            self.mode = mode
            self.info.pop("error", None)

        if mode == "scenario":
            speed = float(params.get("speed", 1.0))
            self.mode_label = f"Scenario demo ({speed:.1f}x)"
            worker = threading.Thread(
                target=self._run_scenario, args=(speed,), daemon=True
            )
        elif mode == "live":
            iface = params.get("iface")
            target = params.get("target", "127.0.0.1")
            self.mode_label = f"Live capture ({iface or 'auto'})"
            worker = threading.Thread(
                target=self._run_capture, args=(iface, target), daemon=True
            )
        elif mode == "replay":
            path = str(params.get("path", ""))
            speed = float(params.get("speed", 5.0))
            self.mode_label = f"Replay ({Path(path).name})"
            worker = threading.Thread(
                target=self._run_replay, args=(path, speed), daemon=True
            )
        else:
            self.running = False
            return

        self.info["source"] = self.mode_label
        ticker = threading.Thread(target=self._ticker_loop, daemon=True)
        self._threads = [worker, ticker]
        worker.start()
        ticker.start()

    def stop(self):
        with self._lock:
            self.running = False
            self._stop_evt.set()
            threads = list(self._threads)
            self._threads = []
        for th in threads:
            th.join(timeout=2)

    def pause(self, paused: bool):
        with self._lock:
            self.paused = paused

    def reset(self):
        with self._lock:
            self._atk_run = 0
            self._nrm_run = 0
            self._mixed_run = 0
            self._low_sec = 0
            self._last_ts = None
            self._arrivals.clear()
            self._sizes.clear()
            self._srcs.clear()
            self._src_second.clear()
            self._src_total.clear()
            self._feat_window.clear()
            self._raw_ml.clear()
            self._last_ml = (0, 1.0)
            self.status = "Normal"
            for lst in (self.t_hist, self.pps_hist, self.ips_hist,
                        self.bps_hist, self.status_hist, self.ml_hist,
                        self.conf_hist, self.regime_hist, self.alerts):
                lst.clear()
            self.total_packets = 0
            self.session_start = self._clock()

    # -------------------------------------------------------- packet feed

    def feed_packet(self, src_ip: str, size: float):
        """Ingest one packet (per-packet features are computed here)."""
        now = self._clock()
        with self._lock:
            if not self.running or self.paused:
                return
            gap = 0.0 if self._last_ts is None else now - self._last_ts
            self._last_ts = now

            self._arrivals.append(now)
            self._sizes.append(size)
            self._srcs.append(src_ip)
            self._src_second[src_ip] += 1
            self._src_total[src_ip] += 1
            self.total_packets += 1

            rate, _ips, _bps = self._measure(now)
            # request_rate feature = measured packets/s at this instant
            self._feat_window.append((float(rate), float(size), float(gap)))

    def _measure(self, now: float):
        """Prune the ~1 s window; return (packets/s, unique srcs, bytes/s)."""
        cutoff = now - 1.0
        while self._arrivals and self._arrivals[0] < cutoff:
            self._arrivals.popleft()
            self._sizes.popleft()
            src = self._srcs.popleft()
            self._src_second[src] -= 1
            if self._src_second[src] <= 0:
                del self._src_second[src]
        return (len(self._arrivals), len(self._src_second),
                sum(self._sizes))

    # ------------------------------------------------------- detection tick

    def tick(self):
        """One detection cycle, normally invoked once per second."""
        now = self._clock()
        with self._lock:
            if not self.running or self.paused:
                return
            rate, ips, bps = self._measure(now)

            # ---------- rule vote ---------------------------------------
            if rate >= self.threshold_pps:
                rule_type = "DoS" if ips <= 2 else "DDoS"
            else:
                rule_type = "Normal"

            # ---------- quiet cooldown ----------------------------------
            # Once traffic clearly drops below the threshold, forget the
            # ML window (it still holds attack packets) so recovery is fast.
            if rule_type == "Normal" and rate <= self.threshold_pps * 0.5:
                self._low_sec += 1
            else:
                self._low_sec = 0
            if self._low_sec >= 1:
                self._feat_window.clear()
                self._raw_ml.clear()
                self._last_ml = (0, 1.0)

            # ---------- ML vote on the current 10-packet window ----------
            ml_idx, ml_conf = self._last_ml
            if self.model is not None and len(self._feat_window) == 10:
                ml_idx, ml_conf = self._ml_vote()

            ml_attack = (self.model is not None and ml_idx != 0
                         and ml_conf >= ML_MIN_CONF)

            # ---------- hybrid reading -----------------------------------
            extreme = rate >= self.threshold_pps * EXTREME_FACTOR
            if rule_type != "Normal" and (extreme or ml_attack):
                reading = rule_type
            elif rule_type == "Normal" and not ml_attack:
                reading = "Normal"
            else:
                reading = "Mixed"   # exactly one signal fires

            # ---------- stability counters -------------------------------
            if reading in ("DoS", "DDoS"):
                self._atk_run += 1
                self._nrm_run = 0
                self._mixed_run = 0
            elif reading == "Normal":
                self._nrm_run += 1
                self._atk_run = 0
                self._mixed_run = 0
            else:
                self._mixed_run += 1

            prev = self.status
            if prev == "Normal" and self._atk_run >= CONFIRM_TICKS:
                self.status = reading
                self._fire_alert(reading, rate, ips,
                                 CLASS_NAMES[ml_idx], ml_conf)
                self._atk_run = 0
            elif prev != "Normal" and self._nrm_run >= RECOVER_TICKS:
                self.status = "Normal"
                self._fire_alert("Normal", rate, ips,
                                 CLASS_NAMES[ml_idx], ml_conf)

            # regime chip (adds ML-only "Suspicious" state for the UI)
            regime = self.status
            if (self.status == "Normal" and ml_attack
                    and rule_type == "Normal" and self._mixed_run >= 2):
                regime = "Suspicious"

            # ---------- history (one row per second) ---------------------
            self.t_hist.append(now - self.session_start)
            self.pps_hist.append(rate)
            self.ips_hist.append(ips)
            self.bps_hist.append(bps)
            self.status_hist.append(self.status)
            self.ml_hist.append(CLASS_NAMES[ml_idx])
            self.conf_hist.append(round(ml_conf, 3))
            self.regime_hist.append(regime)
            self._trim()

    def _ml_vote(self):
        """Run the LSTM over the window; return (consensus_label, conf)."""
        try:
            import torch
            arr = np.asarray(self._feat_window, dtype=float).reshape(1, 10, 3)
            scaled = self.scaler.transform(
                arr.reshape(-1, 3)
            ).reshape(1, 10, 3)
            with torch.no_grad():
                probs = torch.softmax(
                    self.model(torch.tensor(scaled, dtype=torch.float32)),
                    dim=1,
                )[0].numpy()
            raw = int(probs.argmax())
            self._raw_ml.append(raw)
            # stability filter: majority of the last few windows
            counts = Counter(self._raw_ml)
            last_seen = {i: None for i in range(4)}
            for pos, val in enumerate(reversed(self._raw_ml)):
                if last_seen[val] is None:
                    last_seen[val] = pos
            cons = max(
                range(4),
                key=lambda i: (counts.get(i, 0),
                               -1 if last_seen[i] is None else -last_seen[i]),
            )
            self._ml_total[cons] += 1
            self._last_ml = (cons, float(probs[cons]))
            return self._last_ml
        except Exception:
            return self._last_ml  # keep previous vote on any failure

    def _ticker_loop(self):
        while not self._stop_evt.is_set():
            self._sleep(1.0)
            self.tick()

    # ------------------------------------------------------------- alerts

    def _fire_alert(self, status: str, rate: float, ips: int,
                    ml_name: str, ml_conf: float):
        secs = self._clock() - self.session_start
        if status == "DoS":
            msg = (f"⚠️ DoS DETECTED — {rate:.0f} pkt/s from {ips} source"
                   f"{'s' if ips != 1 else ''}  (ML: {ml_name} {ml_conf:.0%})")
            level = "warning"
        elif status == "DDoS":
            msg = (f"🚨 DDoS DETECTED — {rate:.0f} pkt/s from {ips} sources"
                   f"  (ML: {ml_name} {ml_conf:.0%})")
            level = "critical"
        else:
            msg = "✅ Traffic returned to Normal"
            level = "info"
        self.alerts.append({
            "t": round(secs, 1), "level": level,
            "status": status, "msg": msg,
        })
        if len(self.alerts) > 200:
            del self.alerts[: len(self.alerts) - 200]

    def _trim(self):
        if len(self.t_hist) > HISTORY_SECONDS:
            self.t_hist = self.t_hist[-HISTORY_SECONDS:]
            self.pps_hist = self.pps_hist[-HISTORY_SECONDS:]
            self.ips_hist = self.ips_hist[-HISTORY_SECONDS:]
            self.bps_hist = self.bps_hist[-HISTORY_SECONDS:]
            self.status_hist = self.status_hist[-HISTORY_SECONDS:]
            self.ml_hist = self.ml_hist[-HISTORY_SECONDS:]
            self.conf_hist = self.conf_hist[-HISTORY_SECONDS:]
            self.regime_hist = self.regime_hist[-HISTORY_SECONDS:]

    # ------------------------------------------------------- traffic sources

    def _scenario_gen(self, kind: str, rng: np.random.Generator):
        """Yield (src_ip, size); sleeps internally to pace the stream."""
        if kind == "normal":
            while True:
                self._sleep(rng.exponential(0.4) + 0.05)   # ~2.5 pkt/s
                yield rng.choice(self.EV_IPS), int(rng.normal(400, 80))
        elif kind == "dos":   # one EVCS, ~130 pkt/s of 800-1500 B messages
            while True:
                self._sleep(max(rng.exponential(1 / 130.0), 0.001))
                yield "192.168.50.66", int(np.clip(rng.normal(1100, 150),
                                                    800, 1500))
        else:  # ddos: ~75 pkt/s aggregate across 8 bot source IPs
            while True:
                self._sleep(max(rng.exponential(1 / 75.0), 0.001))
                yield rng.choice(self.BOT_IPS), int(np.clip(rng.normal(1000, 200),
                                                            600, 1400))

    def _run_scenario(self, speed: float):
        rng = np.random.default_rng(2024)
        while not self._stop_evt.is_set():
            for kind, seconds in self.SCENARIO:
                seg_start = self._clock()
                gen = self._scenario_gen(kind, rng)
                while (self._clock() - seg_start) < seconds / speed:
                    src, size = next(gen)
                    self.feed_packet(src, size)
            self._sleep(1.0)   # small gap between scenario loops

    def _run_capture(self, iface, target):
        try:
            from scapy.all import IP, sniff
        except Exception as exc:  # pragma: no cover
            self.info["error"] = f"scapy unavailable: {exc}"
            return

        def _cb(pkt):
            if pkt.haslayer(IP) and len(pkt) > 0:
                self.feed_packet(pkt[IP].src, len(pkt))

        try:
            sniff(iface=iface or None, prn=_cb, store=False,
                  filter=f"host {target}")
        except Exception as exc:
            self.info["error"] = (
                f"capture failed on {iface or 'auto'}: {exc} — "
                "on macOS start Streamlit with sudo for live capture"
            )

    def _run_replay(self, path, speed):
        try:
            from scapy.all import IP, rdpcap
        except Exception as exc:  # pragma: no cover
            self.info["error"] = f"scapy unavailable: {exc}"
            return
        try:
            pkts = rdpcap(path)
        except Exception as exc:
            self.info["error"] = f"could not read {path}: {exc}"
            return
        prev = 0.0
        for pkt in pkts:
            if self._stop_evt.is_set():
                break
            if not pkt.haslayer(IP) or len(pkt) <= 0:
                continue
            self.feed_packet(pkt[IP].src, len(pkt))
            ts = float(pkt.time or 0.0)
            if prev and ts > prev:
                self._sleep(min((ts - prev) / speed, 0.02))
            prev = ts
        self.info["source"] = f"{self.info.get('source', 'replay')} — finished"

    # ------------------------------------------------------------- snapshot

    def snapshot(self):
        """Consistent copy of everything the UI needs."""
        with self._lock:
            rate, ips, bps = self._measure(self._clock())
            ml_idx, ml_conf = self._last_ml

            # agreement: does the ML vote (attack vs normal) match the final
            # status over the last 20 seconds?
            st = self.status_hist[-20:]
            ml = self.ml_hist[-20:]
            if st:
                agree = round(100.0 * sum(
                    1 for a, b in zip(st, ml)
                    if (a == "Normal") == (b == "Normal")
                ) / len(st))
            else:
                agree = 100

            return {
                "status": self.status,
                "regime": self.regime_hist[-1] if self.regime_hist else "Normal",
                "pps": rate,
                "ips": ips,
                "bps": bps,
                "ml_label": CLASS_NAMES[ml_idx],
                "ml_conf": round(ml_conf, 3),
                "model_present": self.model is not None,
                "ml_error": self.ml_error,
                "agreement": agree,
                "total_packets": self.total_packets,
                "uptime": self._clock() - self.session_start,
                "threshold": self.threshold_pps,
                "mode": self.mode,
                "mode_label": self.mode_label,
                "paused": self.paused,
                "info": dict(self.info),
                "history": {
                    "t": list(self.t_hist),
                    "pps": list(self.pps_hist),
                    "ips": list(self.ips_hist),
                    "bps": list(self.bps_hist),
                    "status": list(self.status_hist),
                    "regime": list(self.regime_hist),
                    "ml": list(self.ml_hist),
                    "conf": list(self.conf_hist),
                },
                "top_sources": self._src_total.most_common(10),
                "ml_total": list(self._ml_total),
                "alerts": list(self.alerts),
            }


# ---------------------------------------------------------------------------
# process-wide singleton (survives Streamlit reruns)
# ---------------------------------------------------------------------------

_ENGINE = None
_ENGINE_LOCK = threading.Lock()


def get_engine():
    """Return the process-wide engine instance."""
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = IDSEngine()
        return _ENGINE


def interface_choices():
    """Best-effort list of (iface_name, ip) for the live-capture selector."""
    try:
        from scapy.all import get_if_addr, get_if_list
    except Exception:
        return []
    out = []
    for iface in get_if_list():
        try:
            ip = get_if_addr(iface)
        except Exception:
            ip = "0.0.0.0"
        if ip and ip != "0.0.0.0" and not ip.startswith("169."):
            out.append((iface, ip))
    return out
