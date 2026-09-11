"""
Headless engine selftest — drives the full detection lifecycle with a fake
clock so it runs in milliseconds and is fully deterministic.

Verifies:
    1. Normal traffic stays Normal (rules + ML agree, no false alarm).
    2. A single-source flood is raised as DoS within the expected time.
    3. Traffic returns to Normal after the flood stops.
    4. A multi-source flood is raised as DDoS.
    5. Alerts are generated at each transition.
    6. The LSTM model is loaded and voting.

Usage:  python selftest.py     (exit code 0 = all checks passed)
"""
from __future__ import annotations

import sys
import threading

import numpy as np

from ids_backend import IDSEngine, CLASS_NAMES


class FakeClock:
    def __init__(self, start=0.0):
        self.t = start

    def __call__(self):
        return self.t


class FakeSleeper:
    def __call__(self, _secs):
        return None


def sim_seconds(engine, clock, seconds, pps, srcs, size_mu, size_sigma, seed=7):
    """Feed `seconds` of traffic at `pps` pkt/s; tick once per simulated s."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / pps
    for _ in range(int(seconds)):
        for _ in range(int(pps)):
            clock.t += dt
            src = (srcs[int(rng.integers(0, len(srcs)))]
                   if isinstance(srcs, list) else srcs)
            engine.feed_packet(src, max(40.0, rng.normal(size_mu, size_sigma)))
        clock.t = round(clock.t, 6)
        engine.tick()


def status_of(engine):
    snap = engine.snapshot()
    return snap["status"], snap["ml_label"], snap["pps"]


def main() -> int:
    clock = FakeClock(0.0)
    engine = IDSEngine(clock=clock, sleeper=FakeSleeper())
    engine.running = True
    engine.threshold_pps = 30.0

    if engine.model is None:
        print("❌ model not loaded:", engine.ml_error)
        return 1
    print("✅ LSTM model + scaler loaded")

    checks = []

    # ---- phase 1: normal EV traffic (3 EV chargers, ~3 pkt/s) ----------
    evs = ["192.168.50.11", "192.168.50.12", "192.168.50.13"]
    for _ in range(20):
        sim_seconds(engine, clock, 1, 3, evs, 400, 80)
    st, ml, pps = status_of(engine)
    ok = st == "Normal" and ml == "Normal"
    checks.append(("normal traffic stays Normal "
                   f"(status={st}, ml={ml})", ok))

    # ---- phase 2: DoS flood, one attacker -------------------------------
    for _ in range(8):
        sim_seconds(engine, clock, 1, 130, "192.168.50.66", 120, 30)
    st, ml, pps = status_of(engine)
    ok = st == "DoS"
    checks.append((f"single-source flood -> DoS (status={st}, "
                   f"ml={ml}, {pps:.0f} pkt/s)", ok))

    # ---- phase 3: back to normal ----------------------------------------
    for _ in range(10):
        sim_seconds(engine, clock, 1, 3, evs, 400, 80)
    st, ml, pps = status_of(engine)
    ok = st == "Normal"
    checks.append((f"traffic clears after flood (status={st})", ok))

    # ---- phase 4: DDoS flood, eight bot sources -------------------------
    bots = [f"10.0.0.{21 + i}" for i in range(8)]
    for _ in range(8):
        sim_seconds(engine, clock, 1, 80, bots, 1000, 200)
    st, ml, pps = status_of(engine)
    ok = st == "DDoS"
    checks.append((f"multi-source flood -> DDoS (status={st}, "
                   f"ml={ml}, {pps:.0f} pkt/s)", ok))

    # ---- phase 5: recovery again ----------------------------------------
    for _ in range(10):
        sim_seconds(engine, clock, 1, 3, evs, 400, 80)
    st, ml, pps = status_of(engine)
    ok = st == "Normal"
    checks.append((f"final recovery to Normal (status={st})", ok))

    # ---- alert log sanity ------------------------------------------------
    alerts = engine.snapshot()["alerts"]
    kinds = [a["status"] for a in alerts]
    ok = ("DoS" in kinds and "DDoS" in kinds and
          sum(1 for a in alerts if a["status"] == "Normal") >= 2)
    checks.append((f"alerts logged for every transition "
                   f"({[a['status'] for a in alerts]})", ok))

    print("\n=== results ===")
    all_ok = True
    for label, ok in checks:
        print(("  ✅ " if ok else "  ❌ ") + label)
        all_ok = all_ok and ok

    snap = engine.snapshot()
    print(f"\nsummary: {snap['total_packets']} packets | "
          f"LSTM votes {snap['ml_total']} | agreement {snap['agreement']}%")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
