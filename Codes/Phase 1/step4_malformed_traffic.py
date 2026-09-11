import csv
import time
import random

OUTPUT_FILE = "step4_ev_malformed.csv"

with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "timestamp",
        "src_id",
        "request_rate",
        "packet_size",
        "time_gap",
        "label"
    ])

    for _ in range(800):
        timestamp = time.time()
        request_rate = round(random.uniform(5, 50), 2)

        # Extreme packet sizes (too small OR too large)
        packet_size = random.choice([
            random.randint(0, 50),
            random.randint(2000, 5000)
        ])

        time_gap = round(random.uniform(0.001, 2), 4)

        writer.writerow([
            timestamp,
            "FUZZ_1",
            request_rate,
            packet_size,
            time_gap,
            3   # malformed / fuzzed traffic
        ])

print("✅ Malformed EV traffic dataset generated:", OUTPUT_FILE)
