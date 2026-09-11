import csv
import time
import random

OUTPUT_FILE = "step2_ev_dos.csv"

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

    for i in range(1000):  # DoS burst
        timestamp = time.time()
        request_rate = round(random.uniform(50, 200), 2)
        packet_size = random.randint(800, 1500)
        time_gap = round(random.uniform(0.001, 0.05), 4)

        writer.writerow([
            timestamp,
            "ATTACKER_1",
            request_rate,
            packet_size,
            time_gap,
            1   # DoS attack
        ])

print("✅ DoS attack dataset generated:", OUTPUT_FILE)
