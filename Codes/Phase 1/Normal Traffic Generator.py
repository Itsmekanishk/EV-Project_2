import csv
import time
import random

OUTPUT_FILE = "step1_ev_normal.csv"

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

    for ev_id in range(1, 21):        # 20 unique EVs
        print(f"Generating traffic for EV_{ev_id}")  # 🔎 verification
        for sample in range(200):
            timestamp = time.time()
            request_rate = round(random.uniform(1, 3), 2)
            packet_size = random.randint(200, 600)
            time_gap = round(random.uniform(0.3, 1.5), 3)

            writer.writerow([
                timestamp,
                f"EV_{ev_id}",        # ✅ guaranteed dynamic ID
                request_rate,
                packet_size,
                time_gap,
                0                     # normal traffic
            ])

            time.sleep(0.003)

print("✅ Normal EV traffic dataset generated correctly")
