import csv
import time
import random

OUTPUT_FILE = "step3_ev_ddos.csv"
bots = [f"BOT_{i}" for i in range(1, 11)]  # 10 attackers

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

    for bot in bots:
        for _ in range(300):  # traffic per bot
            timestamp = time.time()
            request_rate = round(random.uniform(30, 100), 2)
            packet_size = random.randint(600, 1400)
            time_gap = round(random.uniform(0.005, 0.1), 4)

            writer.writerow([
                timestamp,
                bot,
                request_rate,
                packet_size,
                time_gap,
                2   # DDoS
            ])

print("✅ DDoS attack dataset generated:", OUTPUT_FILE)
