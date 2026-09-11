import pandas as pd
import time

# Load dataset
df = pd.read_csv("C:/Users/ADMIN/Desktop/BTech/Codes/Phase 1/phase1_ev_attack_dataset.csv")

print("Columns in dataset:")
print(df.columns)

# Correct features
features = ["request_rate", "packet_size", "time_gap"]

def stream_data(delay=0.5):
    for i in range(len(df)):
        sample = df.iloc[i][features].values
        
        yield sample
        
        time.sleep(delay)


# Test
if __name__ == "__main__":
    stream = stream_data()

    for i in range(5):
        data = next(stream)
        print(f"Packet {i+1}: {data}")