import pandas as pd

files = [
    "step1_ev_normal.csv",
    "step2_ev_dos.csv",
    "step3_ev_ddos.csv",
    "step4_ev_malformed.csv"
]

dfs = [pd.read_csv(f) for f in files]
final_df = pd.concat(dfs, ignore_index=True)

# Shuffle dataset (important for ML)
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

final_df.to_csv("phase1_ev_attack_dataset.csv", index=False)

print("✅ Phase 1 final dataset created: phase1_ev_attack_dataset.csv")
print(final_df["label"].value_counts())
