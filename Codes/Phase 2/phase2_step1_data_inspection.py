import pandas as pd

# Step 1.1: Load dataset
df = pd.read_csv("C:/Users/ADMIN/Desktop/BTech/Codes/Phase 1/phase1_ev_attack_dataset.csv")

# Step 1.2: Basic structure
print("Dataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

# Step 1.3: Preview data
print("\nFirst 5 Rows:")
print(df.head())

# Step 1.4: Missing values check
print("\nMissing Values per Column:")
print(df.isnull().sum())

# Step 1.5: Label distribution
print("\nLabel Distribution:")
print(df['label'].value_counts())

# Step 1.6: Basic statistics (numerical columns)
print("\nStatistical Summary:")
print(df.describe())
