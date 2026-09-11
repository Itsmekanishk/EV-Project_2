import torch
import torch.nn as nn
import shap
import numpy as np

# ----------------------------
# Recreate LSTM Architecture
# ----------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=1, num_classes=4):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]   # Take last timestep
        out = self.fc(out)
        return out

# ----------------------------
# Load Model Weights
# ----------------------------
model = LSTMModel()
model.load_state_dict(torch.load("C:/Users/ADMIN/Desktop/BTech/Codes/Phase 3/lstm_model.pth"))
model.eval()

print("Model loaded successfully")

# ----------------------------
# Load Test Data
# ----------------------------
X_test = np.load("C:/Users/ADMIN/Desktop/BTech/Codes/Phase 3/X_sequences.npy")
y_test = np.load("C:/Users/ADMIN/Desktop/BTech/Codes/Phase 3/y_sequences.npy")

print("Test data shape:", X_test.shape)

# ---------------------------------
# Convert to Torch Tensors
# ---------------------------------
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# ---------------------------------
# Select Background Dataset (for SHAP)
# ---------------------------------
background = X_test_tensor[:100]   # small subset
print("Background shape:", background.shape)

# ---------------------------------
# Select Samples to Explain
# ---------------------------------
samples_to_explain = X_test_tensor[200:205]   # 5 samples
print("Samples to explain shape:", samples_to_explain.shape)

# ---------------------------------
# SHAP DeepExplainer
# ---------------------------------
print("\nInitializing SHAP DeepExplainer...")

explainer = shap.DeepExplainer(model, background)

print("Computing SHAP values...")
shap_values = explainer.shap_values(samples_to_explain, check_additivity=False)

print("SHAP computation completed.")

# Check shape of SHAP output
if isinstance(shap_values, list):
    print("Number of output classes:", len(shap_values))
    print("SHAP shape for class 0:", np.array(shap_values[0]).shape)
else:
    print("SHAP values shape:", np.array(shap_values).shape)

    # ---------------------------------
# STEP 4: Global Feature Importance
# ---------------------------------

# Convert to numpy
shap_array = np.array(shap_values)

# Select class index (DoS = 1)
class_index = 1

# Extract SHAP values for that class
# shape: (samples, timesteps, features)
class_shap = shap_array[:, :, :, class_index]

# Take absolute values
class_shap_abs = np.abs(class_shap)

# Average over samples and timesteps
feature_importance = class_shap_abs.mean(axis=(0, 1))

feature_names = [
    "request_rate",
    "packet_size",
    "time_gap",
    "burst_duration"
]

print("\nGlobal Feature Importance for DoS:")
for name, value in zip(feature_names, feature_importance):
    print(f"{name}: {value:.6f}")

import matplotlib.pyplot as plt

# ---------------------------------
# Plot Global Feature Importance
# ---------------------------------

plt.figure()
plt.bar(feature_names, feature_importance)
plt.xlabel("Features")
plt.ylabel("Mean |SHAP Value|")
plt.title("Global Feature Importance for DoS Detection (LSTM)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ---------------------------------
# STEP 6: Local Explanation
# ---------------------------------

sample_index = 0
single_sample = samples_to_explain[sample_index:sample_index+1]

with torch.no_grad():
    output = model(single_sample)
    predicted_class = torch.argmax(output, dim=1).item()

print("\nPredicted class for sample 0:", predicted_class)

single_shap = shap_array[sample_index, :, :, predicted_class]
single_feature_importance = np.mean(np.abs(single_shap), axis=0)

print("\nLocal Feature Contribution for Sample 0:")
for name, value in zip(feature_names, single_feature_importance):
    print(f"{name}: {value:.6f}")


# ---------------------------------
# Plot Global Feature Importance
# ---------------------------------

import matplotlib.pyplot as plt

plt.figure()
plt.bar(feature_names, feature_importance)
plt.xlabel("Features")
plt.ylabel("Mean |SHAP Value|")
plt.title("Global Feature Importance for DoS Detection (LSTM)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ---------------------------------
# STEP 8: LIME Integration
# ---------------------------------

from lime.lime_tabular import LimeTabularExplainer

# Flatten sequence data: (N, 10, 4) → (N, 40)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

print("\nFlattened shape for LIME:", X_test_flat.shape)

# Feature names for flattened input
lime_feature_names = []
for t in range(10):
    for f in feature_names:
        lime_feature_names.append(f"{f}_t{t}")

# Prediction function wrapper
def lime_predict(input_data):
    input_tensor = torch.tensor(input_data.reshape(-1, 10, 4), dtype=torch.float32)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
    return probs.numpy()

# ---------------------------------
# STEP 9: Generate LIME Explanation
# ---------------------------------

# Initialize LIME explainer
explainer_lime = LimeTabularExplainer(
    training_data=X_test_flat[:1000],  # small subset for speed
    feature_names=lime_feature_names,
    class_names=["Normal", "DoS", "DDoS", "Malformed"],
    mode="classification"
)

# Choose sample index (same one as SHAP example)
lime_sample_index = 0
lime_sample = X_test_flat[lime_sample_index]

# Generate explanation
exp = explainer_lime.explain_instance(
    lime_sample,
    lime_predict,
    num_features=10
)

print("\nLIME Explanation for Sample 0:")
for feature, weight in exp.as_list():
    print(f"{feature}: {weight:.6f}")

# Save visualization
fig = exp.as_pyplot_figure()
plt.tight_layout()
plt.savefig("lime_explanation_sample0.png")
plt.show()