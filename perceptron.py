from mendeleev import element
import torch
import torch.nn as nn

# --- STEP 1: DATA INGESTION (Your Mendeleev Code) ---
atoms = ['Cu', 'O', 'Sr']
properties = ['atomic_number', 'atomic_weight', 'en_pauling', 'covalent_radius']
raw_matrix = []

for symbol in atoms:
    el = element(symbol)
    row = [getattr(el, p) for p in properties]
    raw_matrix.append(row)

node_features = torch.tensor(raw_matrix, dtype=torch.float)

# Normalize the data (squash between 0 and 1)
min_vals = node_features.min(dim=0, keepdim=True)[0]
max_vals = node_features.max(dim=0, keepdim=True)[0]
normalized_features = (node_features - min_vals) / (max_vals - min_vals)

print("--- 1. NORMALIZED PERIODIC TABLE DATA ---")
print(normalized_features, "\n")

# --- STEP 2: THE AI LAYER ---
# We tell PyTorch: "Take these 4 physical properties and translate them 
# into 8 abstract 'Latent Space' numbers using your random weights."
hidden_layer = nn.Linear(in_features=4, out_features=8)

# Pass all 3 atoms through the AI layer simultaneously
raw_output = hidden_layer(normalized_features)

# Pass the result through the ReLU activation gate
latent_output = torch.relu(raw_output)

print("--- 2. AI LATENT SPACE OUTPUT ---")
print(latent_output)
