import torch
import torch.nn as nn
import torch.optim as optim
from mendeleev import element
import pandas as pd
import re
from tqdm import tqdm
import random
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# --- 1. DATASET CLASSES ---
class MaterialDataset(Dataset):
    def __init__(self, prepared_data):
        self.data = prepared_data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        (feat, w), target = self.data[idx]
        return feat, w, target

def collate_fn(batch):
    features, weights, targets = zip(*batch)
    features_padded = pad_sequence(features, batch_first=True)
    weights_padded = pad_sequence(weights, batch_first=True)
    targets_stack = torch.stack(targets)
    return features_padded, weights_padded, targets_stack

# --- 2. LOGIC FUNCTIONS ---
def parse_formula(formula):
    tokens = re.findall(r'([A-Z][a-z]*)(\d*)', str(formula))
    expanded = []
    for symbol, count in tokens:
        count = int(count) if count else 1
        expanded.extend([symbol] * count)
    return expanded

def get_compound_tensor(symbols, cache):
    global_max = torch.tensor([118.0, 294.0, 3.98, 225.0, 8.0, 25.0, 10.0])
    raw_matrix, weights = [], []
    for s in symbols:
        if s not in cache: cache[s] = element(s)
        el = cache[s]
        weights.append(5.0 if s not in ['O', 'H', 'F', 'Cl', 'Br', 'S', 'N'] else 1.0)
        
        valence = 0.0
        for name in ['n_valence', 'nvalence', 'valence_electrons']:
            if hasattr(el, name):
                v = getattr(el, name); valence = float(v() if callable(v) else v)
                break
        
        d_elec = 0.0
        try: d_elec = float(sum(occ for (n, l, occ) in el.ec.conf if l == 2))
        except: pass

        ion_e = float(el.ionenergies.get(1, 0.0)) if el.ionenergies else 0.0

        row = [float(el.atomic_number), float(el.atomic_weight), float(el.en_pauling or 0.0),
               float(el.covalent_radius or 0.0), valence, ion_e, d_elec]
        raw_matrix.append(row)
    return torch.tensor(raw_matrix, dtype=torch.float) / global_max, torch.tensor(weights, dtype=torch.float).view(-1, 1)

# --- 3. THE UPDATED BATCH-READY BRAIN ---
class BandGapModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 34k samples allows for much higher dimensionality
        self.network = nn.Sequential(
            nn.Linear(7, 256),
            nn.ReLU(),
            nn.Dropout(0.1), # Prevents memorization
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x, w):
        latent = self.network(x)
        weighted_sum = torch.sum(latent * w, dim=1)
        total_weight = torch.sum(w, dim=1)
        return weighted_sum / total_weight

# --- 4. PREPARATION ---
def load_and_prepare_csv(file_path):
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} compounds.")
    unique_elements = set()
    for formula in df['formula']:
        unique_elements.update(re.findall(r'[A-Z][a-z]*', str(formula)))
    cache = {s: element(s) for s in unique_elements}
    
    prepared_list = []
    print("Pre-processing into tensors...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        atoms = parse_formula(row['formula'])
        feat, w = get_compound_tensor(atoms, cache)
        prepared_list.append(((feat, w), torch.tensor([float(row['band_gap'])])))
    return prepared_list, cache

# --- 5. EXECUTION ---
csv_path = "real_materials_data.csv" # Ensure this file exists
prepared, master_cache = load_and_prepare_csv(csv_path)

dataset = MaterialDataset(prepared)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

model = BandGapModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Learning Rate Scheduler: Drops LR by half every 30 epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

print(f"Training on {device} with LR Scheduling...")
for epoch in range(101):
    model.train()
    total_loss = 0
    for feats, weights, targets in tqdm(train_loader, leave=False):
        feats, weights, targets = feats.to(device), weights.to(device), targets.to(device)
        
        optimizer.zero_grad()
        output = model(feats, weights)
        
        # --- CUSTOM WEIGHTED LOSS ---
        # Give 3x more importance to non-metals (target > 0.1)
        loss_weights = torch.where(targets > 0.1, 3.0, 1.0)
        loss = (loss_weights * (output - targets)**2).mean()
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    scheduler.step() # Tick the scheduler
    
    if epoch % 10 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch} | Loss: {total_loss/len(train_loader):.5f} | LR: {current_lr}")

# --- 6. VALIDATION ---
model.eval()
test_set = {"Al(OH)3": ['Al','O','O','O','H','H','H'], "Fe(OH)3": ['Fe','O','O','O','H','H','H']}
print("\n--- Validation Results ---")
for name, atoms in test_set.items():
    f, w = get_compound_tensor(atoms, master_cache)
    # Add batch dimension [1, atoms, features] for the model
    f, w = f.unsqueeze(0).to(device), w.unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(f, w).item()
        print(f"{name:7} | Predicted: {pred:.2f} eV")

torch.save(model.state_dict(), "real_world_brain.pth")
