import torch
import torch.nn as nn
import torch.optim as optim
from mendeleev import element

# 1. THE DATASET (Same 10, but we're going for higher accuracy)
training_data = {
    "NaOH": (['Na', 'O', 'H'], 5.8),
    "LiOH": (['Li', 'O', 'H'], 6.0),
    "KOH": (['K', 'O', 'H'], 5.5),
    "Mg(OH)2": (['Mg', 'O', 'H', 'H'], 5.0),
    "Ca(OH)2": (['Ca', 'O', 'H', 'H'], 5.6),
    "Zn(OH)2": (['Zn', 'O', 'H', 'H'], 3.3),
    "Ni(OH)2": (['Ni', 'O', 'H', 'H'], 3.7),
    "Co(OH)2": (['Co', 'O', 'H', 'H'], 3.0),
    "Cu(OH)2": (['Cu', 'O', 'H', 'H'], 2.6),
    "Mn(OH)2": (['Mn', 'O', 'H', 'H'], 3.5)
}

def get_compound_tensor(symbols):
    # Features: [At_Num, Mass, EN, Radius, Valence, Ion_Energy, D_Electrons]
    global_max = torch.tensor([118.0, 294.0, 3.98, 225.0, 8.0, 25.0, 10.0])
    
    raw_matrix = []
    weights = []
    
    for s in symbols:
        el = element(s)
        # Weighting: Protagonist (Metal) vs Supporting Cast (O, H)
        weights.append(5.0 if s not in ['O', 'H'] else 1.0)
        
        # 1. Standard Props
        row = [
            float(el.atomic_number),
            float(el.atomic_weight),
            float(el.en_pauling or 0.0),
            float(el.covalent_radius or 0.0)
        ]
        
        # 2. Robust Valence Fetch
        valence = 0.0
        for name in ['n_valence', 'nvalence', 'valence_electrons']:
            if hasattr(el, name):
                v_attr = getattr(el, name)
                valence = float(v_attr() if callable(v_attr) else v_attr)
                break
        row.append(valence)
        
        # 3. Ionization Energy (The Physics "Cheat Code")
        ion_e = float(el.ionenergies[1] if el.ionenergies else 0.0)
        row.append(ion_e)
        
        # 4. FIXED: D-Electron Count
        # Orbital l=2 is the d-orbital. el.ec.conf is [(n, l, occupancy), ...]
        d_electrons = 0.0
        try:
            d_electrons = float(sum(occ for (n, l, occ) in el.ec.conf if l == 2))
        except:
            d_electrons = 0.0
        row.append(d_electrons)
        
        raw_matrix.append(row)
        
    tensor = torch.tensor(raw_matrix, dtype=torch.float) / global_max
    weight_tensor = torch.tensor(weights, dtype=torch.float).view(-1, 1)
    return tensor, weight_tensor

# Pre-cache
prepared_data = []
for atoms, gap in training_data.values():
    feat, w = get_compound_tensor(atoms)
    prepared_data.append((feat, w, torch.tensor([[gap]])))

# 2. THE BRAIN (With Weighted Pooling)
class WeightedBandGapModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x, weights):
        latent = self.network(x)
        # WEIGHTED POOLING: (Latent * Weights).sum() / Weights.sum()
        weighted_sum = torch.sum(latent * weights, dim=0, keepdim=True)
        return weighted_sum / torch.sum(weights)

# --- THE DUAL-PATH BRAIN ---
class CationFocusedModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Path A: Just the Metal Atom (The Protagonist)
        self.cation_path = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        # Path B: The whole compound average (The Background)
        self.env_path = nn.Sequential(
            nn.Linear(7, 16),
            nn.ReLU()
        )
        # Final Decision
        self.readout = nn.Linear(16 + 16, 1)
        
    def forward(self, x, weights):
        # 1. Identify the Metal (The one with weight 5.0)
        metal_idx = torch.argmax(weights)
        metal_features = x[metal_idx].unsqueeze(0)
        
        # 2. Get Cation Character
        cation_vibe = self.cation_path(metal_features)
        
        # 3. Get Environment Vibe (Mean Pooling)
        env_vibe = torch.mean(self.env_path(x), dim=0, keepdim=True)
        
        # 4. Combine and Predict
        combined = torch.cat((cation_vibe, env_vibe), dim=1)
        return self.readout(combined)

model = CationFocusedModel()
optimizer = optim.Adam(model.parameters(), lr=0.005)
criterion = nn.MSELoss()

# 3. TRAINING
print("Training with Weighted Metal-Cation Bias...")
for epoch in range(2500):
    total_loss = 0
    for feat, w, target in prepared_data:
        optimizer.zero_grad()
        pred = model(feat, w)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 500 == 0:
        print(f"Epoch {epoch} | Loss: {total_loss/len(prepared_data):.5f}")

# 4. TEST
test_hydroxides = {"Al(OH)3": ['Al','O','O','O','H','H','H'], "Fe(OH)3": ['Fe','O','O','O','H','H','H']}
for name, atoms in test_hydroxides.items():
    f, w = get_compound_tensor(atoms)
    with torch.no_grad():
        print(f"Compound: {name:7} | Predicted Band Gap: {model(f, w).item():.2f} eV")
