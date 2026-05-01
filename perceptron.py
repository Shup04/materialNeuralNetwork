import torch
import torch.nn as nn
import torch.optim as optim
from mendeleev import element
import random
from tqdm import tqdm

# --- 1. THE GENERATOR (Internal Data Factory) ---
def generate_big_dataset_cached(n_entries=500):
    symbols_to_cache = [
        'Li', 'Na', 'K', 'Rb', 'Cs', 'Be', 'Mg', 'Ca', 'Sr', 'Ba',
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Y', 'Zr', 'Nb', 'Mo', 'Ag', 'Cd', 'B', 'Al', 'Ga', 'In', 
        'Sn', 'Pb', 'F', 'O', 'Cl', 'Br', 'S', 'N', 'H'
    ]
    print("Caching Periodic Table properties...")
    cache = {s: element(s) for s in symbols_to_cache}
    
    anions = {
        "F": (['F'], 4.0), "O": (['O'], 3.5), "OH": (['O', 'H'], 3.2),
        "Cl": (['Cl'], 3.0), "Br": (['Br'], 2.8), "S": (['S'], 2.5)
    }
    metals = [s for s in symbols_to_cache if s not in ['F', 'O', 'Cl', 'Br', 'S', 'N', 'H']]
    
    data = {}
    print(f"Generating {n_entries} compounds...")
    while len(data) < n_entries:
        m_sym = random.choice(metals)
        a_name, (a_list, a_chi) = random.choice(list(anions.items()))
        el_m = cache[m_sym]
        
        # Physics Heuristic
        m_chi = el_m.en_pauling or 1.5
        gap = max(0.1, (abs(a_chi - m_chi) * 3.2) + (8.0 / el_m.atomic_number) - (0.9 if el_m.block == 'd' else 0.0))
        
        name = f"{m_sym}{a_name}_{len(data)}"
        data[name] = ([m_sym] + a_list, round(gap, 2))
    return data, cache

# --- 2. FEATURE ENGINEERING ---
def get_compound_tensor(symbols, cache):
    # Features: [At_Num, Mass, EN, Radius, Valence, Ion_Energy, D_Electrons]
    global_max = torch.tensor([118.0, 294.0, 3.98, 225.0, 8.0, 25.0, 10.0])
    raw_matrix, weights = [], []
    
    for s in symbols:
        el = cache[s] if s in cache else element(s)
        # Weight the Metal (Protagonist) higher than the Anions
        weights.append(5.0 if s not in ['O', 'H', 'F', 'Cl', 'Br', 'S', 'N'] else 1.0)
        
        # 1. Valence (Universal Check)
        valence = 0.0
        for name in ['n_valence', 'nvalence', 'valence_electrons']:
            if hasattr(el, name):
                v = getattr(el, name)
                valence = float(v() if callable(v) else v)
                break

        # 2. D-Electrons (Configuration Check)
        d_elec = 0.0
        try: d_elec = float(sum(occ for (n, l, occ) in el.ec.conf if l == 2))
        except: pass

        # 3. Ionization Energy (Dictionary Check - Key 1 is the first energy)
        # Using .get(1) avoids the KeyError: 0
        ion_e = 0.0
        if el.ionenergies:
            ion_e = float(el.ionenergies.get(1, 0.0))

        row = [
            float(el.atomic_number), 
            float(el.atomic_weight), 
            float(el.en_pauling or 0.0),
            float(el.covalent_radius or 0.0), 
            valence, 
            ion_e, 
            d_elec
        ]
        raw_matrix.append(row)
        
    return torch.tensor(raw_matrix, dtype=torch.float) / global_max, torch.tensor(weights, dtype=torch.float).view(-1, 1)

# --- 3. THE BRAIN ---
class BandGapModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(7, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x, w):
        latent = self.network(x)
        return torch.sum(latent * w, dim=0, keepdim=True) / torch.sum(w)

# --- 4. EXECUTION ---
raw_data, master_cache = generate_big_dataset_cached(500)
prepared = [(get_compound_tensor(a, master_cache), torch.tensor([[g]])) for a, g in raw_data.values()]

model = BandGapModel()
optimizer = optim.Adam(model.parameters(), lr=0.005)
criterion = nn.MSELoss()

print("\nTraining on 500 compounds...")
for epoch in tqdm(range(1501)):
    random.shuffle(prepared) # Shuffle every epoch to prevent memorization
    total_loss = 0
    for (feat, w), target in prepared:
        optimizer.zero_grad()
        loss = criterion(model(feat, w), target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 300 == 0:
        print(f"Epoch {epoch} | Avg Loss: {total_loss/len(prepared):.5f}")

# --- 5. THE TEST ---
test_set = {"Al(OH)3": ['Al','O','O','O','H','H','H'], "Fe(OH)3": ['Fe','O','O','O','H','H','H']}
print("\n--- Results ---")
for name, atoms in test_set.items():
    f, w = get_compound_tensor(atoms, master_cache)
    print(f"{name:7} | Predicted: {model(f, w).item():.2f} eV")

# Save the weights
torch.save(model.state_dict(), "hydroxide_brain.pth")
print("Weights saved. No more waiting!")
