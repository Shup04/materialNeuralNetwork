import torch
import torch.nn as nn
import torch.optim as optim
from mendeleev import element

# --- 1. THE BRAIN ---
class CompoundBrain(nn.Module):
    def __init__(self, input_dim):
        super(CompoundBrain, self).__init__()
        self.encoder = nn.Linear(input_dim, 16)
        self.predictor = nn.Linear(16, 1)
        
    def forward(self, atom_features):
        latent_atoms = torch.relu(self.encoder(atom_features))
        compound_vector = torch.mean(latent_atoms, dim=0, keepdim=True)
        return self.predictor(compound_vector)

# --- 2. DATA PREP HELPER ---
def get_compound_data(symbols):
    props = ['atomic_number', 'atomic_weight', 'en_pauling', 'covalent_radius']
    global_max = torch.tensor([118.0, 294.0, 3.98, 225.0])
    raw = [[getattr(element(s), p) or 0.0 for p in props] for s in symbols]
    return torch.tensor(raw, dtype=torch.float) / global_max

# --- 3. THE DATASET (Small Sample for PoC) ---
# Format: "Compound Name": (List of Atoms, Actual Tc)
training_data = {
    "YBCO": (['Y', 'Ba', 'Cu', 'O'], 92.0),
    "Magnesium Diboride": (['Mg', 'B', 'B'], 39.0),
    "Mercury Cuprate": (['Hg', 'Ba', 'Cu', 'O'], 133.0),
    "Lead": (['Pb'], 7.2)
}

# --- 4. THE TRAINING LOOP ---
model = CompoundBrain(input_dim=4)
optimizer = optim.Adam(model.parameters(), lr=0.01) # The Mechanic
criterion = nn.MSELoss() # The Judge

print("Starting Training...")
for epoch in range(500):
    total_loss = 0
    for name, (atoms, real_tc) in training_data.items():
        optimizer.zero_grad() # Reset the mechanic's tools
        
        # Forward pass
        data = get_compound_data(atoms)
        prediction = model(data)
        
        # Calculate Loss
        target = torch.tensor([[real_tc]], dtype=torch.float)
        loss = criterion(prediction, target)
        
        # Backpropagation (The AI learns from its mistake)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Loss: {total_loss/len(training_data):.4f}")

# --- 5. THE HYDRIDE TEST (The Unseen Data) ---
print("\n--- Final Test on Unseen Hydride ---")
hydride_atoms = ['La'] + ['H']*10 # LaH10
data = get_compound_data(hydride_atoms)
prediction = model(data)
print(f"Predicted Tc for Lanthanum Hydride: {prediction.item():.2f} K")
print("(Note: Real LaH10 is ~250K at high pressure!)")
