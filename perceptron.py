import torch
import torch.nn as nn
import torch.optim as optim
from mendeleev import element

# 1. PRE-FETCH DATA (Fast!)
training_list = [
    ('Cu', 1), ('Fe', 1), ('Au', 1), # Metals
    ('O', 0), ('S', 0), ('Ne', 0)    # Non-metals
]

def get_clean_tensor(symbol):
    el = element(symbol)
    # We only need Atomic Number and Electronegativity to prove the point
    return torch.tensor([el.atomic_number / 118.0, (el.en_pauling or 0.0) / 4.0])

# Pre-calculate tensors so the loop is pure math
X_train = torch.stack([get_clean_tensor(s) for s, l in training_list])
Y_train = torch.tensor([[float(l)] for s, l in training_list])

# 2. THE BRAIN
model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)

optimizer = optim.Adam(model.parameters(), lr=0.05) # Faster learning rate
criterion = nn.BCELoss()

# 3. TRAINING (This will be instant)
print("Training...")
for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, Y_train)
    loss.backward()
    optimizer.step()

# 4. TEST
with torch.no_grad():
    for test_s in ['Al', 'Cl', 'Si']:
        test_input = get_clean_tensor(test_s)
        prob = model(test_input).item()
        print(f"Element: {test_s:2} | Confidence: {prob*100:5.1f}% | {'METAL' if prob > 0.5 else 'NON-METAL'}")
