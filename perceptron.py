import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from mendeleev import element
from pymatgen.core import Composition

import pandas as pd
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# -----------------------------
# 1. DATASET
# -----------------------------

class MaterialDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


# -----------------------------
# 2. CHEMISTRY HELPERS
# -----------------------------

NONMETALS = {
    "H", "C", "N", "O", "F", "P", "S", "Cl", "Se", "Br", "I"
}

HALOGENS = {
    "F", "Cl", "Br", "I"
}

COMMON_ANIONS = {
    "O", "H", "F", "Cl", "Br", "S", "N", "P", "C", "I", "Se"
}


def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def get_element_data(symbol, cache):
    if symbol not in cache:
        cache[symbol] = element(symbol)

    el = cache[symbol]

    atomic_number = safe_float(el.atomic_number)
    atomic_weight = safe_float(el.atomic_weight)
    electronegativity = safe_float(el.en_pauling)
    covalent_radius = safe_float(el.covalent_radius)

    ion_e = 0.0
    try:
        if el.ionenergies:
            ion_e = safe_float(el.ionenergies.get(1, 0.0))
    except Exception:
        ion_e = 0.0

    valence = 0.0
    for name in ["n_valence", "nvalence", "valence_electrons"]:
        try:
            if hasattr(el, name):
                v = getattr(el, name)
                valence = safe_float(v() if callable(v) else v)
                break
        except Exception:
            pass

    d_elec = 0.0
    try:
        d_elec = float(sum(occ for (n, l, occ) in el.ec.conf if l == 2))
    except Exception:
        d_elec = 0.0

    is_nonmetal = 1.0 if symbol in NONMETALS else 0.0
    is_halogen = 1.0 if symbol in HALOGENS else 0.0
    is_common_anion = 1.0 if symbol in COMMON_ANIONS else 0.0
    is_metal = 1.0 if symbol not in NONMETALS else 0.0

    return {
        "atomic_number": atomic_number,
        "atomic_weight": atomic_weight,
        "electronegativity": electronegativity,
        "covalent_radius": covalent_radius,
        "ion_e": ion_e,
        "valence": valence,
        "d_elec": d_elec,
        "is_nonmetal": is_nonmetal,
        "is_halogen": is_halogen,
        "is_common_anion": is_common_anion,
        "is_metal": is_metal,
    }


def weighted_mean(values, weights):
    total = sum(weights)
    if total == 0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / total


def weighted_variance(values, weights):
    total = sum(weights)
    if total == 0:
        return 0.0
    mean = weighted_mean(values, weights)
    return sum(w * ((v - mean) ** 2) for v, w in zip(values, weights)) / total


def composition_to_features(formula, cache):
    """
    Converts a chemical formula into one feature vector.

    This avoids the old problem where the model was basically averaging
    independent atom predictions.
    """
    comp = Composition(formula)
    el_amounts = comp.get_el_amt_dict()

    symbols = list(el_amounts.keys())
    amounts = [float(el_amounts[s]) for s in symbols]
    total_atoms = sum(amounts)

    if total_atoms == 0:
        raise ValueError(f"Could not parse formula: {formula}")

    element_rows = [get_element_data(s, cache) for s in symbols]

    def prop_list(prop):
        return [row[prop] for row in element_rows]

    atomic_numbers = prop_list("atomic_number")
    atomic_weights = prop_list("atomic_weight")
    electronegativities = prop_list("electronegativity")
    covalent_radii = prop_list("covalent_radius")
    ion_es = prop_list("ion_e")
    valences = prop_list("valence")
    d_elecs = prop_list("d_elec")

    is_nonmetal = prop_list("is_nonmetal")
    is_halogen = prop_list("is_halogen")
    is_common_anion = prop_list("is_common_anion")
    is_metal = prop_list("is_metal")

    # Fractions of specific chemically important elements
    frac_o = el_amounts.get("O", 0.0) / total_atoms
    frac_h = el_amounts.get("H", 0.0) / total_atoms
    frac_n = el_amounts.get("N", 0.0) / total_atoms
    frac_s = el_amounts.get("S", 0.0) / total_atoms
    frac_c = el_amounts.get("C", 0.0) / total_atoms
    frac_f = el_amounts.get("F", 0.0) / total_atoms
    frac_cl = el_amounts.get("Cl", 0.0) / total_atoms

    # Weighted means
    mean_z = weighted_mean(atomic_numbers, amounts)
    mean_weight = weighted_mean(atomic_weights, amounts)
    mean_en = weighted_mean(electronegativities, amounts)
    mean_radius = weighted_mean(covalent_radii, amounts)
    mean_ion_e = weighted_mean(ion_es, amounts)
    mean_valence = weighted_mean(valences, amounts)
    mean_d = weighted_mean(d_elecs, amounts)

    # Ranges and variation
    en_range = max(electronegativities) - min(electronegativities)
    z_range = max(atomic_numbers) - min(atomic_numbers)
    radius_range = max(covalent_radii) - min(covalent_radii)
    ion_e_range = max(ion_es) - min(ion_es)

    en_var = weighted_variance(electronegativities, amounts)
    z_var = weighted_variance(atomic_numbers, amounts)
    radius_var = weighted_variance(covalent_radii, amounts)

    # Composition categories
    frac_metal = weighted_mean(is_metal, amounts)
    frac_nonmetal = weighted_mean(is_nonmetal, amounts)
    frac_halogen = weighted_mean(is_halogen, amounts)
    frac_common_anion = weighted_mean(is_common_anion, amounts)

    num_unique_elements = len(symbols)

    # Crude stoichiometry information
    max_fraction = max(amounts) / total_atoms
    min_fraction = min(amounts) / total_atoms

    features = [
        mean_z,
        mean_weight,
        mean_en,
        mean_radius,
        mean_ion_e,
        mean_valence,
        mean_d,

        en_range,
        z_range,
        radius_range,
        ion_e_range,

        en_var,
        z_var,
        radius_var,

        frac_metal,
        frac_nonmetal,
        frac_halogen,
        frac_common_anion,

        frac_o,
        frac_h,
        frac_n,
        frac_s,
        frac_c,
        frac_f,
        frac_cl,

        num_unique_elements,
        total_atoms,
        max_fraction,
        min_fraction,
    ]

    return features


# -----------------------------
# 3. FEATURE NORMALIZATION
# -----------------------------

class StandardScaler:
    def fit(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        self.mean = x.mean(dim=0)
        self.std = x.std(dim=0)

        # Avoid divide-by-zero for constant columns
        self.std[self.std == 0] = 1.0

    def transform(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        return ((x - self.mean) / self.std).numpy()

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)


# -----------------------------
# 4. MODEL
# -----------------------------

class BandGapModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.10),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.10),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)


# -----------------------------
# 5. EVALUATION
# -----------------------------

def evaluate_model(model, loader, device):
    model.eval()

    preds = []
    actuals = []

    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)

            output = model(features)

            preds.append(output.cpu())
            actuals.append(targets.cpu())

    preds = torch.cat(preds)
    actuals = torch.cat(actuals)

    mse = ((preds - actuals) ** 2).mean()
    rmse = torch.sqrt(mse)
    mae = torch.abs(preds - actuals).mean()

    return {
        "mse": mse.item(),
        "rmse": rmse.item(),
        "mae": mae.item(),
        "preds": preds,
        "actuals": actuals,
    }


def mean_baseline(targets):
    targets = torch.tensor(targets, dtype=torch.float32).view(-1, 1)
    mean_pred = targets.mean()
    mse = ((targets - mean_pred) ** 2).mean()
    rmse = torch.sqrt(mse)
    mae = torch.abs(targets - mean_pred).mean()

    return {
        "mean_prediction": mean_pred.item(),
        "mse": mse.item(),
        "rmse": rmse.item(),
        "mae": mae.item(),
    }


# -----------------------------
# 6. LOAD DATA
# -----------------------------

def load_and_prepare_csv(file_path):
    df = pd.read_csv(file_path)

    print(f"Loaded {len(df)} compounds.")

    required_columns = {"formula", "band_gap"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    # Remove missing or invalid rows
    df = df.dropna(subset=["formula", "band_gap"])
    df["band_gap"] = pd.to_numeric(df["band_gap"], errors="coerce")
    df = df.dropna(subset=["band_gap"])

    print(f"Using {len(df)} valid compounds after cleanup.")

    cache = {}
    features = []
    targets = []

    print("Converting formulas into composition features...")

    skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        formula = str(row["formula"])
        band_gap = float(row["band_gap"])

        try:
            feat = composition_to_features(formula, cache)
            features.append(feat)
            targets.append(band_gap)
        except Exception as e:
            skipped += 1

    print(f"Prepared {len(features)} compounds.")
    print(f"Skipped {skipped} compounds due to parsing errors.")

    return features, targets, cache


# -----------------------------
# 7. MAIN EXECUTION
# -----------------------------

def main():
    csv_path = "real_materials_data.csv"

    # Sanity-check formula parsing
    print("--- Formula parsing sanity check ---")
    for f in ["Al(OH)3", "Fe(OH)3", "Al2O3", "Ca(OH)2"]:
        comp = Composition(f)
        print(f, comp.get_el_amt_dict())

    features, targets, master_cache = load_and_prepare_csv(csv_path)

    # Train/validation split
    x_train, x_val, y_train, y_val = train_test_split(
        features,
        targets,
        test_size=0.20,
        random_state=42
    )

    # Scale using train data only
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)

    train_dataset = MaterialDataset(x_train, y_train)
    val_dataset = MaterialDataset(x_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=512,
        shuffle=False
    )

    # Baseline comparisons
    train_baseline = mean_baseline(y_train)
    val_baseline = mean_baseline(y_val)

    print("\n--- Mean Baseline ---")
    print(f"Train mean prediction: {train_baseline['mean_prediction']:.4f} eV")
    print(f"Train baseline MSE:    {train_baseline['mse']:.4f}")
    print(f"Train baseline RMSE:   {train_baseline['rmse']:.4f} eV")
    print(f"Train baseline MAE:    {train_baseline['mae']:.4f} eV")

    print(f"\nVal mean prediction:   {val_baseline['mean_prediction']:.4f} eV")
    print(f"Val baseline MSE:      {val_baseline['mse']:.4f}")
    print(f"Val baseline RMSE:     {val_baseline['rmse']:.4f} eV")
    print(f"Val baseline MAE:      {val_baseline['mae']:.4f} eV")

    # Model setup
    input_dim = len(features[0])
    model = BandGapModel(input_dim=input_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4
    )

    criterion = nn.MSELoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10
    )

    print(f"\nTraining on {device}.")
    print(f"Input feature count: {input_dim}")

    best_val_rmse = float("inf")
    best_model_path = "best_bandgap_model.pth"

    # Training loop
    for epoch in range(1, 151):
        model.train()
        total_loss = 0.0

        for batch_features, batch_targets in tqdm(train_loader, leave=False):
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()

            output = model(batch_features)
            loss = criterion(output, batch_targets)

            loss.backward()

            # Helps prevent occasional unstable updates
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        train_metrics = evaluate_model(model, train_loader, device)
        val_metrics = evaluate_model(model, val_loader, device)

        scheduler.step(val_metrics["rmse"])

        current_lr = optimizer.param_groups[0]["lr"]

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "scaler_mean": scaler.mean,
                    "scaler_std": scaler.std,
                    "feature_count": input_dim,
                },
                best_model_path
            )

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Train RMSE: {train_metrics['rmse']:.4f} eV | "
                f"Val RMSE: {val_metrics['rmse']:.4f} eV | "
                f"Val MAE: {val_metrics['mae']:.4f} eV | "
                f"LR: {current_lr:.6f}"
            )

    print("\n--- Final Results ---")
    train_metrics = evaluate_model(model, train_loader, device)
    val_metrics = evaluate_model(model, val_loader, device)

    print(f"Train RMSE: {train_metrics['rmse']:.4f} eV")
    print(f"Train MAE:  {train_metrics['mae']:.4f} eV")
    print(f"Val RMSE:   {val_metrics['rmse']:.4f} eV")
    print(f"Val MAE:    {val_metrics['mae']:.4f} eV")

    print("\n--- Compare Against Mean Baseline ---")
    print(f"Val baseline RMSE: {val_baseline['rmse']:.4f} eV")
    print(f"Model val RMSE:    {val_metrics['rmse']:.4f} eV")

    if val_metrics["rmse"] < val_baseline["rmse"]:
        print("Model beats the mean baseline.")
    else:
        print("Model does NOT beat the mean baseline yet.")

    print(f"\nBest model saved to: {best_model_path}")

    # -----------------------------
    # 8. MANUAL TESTING
    # -----------------------------

    print("\n--- Manual Formula Predictions ---")

    test_formulas = [
        "Al(OH)3",
        "Fe(OH)3",
        "Al2O3",
        "SiO2",
        "GaAs",
        "Fe",
        "Cu",
    ]

    model.eval()

    for formula in test_formulas:
        try:
            feat = composition_to_features(formula, master_cache)
            feat_scaled = scaler.transform([feat])
            feat_tensor = torch.tensor(feat_scaled, dtype=torch.float32).to(device)

            with torch.no_grad():
                pred = model(feat_tensor).item()

            print(f"{formula:10} | Predicted band gap: {pred:.3f} eV")

        except Exception as e:
            print(f"{formula:10} | Error: {e}")


if __name__ == "__main__":
    main()
