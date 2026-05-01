import os
from pathlib import Path

from mp_api.client import MPRester
import pandas as pd


def load_api_key():
    api_key = os.getenv("API_KEY")
    if api_key:
        return api_key

    env_path = Path(__file__).resolve().with_name(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == "API_KEY":
                value = value.strip().strip("'\"")
                if value:
                    return value

    raise RuntimeError("Missing API_KEY. Add it to the repo-root .env file or export API_KEY in your shell.")


API_KEY = load_api_key()

with MPRester(API_KEY) as mpr:
    # Query for stable materials with a calculated band gap
    # We'll pull formula, band_gap, and elements for our features
    docs = mpr.materials.summary.search(
        is_stable=True,
        fields=["formula_pretty", "band_gap", "elements"]
    )
    
    # Convert to a list of dicts for easy DataFrame conversion
    data = [{
        "formula": doc.formula_pretty,
        "band_gap": doc.band_gap,
        "elements": [str(e) for e in doc.elements]
    } for doc in docs]

df = pd.DataFrame(data)
df.to_csv("real_materials_data.csv", index=False)
print(f"Saved {len(df)} real entries to real_materials_data.csv")
