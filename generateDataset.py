from mp_api.client import MPRester
import pandas as pd

# Replace with your actual API key from the MP dashboard
API_KEY = "key here"

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
