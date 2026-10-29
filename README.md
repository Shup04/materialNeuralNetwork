# materialNeuralNetwork

A materials-learning experiment that asks whether a chemical formula, represented
through the properties and proportions of its elements, contains enough information
to predict a useful electronic property. The implemented model predicts **band gap
in electronvolts (eV)** from composition. The original motivation was much more
ambitious: learning from crystal structures to explore superconducting materials.

The repository captures the steps between those ideas: small atom-level networks,
predictions collapsing toward the mean, a larger Materials Project dataset, and a
replacement model that learns from whole-composition features. It contains the
training scripts, downloaded data and saved weights. It does not yet implement a
crystal graph network or predict superconducting critical temperature.

## The original idea

The initial blueprint proposed searching a large space of possible materials using
continuous physical features rather than treating element names as unrelated IDs.
The hoped-for benefit was to learn relationships between chemistry, atomic
arrangement and a target property instead of manually trying every composition.

Its particular materials hypothesis was a **chemical-pressure, doped hydride
ceramic**: a rigid transition-metal/electronegative lattice, lightweight hydrogen
interstitials and substitutional doping. The blueprint associated those ingredients
with confinement, vibrations and altered electronic behavior, with room-temperature
superconductivity as the eventual ambition. These are the project's motivating
hypotheses, not findings established by this repository. Neither the dataset nor the
current model tests that proposed mechanism.

The proposed representation was spatial: atoms as feature-bearing nodes, neighbors
and distances as edges, and larger supercells with vacancies, interstitials or
substitutions. Three or four message-passing steps would update each atom from its
neighbors, followed by global pooling and a readout predicting critical temperature
(`Tc`). The planned first inputs were crystal structures such as CIF files paired
with appropriate target measurements.

That graph pipeline remains a future research direction. The working code instead
tackles a narrower question using a readily available band-gap dataset. Band-gap
prediction does not, by itself, demonstrate superconductivity discovery.

## How the experiment evolved

The Git history records several concrete changes in approach:

- Early versions built physical-property arrays for atoms and inspected the
  network's intermediate representation. Small training sets produced inaccurate
  predictions, including outputs clustered around the mean.
- The dataset grew from a handful of examples to a downloaded materials table.
  An intermediate model applied the same network to each atom, then averaged its
  scalar predictions using hand-selected element weights.
- The current [`composition_to_features`](perceptron.py#L188) explicitly replaces
  that independent-atom averaging with one vector describing the whole formula.
  It combines property means with differences, variation, element fractions and
  stoichiometry, so the network can learn interactions among those summaries.
- Formula parsing moved to `pymatgen.Composition`, including grouped formulas such
  as `Al(OH)3`. Training gained a validation split, train-only standardization,
  baseline comparisons and best-checkpoint saving.

The final commit is titled “predicting band gaps works.” That records the author's
stopping point; the repository does not include a reproducible held-out result that
establishes how accurate or generalizable the saved model is.

## What is in the repository

| File | Role |
| --- | --- |
| [`generateDataset.py`](generateDataset.py) | Queries Materials Project for stable materials and writes formulas, calculated band gaps and element lists. |
| [`real_materials_data.csv`](real_materials_data.csv) | Committed dataset: 33,973 rows and 33,822 distinct formula strings. |
| [`perceptron.py`](perceptron.py) | Formula parsing, feature construction, normalization, MLP training, evaluation and manual formula predictions. |
| `best_bandgap_model.pth` | Checkpoint format written by the current trainer: model weights, input dimension and scaler mean/standard deviation. |
| `real_world_brain.pth` | Legacy weights from the earlier atom-level model; the current trainer does not load them. |

There is no inference-only command, dependency lock or automated test suite in this
snapshot. The two weight files belong to different model designs and should not be
treated as interchangeable.

## The implemented data flow

```text
Materials Project stable-material summaries
    -> formula + calculated band_gap + elements CSV
    -> formula parsing and 29 composition features
    -> 80/20 row split
    -> standardization fitted on training rows only
    -> PyTorch MLP
    -> validation metrics and best checkpoint
```

### Data collection

[`generateDataset.py`](generateDataset.py#L30) requests entries with
`is_stable=True`, selecting `formula_pretty`, `band_gap` and `elements`. It writes
`real_materials_data.csv` in the current directory. It does not download crystal
coordinates, bonds, CIF files, superconducting temperatures or material IDs.

The trainer reads the formula and band-gap columns, removes missing/non-numeric
targets and skips formulas that fail feature construction. The CSV's element-list
column is not used by the current model; the formula is parsed again instead.

### Why these 29 features?

[`get_element_data`](perceptron.py#L119) caches properties from `mendeleev`.
`composition_to_features` weights them by the atom counts parsed from each formula.

| Group | Features | What the representation retains |
| --- | --- | --- |
| Weighted means (7) | Atomic number, atomic weight, electronegativity, covalent radius, first ionization energy, valence electrons, d-electron count | The average elemental character of the formula. |
| Ranges (4) | Electronegativity, atomic number, covalent radius, ionization energy | Differences between constituent elements that an average alone would hide. |
| Weighted variances (3) | Electronegativity, atomic number, covalent radius | How unevenly those properties are distributed. |
| Category fractions (4) | Metal, nonmetal, halogen, common-anion categories | The proportions assigned by the code's explicit element sets. |
| Specific fractions (7) | O, H, N, S, C, F, Cl | Direct composition signals for selected elements. |
| Stoichiometry (4) | Unique-element count, total atom count, largest and smallest element fraction | Basic size and balance of the formula. |

These are hand-designed summaries, not a structural description. Missing element
properties generally fall back to zero. The metal/nonmetal and related categories
are the script's heuristic sets, not a learned chemistry classification.

The design trades crystal geometry for a small fixed-size input: formulas of
different lengths can enter the same ordinary feed-forward model. The cost is that
different structures sharing the same formula receive the same representation.

### Model and training

[`BandGapModel`](perceptron.py#L329) has widths `29 -> 256 -> 128 -> 64 -> 1`.
Hidden layers use ReLU, with batch normalization and 10% dropout in the first two
blocks. A final Softplus constrains the predicted value to be positive.

The trainer uses an 80/20 random row split with `random_state=42`, fits its custom
scaler only on training features, and runs 150 epochs. It uses AdamW, mean squared
error, gradient clipping and a learning-rate reduction when validation RMSE stops
improving. CUDA is selected if available; otherwise it uses the CPU.

The script prints RMSE and MAE in eV and compares against constant-mean baselines.
It also fetches Materials Project reference entries and prints predictions for
`Al(OH)3`, `Fe(OH)3`, `Al2O3`, `SiO2`, `GaAs`, `Fe` and `Cu`. Those examples are
sanity checks, not an independent benchmark or evidence of novel-material discovery.

## How to run this snapshot

Use a separate Python environment. The project does not record compatible package
versions; the following dependency list comes from the imports, not a verified lock.
The existing `.gitignore` only ignores `.env`, so this example keeps the environment
outside the repository:

```sh
python3 -m venv ../material-neural-network-env
source ../material-neural-network-env/bin/activate
python -m pip install torch mendeleev pymatgen pandas tqdm scikit-learn mp-api
```

Supply a Materials Project credential as the `API_KEY` environment variable or in a
local repository-root `.env` file. The loader checks the environment first. Keep the
credential local rather than adding it to a tracked file.

From the repository root:

```sh
# Train using the checked-in CSV; may overwrite best_bandgap_model.pth.
python perceptron.py

# Optional dataset refresh; queries Materials Project and replaces the CSV.
python generateDataset.py
```

The trainer **still needs API access when using the existing CSV**: it loads the key
at module import and fetches reference values before training. It does not resume
the saved checkpoint; it initializes a new model. Preserve the current dataset and
weights before replacing either. These commands were not run to write this README,
and compatibility with a newly installed environment remains unverified.

## What the current results can and cannot establish

**Composition is not structure.** The committed CSV contains 151 repeated-formula
groups; ten have band gaps spanning more than 1 eV. Identical formula inputs cannot
distinguish those targets. The script already prints duplicate-formula diagnostics.
This is a limit of the chosen input representation, not something more training
epochs can resolve.

**The validation split is by row.** Repeated compositions can appear on both sides
of that split, so it does not establish performance on unseen compositions. No
specific amount of metric inflation is claimed here. Also, the printed validation
mean baseline is calculated from validation targets themselves; a predictive
constant baseline for a held-out evaluation should be fitted on training targets.

**The saved and reported models can differ.** The loop saves weights whenever
validation RMSE improves, but final metrics and manual predictions use the last
epoch's in-memory model without reloading that checkpoint. The filename alone does
not identify which model produced the printed final results.

**The run is not fully reproducible yet.** The split seed is fixed, but model
initialization and training randomness are not. Package versions, the acquisition
date/query snapshot and a complete evaluation record are not captured alongside the
checkpoint. The saved weights were not loaded or scientifically validated during
this documentation pass.

## Definition of done

The composition-to-band-gap experiment can be finished on its own terms. It does
not need to become the original superconductivity discovery system to count as a
completed project.

- [ ] **Make one run reproducible.** Record the working environment, dataset
  identity/provenance, feature order, random seeds and split membership. A fresh
  environment should reproduce the documented training/evaluation procedure.
- [ ] **Evaluate generalization honestly.** Hold out whole canonical compositions,
  keep that final evaluation separate from checkpoint selection, and compare the
  selected model with a training-mean baseline using RMSE/MAE. Report how duplicate
  formulas and conflicting targets were handled.
- [ ] **Make the checkpoint the deliverable.** Reload the selected weights and
  scaler for reported results, and provide a prediction path that uses them without
  retraining or fetching unrelated reference data. Verify that reloaded predictions
  match the values recorded for that checkpoint.
- [ ] **Write the experiment's conclusion.** Preserve the evaluation results and a
  few representative successes/failures. State whether composition-only features
  added useful predictive information over the baseline, and record the limits.
  An honest negative result also completes the experiment; crystal graphs and
  superconducting targets can remain a separately scoped future project.
