"""Per-class force/energy std on the bio_dimers test set.

Reads the published test xyz, groups structures by atoms.info['label'],
and writes class_stats.npz with `<label>_force_std` (meV/A) and
`<label>_energy_std` (meV/atom). Cached so the figure scripts in results/
don't re-parse 13k structures on every plot regeneration.

Energy std is computed on offset-subtracted energies: per-element
baselines are read from the LOREM checkpoint's baseline.yaml (the same
offsets the model adds back when run with add_offset=True). Without
this subtraction the std is dominated by composition variation across
dimers in the class (absolute DFT energies are ~100s of eV/atom, while
interaction errors are ~meV/atom).
"""

from pathlib import Path
import numpy as np
import yaml
from ase.data import atomic_numbers
from ase.io import read

HERE = Path(__file__).resolve().parent
TEST = HERE / "../../datasets/bio_dimers_test.xyz"
BASELINE = HERE / "lorem/run/checkpoints/R2_E+F/model/baseline.yaml"
OUT = HERE / "class_stats.npz"

with open(BASELINE) as f:
    baseline = yaml.safe_load(f)
offset = {z: float(e) for z, e in baseline["elemental"].items()}
print(f"per-element offsets (eV) by Z: {offset}")

systems = read(TEST, index=":")

forces_by_class = {}
energy_by_class = {}
for atoms in systems:
    label = atoms.info["label"]
    forces_by_class.setdefault(label, []).append(atoms.get_forces())
    baseline_e = sum(offset[atomic_numbers[s]] for s in atoms.get_chemical_symbols())
    energy_by_class.setdefault(label, []).append(
        (atoms.get_potential_energy() - baseline_e) / len(atoms)
    )

stats = {}
for label in forces_by_class:
    f = np.concatenate(forces_by_class[label], axis=0)
    e = np.array(energy_by_class[label])
    stats[f"{label}_force_std"] = float(f.std()) * 1e3
    stats[f"{label}_energy_std"] = float(e.std()) * 1e3
    stats[f"{label}_n_systems"] = len(energy_by_class[label])
    stats[f"{label}_n_atoms"] = f.shape[0]

np.savez(OUT, **stats)
print(f"wrote {OUT}")
for k, v in sorted(stats.items()):
    print(f"  {k}: {v}")
