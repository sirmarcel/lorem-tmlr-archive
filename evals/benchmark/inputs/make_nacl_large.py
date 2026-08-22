"""Generate large NaCl supercells for scaling benchmarks up to ~150k atoms.

Cubic (k x k x k) supercells only, k=1..27 → 8 to 157,464 atoms.
Used for PME + nolr inference speed benchmarks.

Each structure has atoms.info fields:
- crystal_name: "NaCl_cubic"
- supercell: "kxkxk"
- is_cubic: True
"""

from ase.build import bulk
from ase.io import write

# NaCl conventional cell: 8 atoms (4 Na + 4 Cl)
unit = bulk("NaCl", crystalstructure="rocksalt", a=5.64, cubic=True)
atoms_per_cell = len(unit)  # 8

max_atoms = 160000  # 8 * 27^3 = 157464

# cubic supercells: 8*k^3 for k = 1..27
structures = []
k = 1
while atoms_per_cell * k**3 <= max_atoms:
    n = atoms_per_cell * k**3
    atoms = unit.repeat((k, k, k))
    atoms.info["crystal_name"] = "NaCl_cubic"
    atoms.info["supercell"] = f"{k}x{k}x{k}"
    atoms.info["is_cubic"] = True
    assert len(atoms) == n
    structures.append(atoms)
    k += 1

print(f"{len(structures)} structures, {len(structures[0])}–{len(structures[-1])} atoms")
print()
for atoms in structures:
    k = atoms.info["supercell"]
    print(f"  {k:>10s} = {len(atoms):>7d} atoms")

write("nacl_large_crystals.xyz", structures, format="extxyz")
print(f"\nWrote {len(structures)} structures to nacl_large_crystals.xyz")
