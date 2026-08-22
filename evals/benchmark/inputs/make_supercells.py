"""Generate supercells for benchmarking, up to ~32k atoms.

Usage:
    python make_supercells.py                    # all crystals, cubic + rect
    python make_supercells.py --cubic-only       # only n x n x n supercells
    python make_supercells.py --max-atoms 8000   # limit size

Reads unit cells from *_cubic.xyz files in the same directory.
Outputs <name>_crystals.xyz for each crystal type.

Each structure has atoms.info fields:
- crystal_name: "<Name>" (e.g. "NaCl", "CsCl", "Cu2O")
- supercell: "axbxc"
- is_cubic: True/False
"""

import argparse
from pathlib import Path

import numpy as np
from ase.build import bulk
from ase.io import read, write

parser = argparse.ArgumentParser()
parser.add_argument("--cubic-only", action="store_true",
                    help="only generate n x n x n supercells")
parser.add_argument("--max-atoms", type=int, default=32768)
parser.add_argument("--n-rect", type=int, default=50,
                    help="approximate number of rectangular supercells")
args = parser.parse_args()

here = Path(__file__).parent


def make_supercells(name, unit, max_atoms, cubic_only=False, n_rect=50):
    """Generate supercells for a given unit cell."""
    n_unit = len(unit)

    # -- cubic supercells: n_unit * k^3 --
    cubic = []
    k = 1
    while n_unit * k**3 <= max_atoms:
        cubic.append((k, k, k))
        k += 1

    # -- rectangular supercells for coverage --
    rect_candidates = {}
    if not cubic_only:
        for a in range(1, 50):
            for b in range(a, 50):
                for c in range(b, 50):
                    n = n_unit * a * b * c
                    if n > max_atoms:
                        break
                    if a == b == c:
                        continue
                    aspect = c / a
                    if n not in rect_candidates or aspect < rect_candidates[n][0]:
                        rect_candidates[n] = (aspect, (a, b, c))

        all_rect_sizes = sorted(rect_candidates.keys())

        # thin to well-spaced sizes
        targets = list(np.geomspace(max(n_unit, 8), 2048, n_rect // 2))
        targets += list(np.linspace(2048, max_atoms, n_rect // 2))
        selected_rect = []
        used = set()
        for t in targets:
            if not all_rect_sizes:
                break
            best = min(all_rect_sizes, key=lambda s: abs(s - t))
            if best not in used:
                selected_rect.append(best)
                used.add(best)
        selected_rect.sort()
    else:
        selected_rect = []

    # -- build structures --
    structures = []

    for a, b, c in cubic:
        n = n_unit * a * b * c
        atoms = unit.repeat((a, b, c))
        atoms.info["crystal_name"] = name
        atoms.info["supercell"] = f"{a}x{b}x{c}"
        atoms.info["is_cubic"] = True
        assert len(atoms) == n
        structures.append(atoms)

    for n in selected_rect:
        _, (a, b, c) = rect_candidates[n]
        atoms = unit.repeat((a, b, c))
        atoms.info["crystal_name"] = name
        atoms.info["supercell"] = f"{a}x{b}x{c}"
        atoms.info["is_cubic"] = False
        assert len(atoms) == n
        structures.append(atoms)

    structures.sort(key=len)
    return structures


# -- discover unit cells --
# NaCl is built from ase (conventional rocksalt)
crystals = {
    "NaCl": bulk("NaCl", crystalstructure="rocksalt", a=5.64, cubic=True),
}

# read additional *_cubic.xyz files
for path in sorted(here.glob("*_cubic.xyz")):
    name = path.stem.replace("_cubic", "")
    unit = read(path, format="extxyz")
    crystals[name] = unit
    print(f"Loaded {name} from {path.name}: {len(unit)} atoms/cell, "
          f"a={unit.cell[0, 0]:.4f} Å")

print()

# -- generate and write --
for name, unit in crystals.items():
    structures = make_supercells(
        name, unit, args.max_atoms,
        cubic_only=args.cubic_only, n_rect=args.n_rect,
    )
    n_cubic = sum(1 for a in structures if a.info["is_cubic"])
    n_rect = len(structures) - n_cubic

    outfile = f"{name.lower()}_crystals.xyz"
    print(f"=== {name} ({len(unit)} atoms/cell) ===")
    print(f"  {len(structures)} structures, {n_cubic} cubic, {n_rect} rectangular")
    print()
    for atoms in structures:
        tag = "CUBIC" if atoms.info["is_cubic"] else "     "
        print(f"  {tag}  {atoms.info['supercell']:>10s} = {len(atoms):>5d} atoms")

    write(here / outfile, structures, format="extxyz")
    print(f"\n  -> {outfile}\n")
