# Runtime benchmark

Timing data for Ewald vs PME long-range summation in LOREM, reported in Appendix A.2 (Tables A.1-A.3, Figure A.1).

## Setup

All benchmarks were run on a single NVIDIA H100 GPU, averaged over 10 repetitions (after JIT warmup), computing energy and forces.

Model hyperparameters: `l_max=6`, `l_max_lr=2`, 128 features, 32 radial basis, 8 spherical channels, cutoff 5.0 A. PME mesh spacing: `cutoff/8 = 0.625 A`, smearing: `cutoff/4 = 1.25 A`, with {2,3,5,7}-smooth FFT grid padding.

## Contents

- `*_times.yaml`: timing data (seconds per prediction step). Keys are `<crystal>_<N_atoms>`.
  - `pme_{nacl,cscl,cu2o}_times.yaml`: PME, all sizes, cubic supercells
  - `ewald_unbatched_{nacl,cscl,cu2o}_times.yaml`: Ewald, capped at ~8000 atoms
  - `nolr_{nacl,cscl,cu2o}_times.yaml`: short-range only baseline (no long-range block)
  - `{pme,nolr}_nacl_large_times.yaml`: large-scale NaCl scaling up to 32k atoms
- `run.py`: benchmark script
- `lorem_pme.py`: unbatched LOREM model variants for PME, Ewald, and short-range-only inference
- `inputs/`: crystal unit cells and supercell generation scripts
  - `CsCl_cubic.xyz`, `Cu2O_cubic.xyz`: unit cells (NaCl is built from ASE, see below)
  - `make_supercells.py`: generates `{nacl,cscl,cu2o}_crystals.xyz` supercell datasets
  - `make_nacl_large.py`: generates large NaCl supercells up to ~157k atoms

## Reproducing

```bash
# generate benchmark structures from unit cells
cd inputs
python make_supercells.py --cubic-only
# -> nacl_crystals.xyz, cscl_crystals.xyz, cu2o_crystals.xyz

# run benchmarks (requires GPU)
cd ..
python run.py --modes pme nolr --dataset inputs/nacl_crystals.xyz --cubic-only --repeats 10
python run.py --modes ewald_unbatched --dataset inputs/nacl_crystals.xyz --cubic-only --repeats 10 --max-atoms 8000
# repeat for cscl_crystals.xyz and cu2o_crystals.xyz
```

NaCl is built directly from ASE via `bulk("NaCl", crystalstructure="rocksalt", a=5.64, cubic=True)` (8 atoms/cell). CsCl (2 atoms/cell) and Cu2O (6 atoms/cell) unit cells are provided as `.xyz` files.
