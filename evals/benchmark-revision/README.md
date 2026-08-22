# Runtime benchmark (revision: matched accuracy)

Revised Ewald-vs-PME timing data for the long-range summation in LOREM.

The original benchmark in `../benchmark/` compared the two methods at unmatched accuracy: it set both the Ewald k-space cutoff and the PME mesh spacing to `cutoff/8`, which over-converges Ewald (to near machine precision) while PME delivers a relative force error of ~1.3e-2 (rattled NaCl-64, fp64). This folder contains a re-run of the comparison with settings matched at equal force accuracy. It underlies the revised runtime benchmark in the paper (Appendix A.2: the scaling figure and Tables A.1-A.3) and the comparison document `../../tex/comparison.tex`; the originally-published timings remain in `../benchmark/`.

## Regimes

Smearing is fixed at `cutoff/4 = 1.25 A` throughout; a `factor` sets `lr_wavelength` (Ewald) and `mesh_spacing` (PME) to `cutoff/factor`. The regimes were selected with an accuracy sweep on a rattled NaCl-64 supercell with formal +-1 charges, against a tightly-converged Ewald reference at matched smearing (the sweep itself is not included here). PME's factor sets the target accuracy of each regime; Ewald's factor is then the coarsest swept setting whose force error matches or beats PME's. Because Ewald converges exponentially in the factor, the matched Ewald ends up somewhat *better*-converged than PME, so the reported Ewald timings are a (mild) upper bound on the cost of an exactly-matched Ewald.

| regime | Ewald factor | Ewald F_rel | PME factor | PME F_rel |
|---|---|---|---|---|
| **light** | 4 | ~1.5e-3 | 8 | ~1.3e-2 |
| **tight** | 6 | ~4.3e-6 | 32 | ~2.1e-4 |

`light` matches the accuracy the original benchmark's PME setting was actually delivering, and is what the revised paper figures and tables use; `tight` is the most accurate PME setting swept.

## Contents

- `{light,tight}/{mode}_{crystal}_times.yaml`: timing data (seconds per prediction step, energy + forces). Keys are `<crystal>_<N_atoms>`; `mode` is one of `pme`, `ewald_unbatched`, `nolr`; `crystal` is one of `nacl`, `cscl`, `cu2o`.
  - `nolr` (short-range-only baseline) is factor-independent; the same data appears in both regimes. It was re-run together with the revision so that the long-range-fraction breakdown does not mix timings from different code versions.
  - In the `light` regime, NaCl `pme` and `nolr` extend to ~85k atoms.
- `run_rebench.py`: benchmark script, a parametrised version of `../benchmark/run.py` exposing `--factor`, `--smearing-factor`, and the output regime `--tag`
- `lorem_pme.py`: unbatched LOREM model variants for PME, Ewald, and short-range-only inference (as in `../benchmark/`)

The scripts in `../../results/` (`benchmark.py`, `benchmark_figure.py`, `benchmark_table.py`, `benchmark_full_figure.py`, and the `benchmark_comparison*` scripts) read from this folder.

## Setup

Same as `../benchmark/`: single NVIDIA H100 GPU, 10 timing repetitions after JIT warmup, default LOREM hyperparameters (see `../benchmark/README.md`).

## Reproducing

```bash
# generate benchmark structures (see ../benchmark/README.md)
cd ../benchmark/inputs && python make_supercells.py --cubic-only && cd ../../benchmark-revision

# light regime
python run_rebench.py --modes ewald_unbatched --factor 4 --tag light \
    --dataset ../benchmark/inputs/nacl_crystals.xyz --cubic-only --repeats 10
python run_rebench.py --modes pme --factor 8 --tag light \
    --dataset ../benchmark/inputs/nacl_crystals.xyz --cubic-only --repeats 10

# tight regime
python run_rebench.py --modes ewald_unbatched --factor 6 --tag tight \
    --dataset ../benchmark/inputs/nacl_crystals.xyz --cubic-only --repeats 10
python run_rebench.py --modes pme --factor 32 --tag tight \
    --dataset ../benchmark/inputs/nacl_crystals.xyz --cubic-only --repeats 10

# nolr baseline (factor-independent; the factor value is ignored)
python run_rebench.py --modes nolr --factor 8 --tag light \
    --dataset ../benchmark/inputs/nacl_crystals.xyz --cubic-only --repeats 10

# repeat for cscl_crystals.xyz and cu2o_crystals.xyz;
# outputs land in run/{tag}/{mode}_{crystal}_times.yaml
```

Modes that run out of memory at large N (Ewald beyond ~14k atoms; PME at `factor=32` beyond a few thousand atoms) log the failure and continue with the next structure.
