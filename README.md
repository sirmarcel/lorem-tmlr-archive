# Learning Long-Range Representations with Equivariant Messages: Supplemental materials

This archive is a supplement to the TMLR paper ["Learning Long-Range Representations with Equivariant Messages" (LOREM)](https://openreview.net/forum?id=pZI9e4SW9P). It contains code and data required to reconstruct the results from the paper. The repository is assigned [doi:10.5281/zenodo.19421697](https://doi.org/10.5281/zenodo.19421697) and available at [github.com/sirmarcel/lorem-tmlr-archive](https://github.com/sirmarcel/lorem-tmlr-archive).

**Note:** The input files in this repository have been lightly updated to be compatible with the reference implementation of LOREM, [`lorem-jax`](https://github.com/lab-cosmo/lorem-jax) to aid future reproduction. The results in the paper were generated with different pre-release versions of both LOREM and the backend libraries [`marathon`](https://github.com/sirmarcel/marathon) and [`jax-pme`](https://github.com/lab-cosmo/jax-pme). Copies of the original code can be found in the submission data share at [doi:10.5281/zenodo.17789350](https://doi.org/10.5281/zenodo.17789350).

**Note:** This archive contains updated results for biodimers (Fig. 4) and the benchmark (Fig. 6) that, as of October 2026, are not yet reflected on the TMLR-published version of the paper. We will update the Github archive once the revised version has been published, and likely leave Zenodo unchanged.

***

The archive is structured as follows:

- `evals/` contains the experiment-specific scripts and results with different models
- `lorem/` contains per-experiment wrapper scripts that bridge shipped LOREM checkpoints to the metric/result fragments in `evals/`
- `results/` contains scripts that process the data from `evals/` into tables and figures for the paper
- `datasets/` contains training data and other datasets as well as instructions on how to process them
- `tex/` is the LaTeX source of the paper

## Setup/Installation

Follow the [`lorem-jax`](https://github.com/lab-cosmo/lorem-jax) installation instructions, then install two additional runtime dependencies used by the evaluation scripts:

```
pip install scikit-learn tqdm
```

`scikit-learn` and `tqdm` are imported by the `evals/metrics_*.py` fragments but are not declared as dependencies of `lorem-jax` or `marathon`.

Before running any script, export `DATASETS` to point at this archive's `datasets/` directory (absolute path required, so `marathon.data.datasets` resolves correctly):

```
export DATASETS="$(pwd)/datasets"
```

A smoke test should succeed:

```
python -c "import lorem, marathon, jaxpme"
```

**Note:** the three largest training files (`datasets/{sn2,AuMgO,NaCl}_train.xyz`) are kept out of the git repository; they are included in the Zenodo mirror and fully regenerable from public sources (see `datasets/README.md`). `datasets/prepare.py` skips the corresponding blocks when the files are absent.

## Running training/evaluation

Training is done via `lorem-jax` (see documentation), expecting a `settings.yaml` and `model.yaml` in the current working directory. The results are stored in `run/`. Note that you cannot restore the training state from this repo, we only shipp model checkpoints. For this reason, if you wish to train a model, you need to either rename or remove `run/` to avoid restoring the checkpoint, which will fail.

Computing results or metrics is done via `python /path/to/archive/lorem/{results,metrics}_XXX.py`, from the eval directory. For example, to regenerate the `metrics.npz` for AuMgO from the shipped checkpoint:

```
cd evals/AuMgO/lorem
DATASETS=../../../datasets python ../../../lorem/metrics_AuMgO.py
```

The wrappers in `lorem/` load `run/checkpoints/R2_E+F` via `lorem.calculator.Calculator.from_checkpoint` and hand it to the shared `evals/metrics_XXX.py` fragment, which computes RMSE/MAE over the experiment's test set and writes `metrics.npz`. Non-LOREM models have their own `<model>/<expt>_rmse.py` drivers with equivalent behavior, invoked the same way from their respective eval directories. **Note that this overwrites the shipped `metrics.npz` in the eval directory in place** — make a copy first if you want to keep the original for comparison.

## Training configuration: `settings.yaml` and `settings.legacy.yaml`

Most `evals/<expt>/<variant>/` directories ship two configuration files:

- **`settings.yaml`**: today-compatible, consumed by `lorem-train`. Mechanically migrated from the pre-release config with these rules:
    - `batcher.batch_size` ← `chunk_size` if `chunk_size > 1` (pre-release vmapped `chunk_size` real samples over `num_graphs=2` padding graphs), else ← `num_graphs`.
    - `batcher.max_total_atoms` ← `chunk_size × num_nodes` (the per-step atom budget; reduces to `num_nodes` when `chunk_size = 1`).
    - `num_edges` is dropped (no direct analog in today's batcher).
    - `use_wandb` is set to `False` by default; change it if you want your own wandb logging.
    - Internal dataset path prefixes (`4G/…`, `efa/…`, `hlhc23/…`) are rewritten to the flat archive-relative names (`AuMgO_train`, `cumulene_train`, etc.).
- **`settings.legacy.yaml`**: copy of the original pre-release configuration, verbatim except that in a few files the dataset paths (`train:`/`valid:`) have been rewritten to the archive-relative names. Kept for posterity; reading this tells you exactly what the paper's training run was configured with.

The `adapt` models were trained (during rebuttals) with the current `marathon`/`lorem-jax` stack rather than the pre-release one, so they ship only `settings.yaml` (no `settings.legacy.yaml`).

## Non-PBC LR Coulomb convention

The shipped non-PBC LOREM checkpoints (in `evals/{NaCl,cumulene,sn2}/lorem*`, excluding the `-nolr-*` variants and the short-range `-sr-*` variants which do not invoke the Coulomb path) were trained with a pre-production `lorem.py` that (accidentally) computed the non-PBC long-range potential as `V_i = sum_{j ≠ i} q_j / r_ij`. Today's `jax-pme` applies a `1/2` factor on the non-PBC Coulomb path to match the PBC convention (where the returned potential is the one for which `E = sum_i q_i V_i` gives the per-atom energy directly, without an outer `1/2`).

As a consequence, `lorem.calculator.Calculator.from_checkpoint(...)` on these shipped non-PBC checkpoints under today's `jax-pme` feeds a halved LR potential into the downstream MLP, producing wrong results.

**The shipped `metrics.npz` files are the ground truth from the paper-era training.** To reproduce those numbers from the shipped non-PBC checkpoints under today's code, either:

1. pre-scale the non-PBC LR potential by `2` before feeding it to the downstream MLP — the relevant call site is the `Ewald()` calculator in the long-range block of `lorem/models/mlip.py` (this is the live path for these checkpoints; the `Ewald()` in `lorem/models/bec.py` is not), or
2. retrain the affected variant with today's `lorem-jax` (paper figures stay frozen as they load from shipped `.npz` files, independent of the checkpoints).

All PBC checkpoints (AuMgO, bio_dimers, adapt) are unaffected since the PBC Ewald convention has not changed.
