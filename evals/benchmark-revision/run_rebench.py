"""Re-benchmark Ewald vs PME at accuracy-matched settings for LOREM.

Parametrised version of `../benchmark/run.py`. The original hardcoded
`mesh_spacing = lr_wavelength = cutoff/8`; here we expose the factor (and
the smearing factor) so we can run the matched-accuracy regimes, which
were selected from an accuracy sweep on NaCl-64 (see README.md).

Usage:

    python run_rebench.py --modes ewald_unbatched --factor 4 --tag light \\
        --dataset ../benchmark/inputs/nacl_crystals.xyz --cubic-only --repeats 10

Writes `run/{tag}/{mode}_{crystal}_times.yaml`, matching the per-crystal
filename convention of `../benchmark/`. The `{crystal}` slug is
derived from the dataset filename (`nacl_crystals.xyz` -> `nacl`).

Selected regimes (from the NaCl-64 sweep, σ=cutoff/4=1.25 fixed):
  - light:  Ewald factor=4, PME factor=8     (F_rel ~ 1.3e-2)
  - tight:  Ewald factor=6, PME factor=32    (F_rel ~ 2e-4)
"""

import argparse
import os
import re
import traceback

import jax
import jax.numpy as jnp
from ase.io import read
from time import monotonic

from marathon.io import write_yaml


cutoff = 5.0

parser = argparse.ArgumentParser()
parser.add_argument("--modes", nargs="+", default=["ewald_unbatched", "pme"],
                    choices=["ewald", "pme", "ewald_unbatched", "nolr"])
parser.add_argument("--factor", type=int, required=True,
                    help="lr_wavelength/mesh_spacing = cutoff/factor")
parser.add_argument("--smearing-factor", type=int, default=4,
                    help="smearing = cutoff/smearing_factor "
                    "(default 4, matches benchmark)")
parser.add_argument("--max-atoms", type=int, default=32768)
parser.add_argument("--min-atoms", type=int, default=0)
parser.add_argument("--crystals", nargs="+", default=None,
                    help="filter by crystal name (substring match)")
parser.add_argument("--repeats", type=int, default=10)
parser.add_argument("--dataset", required=True,
                    help="path to extxyz file with structures")
parser.add_argument("--cubic-only", action="store_true",
                    help="only run cubic (n x n x n) supercells")
parser.add_argument("--tag", required=True,
                    help="output regime tag (e.g. light, tight); "
                    "writes run/{tag}/{mode}_{crystal}_times.yaml")
args = parser.parse_args()

# crystal slug from dataset filename: nacl_crystals.xyz -> nacl
crystal_slug = re.sub(r"_crystals\.xyz$", "",
                      os.path.basename(args.dataset))
if crystal_slug == os.path.basename(args.dataset):
    crystal_slug = os.path.splitext(os.path.basename(args.dataset))[0]

out_dir = os.path.join("run", args.tag)
os.makedirs(out_dir, exist_ok=True)

# derived hyperparameters
k_param = cutoff / args.factor
smearing = cutoff / args.smearing_factor

# ── data ──────────────────────────────────────────────────────────────

crystals = read(args.dataset, format="extxyz", index=":")

if args.crystals:
    crystals = [
        a for a in crystals
        if any(c in a.info["crystal_name"] for c in args.crystals)
    ]
    print(f"Filtered to {len(crystals)} structures matching {args.crystals}")

if args.cubic_only:
    crystals = [a for a in crystals if a.info.get("is_cubic", False)]
    print(f"Cubic only: {len(crystals)} structures")

crystals = [a for a in crystals if len(a) <= args.max_atoms]
crystals = [a for a in crystals if len(a) >= args.min_atoms]

print(f"{len(crystals)} structures, "
      f"{args.min_atoms}–{args.max_atoms} atoms, "
      f"modes: {args.modes}, repeats: {args.repeats}, "
      f"tag: {args.tag}, crystal: {crystal_slug}")
print(f"cutoff={cutoff}, factor={args.factor} => k_param={k_param:.4f},"
      f" smearing={smearing:.4f} (cutoff/{args.smearing_factor})")

# ── helpers ───────────────────────────────────────────────────────────


def make_batch(atoms):
    from lorem import to_sample, to_batch

    sample = to_sample(
        atoms,
        cutoff,
        energy=False,
        forces=False,
        stress=False,
        lr_wavelength=k_param,
        smearing=smearing,
    )
    from jaxpme.batched_mixed.batching import count_halfspace_kvectors

    n_atoms = len(atoms) + 1
    n_pairs = sample.structure["centers"].shape[0] + 1
    k_shape = sample.structure["lr"].k_grid.shape
    n_k = count_halfspace_kvectors(k_shape)
    batch = to_batch(
        [sample], [],
        batch_size=2,
        shapes={
            "atoms": n_atoms,
            "atoms_pbc": len(atoms),
            "pairs": n_pairs,
            "pairs_nonpbc": 1,
            "pbc": 1,
            "k": max(n_k, 1),
        },
    )
    return jax.tree.map(lambda x: jnp.array(x), batch)


def init_ewald(rng):
    from lorem import Lorem

    model = Lorem(
        cutoff=cutoff,
        max_degree=6,
        initialize_node_features=True,
        lr=True,
    )
    params = model.init(rng, *model.dummy_inputs())
    pred_fn = jax.jit(model.predict)
    return params, pred_fn


def init_pme(rng):
    from lorem_pme import LoremPME, prepare_inputs, predict as pme_predict

    model = LoremPME(
        cutoff=cutoff,
        max_degree=6,
        initialize_node_features=True,
    )
    dummy = prepare_inputs(
        crystals[0], cutoff,
        mesh_spacing=k_param,
        smearing=smearing,
    )
    params = model.init(rng, *dummy)
    pred_fn = jax.jit(
        lambda params, inputs: pme_predict(model, params, inputs)
    )
    return params, pred_fn


def init_ewald_unbatched(rng):
    from lorem_pme import LoremUnbatched, prepare_inputs, predict as ub_predict

    model = LoremUnbatched(
        cutoff=cutoff,
        max_degree=6,
        initialize_node_features=True,
        lr_calculator="ewald",
    )
    dummy = prepare_inputs(
        crystals[0], cutoff,
        mesh_spacing=k_param,
        smearing=smearing,
        mode="ewald",
    )
    params = model.init(rng, *dummy)
    pred_fn = jax.jit(
        lambda params, inputs: ub_predict(model, params, inputs)
    )
    return params, pred_fn


def init_nolr(rng):
    from lorem_pme import LoremUnbatched, prepare_inputs, predict as nolr_predict

    model = LoremUnbatched(
        cutoff=cutoff,
        max_degree=6,
        initialize_node_features=True,
        lr=False,
    )
    dummy = prepare_inputs(
        crystals[0], cutoff,
        mesh_spacing=k_param,
        smearing=smearing,
        mode="ewald",
    )
    params = model.init(rng, *dummy)
    pred_fn = jax.jit(
        lambda params, inputs: nolr_predict(model, params, inputs)
    )
    return params, pred_fn


def prepare_ewald(atoms):
    return jax.device_put(make_batch(atoms))


def prepare_ewald_unbatched(atoms):
    from lorem_pme import prepare_inputs

    inputs = prepare_inputs(
        atoms, cutoff, mesh_spacing=k_param, smearing=smearing,
        mode="ewald",
    )
    k_grid = inputs[6]
    print(f"    k_grid: {k_grid.shape}", flush=True)
    return jax.device_put(inputs)


def prepare_pme(atoms):
    from lorem_pme import prepare_inputs

    inputs = prepare_inputs(
        atoms, cutoff, mesh_spacing=k_param, smearing=smearing,
    )
    k_grid = inputs[6]
    print(f"    k_grid: {k_grid.shape}", flush=True)
    return jax.device_put(inputs)


def prepare_nolr(atoms):
    from lorem_pme import prepare_inputs

    # LR is disabled, k_grid unused; use ewald mode for a small k_grid
    inputs = prepare_inputs(
        atoms, cutoff, mesh_spacing=k_param, smearing=smearing,
        mode="ewald",
    )
    return jax.device_put(inputs)


# ── benchmark loop ────────────────────────────────────────────────────

rng = jax.random.key(42)

for mode in args.modes:
    print(f"\n=== {mode.upper()} (factor={args.factor}) ===\n", flush=True)

    if mode == "ewald":
        params, pred_fn = init_ewald(rng)
        prepare_fn = prepare_ewald
    elif mode == "ewald_unbatched":
        params, pred_fn = init_ewald_unbatched(rng)
        prepare_fn = prepare_ewald_unbatched
    elif mode == "nolr":
        params, pred_fn = init_nolr(rng)
        prepare_fn = prepare_nolr
    else:
        params, pred_fn = init_pme(rng)
        prepare_fn = prepare_pme

    out_path = os.path.join(out_dir, f"{mode}_{crystal_slug}_times.yaml")
    times = {}
    for atoms in crystals:
        name = f"{atoms.info['crystal_name']}_{len(atoms)}"
        try:
            print(f"{name} (N={len(atoms)})...", flush=True)

            inputs = prepare_fn(atoms)

            # warmup (includes jit compilation)
            results = pred_fn(params, inputs)
            results["energy"].block_until_ready()
            results["forces"].block_until_ready()

            start = monotonic()
            for _ in range(args.repeats):
                results = pred_fn(params, inputs)
                results["energy"].block_until_ready()
                results["forces"].block_until_ready()

            time = (monotonic() - start) / args.repeats
            times[name] = time
            print(
                f"  -> {time*1e6/len(atoms):.2f} µs/atom"
                f" ({1e3*time:.2f} ms total)",
                flush=True,
            )

            write_yaml(out_path, times)
        except KeyboardInterrupt:
            raise
        except Exception:
            print(f"  FAILED (N={len(atoms)})")
            traceback.print_exc()
            jax.clear_caches()
            print("  soldiering on...", flush=True)

print("\ndone.")
