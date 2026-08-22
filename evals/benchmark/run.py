"""Benchmark Ewald vs PME long-range summation in Lorem.

Configure via the settings dict below, or override from the command line:
    python run.py                     # run with defaults
    python run.py --modes pme         # PME only
    python run.py --modes ewald pme   # both (default)
    python run.py --max-atoms 1024    # limit atom count
    python run.py --min-atoms 512     # skip small structures
    python run.py --crystals NaCl_cubic wigner_bcc  # specific crystals
    python run.py --repeats 5         # timing repetitions
"""

import argparse
import os
import traceback

import jax
import jax.numpy as jnp
from ase.io import read
from time import monotonic

from marathon.io import write_yaml

os.makedirs("run", exist_ok=True)

# ── settings ──────────────────────────────────────────────────────────

cutoff = 5.0

parser = argparse.ArgumentParser()
parser.add_argument("--modes", nargs="+", default=["ewald", "pme"],
                    choices=["ewald", "pme", "ewald_unbatched", "nolr"])
parser.add_argument("--max-atoms", type=int, default=32768)
parser.add_argument("--min-atoms", type=int, default=0)
parser.add_argument("--crystals", nargs="+", default=None,
                    help="filter by crystal name (substring match)")
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--dataset", default="crystals.xyz",
                    help="path to extxyz file with structures")
parser.add_argument("--cubic-only", action="store_true",
                    help="only run cubic (n x n x n) supercells")
parser.add_argument("--tag", default=None,
                    help="output tag: writes run/{mode}_{tag}_times.yaml "
                    "(default: derived from dataset filename)")
args = parser.parse_args()

# derive tag from dataset name if not given
if args.tag is None:
    import re
    args.tag = re.sub(r"_crystals\.xyz$", "",
                      os.path.basename(args.dataset))
    # fallback for other filenames
    if args.tag == os.path.basename(args.dataset):
        args.tag = os.path.splitext(os.path.basename(args.dataset))[0]

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
      f"tag: {args.tag}")

# ── helpers ───────────────────────────────────────────────────────────


def make_batch(atoms):
    from lorem import to_sample, to_batch

    sample = to_sample(
        atoms,
        cutoff,
        energy=False,
        forces=False,
        stress=False,
        lr_wavelength=cutoff / 8,
        smearing=cutoff / 4,
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
        mesh_spacing=cutoff / 8,
        smearing=cutoff / 4,
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
        mesh_spacing=cutoff / 8,
        smearing=cutoff / 4,
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
        mesh_spacing=cutoff / 8,
        smearing=cutoff / 4,
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
        atoms, cutoff, mesh_spacing=cutoff / 8, smearing=cutoff / 4,
        mode="ewald",
    )
    k_grid = inputs[6]
    print(f"    k_grid: {k_grid.shape}", flush=True)
    return jax.device_put(inputs)


def prepare_pme(atoms):
    from lorem_pme import prepare_inputs

    inputs = prepare_inputs(
        atoms, cutoff, mesh_spacing=cutoff / 8, smearing=cutoff / 4,
    )
    k_grid = inputs[6]
    print(f"    k_grid: {k_grid.shape}", flush=True)
    return jax.device_put(inputs)


def prepare_nolr(atoms):
    from lorem_pme import prepare_inputs

    # LR is disabled, k_grid unused; use ewald mode for a small k_grid
    inputs = prepare_inputs(
        atoms, cutoff, mesh_spacing=cutoff / 8, smearing=cutoff / 4,
        mode="ewald",
    )
    return jax.device_put(inputs)


# ── benchmark loop ────────────────────────────────────────────────────

rng = jax.random.key(42)

for mode in args.modes:
    print(f"\n=== {mode.upper()} ===\n", flush=True)

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

            write_yaml(f"run/{mode}_{args.tag}_times.yaml", times)
        except KeyboardInterrupt:
            raise
        except Exception:
            print(f"  FAILED (N={len(atoms)})")
            traceback.print_exc()
            # clear JIT caches to free memory after OOM
            jax.clear_caches()
            print("  soldiering on...", flush=True)

print("\ndone.")
