import numpy as np

from marathon.data import datasets
from ase.io import read

from marathon import comms
from marathon.extra.hermes.data_source import prepare, DataSource


# The heaviest training .xyz files are excluded from the git repository (the
# complete archive lives on Zenodo); skip their blocks gracefully if absent.
def have(*names):
    gone = [n for n in names if not (datasets / n).exists()]
    if gone:
        print(f"skipping: {', '.join(gone)} not present (see README.md for provenance)")
    return not gone


# -- AuMgO --
if have("AuMgO_train.xyz"):
    reporter = comms.reporter()
    reporter.start("processing AuMgO")

    reporter.step("load")
    train = read(datasets / "AuMgO_train.xyz", format="extxyz", index=":")
    valid = read(datasets / "AuMgO_valid.xyz", format="extxyz", index=":")

    reporter.step("process")
    prepare(
        train,
        folder=datasets / "AuMgO_train",
        reporter=reporter,
        batch_size=500,
        samples_per_composition=100,
    )

    prepare(
        valid,
        folder=datasets / "AuMgO_valid",
        reporter=reporter,
        batch_size=500,
        samples_per_composition=100,
    )

    reporter.done()

# -- bio_dimers --

reporter = comms.reporter()
reporter.start("processing bio_dimers")

reporter.step("load")
train = read(datasets / "bio_dimers_train.xyz", format="extxyz", index=":")
valid = read(datasets / "bio_dimers_valid.xyz", format="extxyz", index=":")

reporter.step("process")
prepare(
    train,
    folder=datasets / "bio_dimers_train",
    reporter=reporter,
    batch_size=500,
    samples_per_composition=100,
)

prepare(
    valid,
    folder=datasets / "bio_dimers_valid",
    reporter=reporter,
    batch_size=500,
    samples_per_composition=100,
)

reporter.done()

# -- cumulene --

reporter = comms.reporter()
reporter.start("processing cumulene")

reporter.step("load")
train = read(datasets / "cumulene_train.xyz", format="extxyz", index=":")
valid = read(datasets / "cumulene_valid.xyz", format="extxyz", index=":")


reporter.step("process")
prepare(
    train,
    folder=datasets / "cumulene_train",
    reporter=reporter,
    batch_size=500,
    samples_per_composition=100,
)

prepare(
    valid,
    folder=datasets / "cumulene_valid",
    reporter=reporter,
    batch_size=500,
    samples_per_composition=100,
)

reporter.done()

# -- NaCl --
if have("NaCl_train.xyz"):
    reporter = comms.reporter()
    reporter.start("processing NaCl")

    reporter.step("load")
    train = read(datasets / "NaCl_train.xyz", format="extxyz", index=":")
    valid = read(datasets / "NaCl_valid.xyz", format="extxyz", index=":")

    reporter.step("process")
    prepare(
        train,
        folder=datasets / "NaCl_train",
        reporter=reporter,
        batch_size=500,
        samples_per_composition=100,
    )

    prepare(
        valid,
        folder=datasets / "NaCl_valid",
        reporter=reporter,
        batch_size=500,
        samples_per_composition=100,
    )

    reporter.done()

# -- SN2 --
if have("sn2_train.xyz"):
    reporter = comms.reporter()
    reporter.start("processing SN2")

    reporter.step("load")
    train = read(datasets / "sn2_train.xyz", format="extxyz", index=":")
    valid = read(datasets / "sn2_valid.xyz", format="extxyz", index=":")

    reporter.step("process")
    prepare(
        train,
        folder=datasets / "sn2_train",
        reporter=reporter,
        batch_size=500,
        samples_per_composition=100,
    )

    prepare(
        valid,
        folder=datasets / "sn2_valid",
        reporter=reporter,
        batch_size=500,
        samples_per_composition=100,
    )

    reporter.done()

# -- ADAPT --
#
# The training xyz is not redistributed here (~5.5 GB). Download
# dataset_train_pbc_updated.xyz from https://doi.org/10.5281/zenodo.17411327
# and place it under $DATASETS/adapt/ before running this block.
#
# The train/valid split is np.random.default_rng(23) + shuffle + first 20% as
# valid; see README.md. The official 100-structure test set is distributed as
# adapt_test_official.xyz in this folder.
#
# For the HP sweep, a 10,000-structure debug subset (9,500 train / 500 valid)
# was used; see prepare_adapt_debug() below.

from ase.calculators.singlepoint import SinglePointCalculator


def prepare_adapt(train_all, name, seed, valid_fraction, reporter):
    """Split and prepare ADAPT train/valid sets."""
    rng = np.random.default_rng(seed)
    indices = np.arange(len(train_all))
    rng.shuffle(indices)
    n_valid = int(len(train_all) * valid_fraction)
    valid = [train_all[i] for i in indices[:n_valid]]
    train = [train_all[i] for i in indices[n_valid:]]

    prefix = f"adapt/{name}_" if name else "adapt/"
    reporter.step(f"process {name or 'full'}")
    prepare(
        train,
        folder=datasets / f"{prefix}train",
        reporter=reporter,
        batch_size=500,
        samples_per_composition=1000,
    )
    prepare(
        valid,
        folder=datasets / f"{prefix}valid",
        reporter=reporter,
        batch_size=500,
        samples_per_composition=1000,
    )
    return train, valid


reporter = comms.reporter()
reporter.start("processing ADAPT")

if have("adapt/dataset_train_pbc_updated.xyz"):
    reporter.step("load")
    train_all = read(
        datasets / "adapt/dataset_train_pbc_updated.xyz",
        format="extxyz",
        index=":",
    )

    # rename REF_* labels and attach a SinglePointCalculator
    for at in train_all:
        at.arrays["forces"] = at.arrays["REF_forces"]
        at.info["energy"] = at.info["REF_energy"]
        at.calc = SinglePointCalculator(
            at, energy=at.info["energy"], forces=at.arrays["forces"]
        )

    # -- full dataset: seed=23, 20% valid --
    prepare_adapt(train_all, "", seed=23, valid_fraction=0.20, reporter=reporter)

    # -- debug subset: 10,000 structures, seed=0, 5% valid --
    # used for HP sweep (24 combinations, 100 epochs; see paper Appendix J)
    rng_debug = np.random.default_rng(0)
    debug_indices = np.arange(len(train_all))
    rng_debug.shuffle(debug_indices)
    train_debug = [train_all[i] for i in debug_indices[:10000]]
    prepare_adapt(train_debug, "debug", seed=0, valid_fraction=0.05, reporter=reporter)

# -- test set (adapt_test_official.xyz ships with the archive) --
test = read(datasets / "adapt_test_official.xyz", format="extxyz", index=":")
prepare(
    test,
    folder=datasets / "adapt/test_official",
    reporter=reporter,
    batch_size=100,
    samples_per_composition=100,
)

reporter.done()
