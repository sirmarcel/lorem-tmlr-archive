"""Provenance copy: loads the originally-published Ewald vs PME benchmark
data (mesh_spacing = lr_wavelength = cutoff/8). The accuracy-matched re-run
(collected into `../evals/benchmark-revision/`) is now the shipped data via
`benchmark.py`; this loader is kept so the historical figure and table stay
reproducible."""

from tbx import *
from pathlib import Path
import yaml

evaldir = Path(__file__).parent / "../evals/benchmark"


def read_yaml(filename):
    with open(filename) as f:
        return yaml.safe_load(f)


def parse_times_simple(raw):
    """Parse {Name_N: time} -> {N: time} dict (single crystal type)."""
    result = {}
    for k, v in raw.items():
        n = int(k.rsplit("_", 1)[1])
        result[n] = v
    return result


CRYSTALS = ["nacl", "cscl", "cu2o"]
CRYSTAL_LABELS = {"nacl": "NaCl", "cscl": "CsCl", "cu2o": r"Cu$_2$O"}

# Load all data: {crystal: {"pme": {N: t}, "ewald": {N: t}, "nolr": {N: t}}}
data = {}
for crystal in CRYSTALS:
    d = {}
    for mode, prefix in [
        ("pme", "pme"),
        ("ewald", "ewald_unbatched"),
        ("nolr", "nolr"),
    ]:
        path = evaldir / f"{prefix}_{crystal}_times.yaml"
        if path.exists():
            d[mode] = parse_times_simple(read_yaml(path))
        else:
            d[mode] = {}
    data[crystal] = d


def get_table_data(crystal):
    """Return (sizes, ewald, pme, nolr, lr_ewald, lr_pme) arrays.

    sizes: atom counts where all three modes have data
    For sizes without Ewald data, returns NaN for ewald/lr_ewald.
    """
    d = data[crystal]
    # sizes where both PME and nolr are available
    sizes_pme = sorted(
        set(d["pme"].keys()) & set(d["nolr"].keys())
    )
    # ewald may have fewer sizes (capped at 8000)
    ewald_sizes = set(d["ewald"].keys())

    sizes = np.array(sizes_pme)
    t_pme = np.array([d["pme"][n] for n in sizes_pme])
    t_nolr = np.array([d["nolr"][n] for n in sizes_pme])
    t_ewald = np.array([
        d["ewald"][n] if n in ewald_sizes else np.nan
        for n in sizes_pme
    ])

    lr_pme = t_pme - t_nolr
    lr_ewald = t_ewald - t_nolr

    return sizes, t_ewald, t_pme, t_nolr, lr_ewald, lr_pme
