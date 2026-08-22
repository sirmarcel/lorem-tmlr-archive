"""Load multi-crystal Ewald vs PME benchmark data (cubic supercells).

Shipped data is the **light** regime of the matched-accuracy re-run.

For provenance, the originally-published timings (mesh_spacing=lr_wavelength
=cutoff/8, unmatched accuracy) live in `benchmark_broken.py` and feed
`benchmark_figure_broken.py` / `benchmark_table_broken.py`.
"""

from tbx import *
from pathlib import Path
import yaml

revdir = Path(__file__).parent / "../evals/benchmark-revision/light"


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

data = {}
for crystal in CRYSTALS:
    d = {}
    for mode, prefix in [
        ("pme", "pme"),
        ("ewald", "ewald_unbatched"),
        ("nolr", "nolr"),
    ]:
        path = revdir / f"{prefix}_{crystal}_times.yaml"
        if path.exists():
            d[mode] = parse_times_simple(read_yaml(path))
        else:
            d[mode] = {}
    data[crystal] = d


def get_table_data(crystal):
    """Return (sizes, ewald, pme, nolr, lr_ewald, lr_pme) arrays.

    sizes: atom counts where both PME and nolr have data.
    For sizes without Ewald data, returns NaN for ewald/lr_ewald.
    """
    d = data[crystal]
    sizes_pme = sorted(
        set(d["pme"].keys()) & set(d["nolr"].keys())
    )
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
