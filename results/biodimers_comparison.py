"""Generate a LaTeX comparison table for the biodimers totals across the
LOREM variants (the per-class indexing bug only affected LOREM evals;
non-LOREM models read the same `metrics.npz` then and now).

Each cell renders as either:
  - a single \\num{value} when old and new agree to 3 d.p., or
  - a \\chg{old}{new} call (small struck-through old in grey, then new),
    so a glance shows what shifted.

Outputs `tables/biodimers_comparison.{tex,md}`. The \\chg macro is shared
with `benchmark_table_comparison.py` and is defined in
`../tex/comparison.tex`.
"""

import numpy as np
from pathlib import Path

from tbx import *
from tbx.tables import *


# Variants to show, in roughly the order they appear in the paper's
# overview tables (main lorem first, then ablations).
VARIANTS = [
    ("lorem", r"\textsc{Lorem}"),
    ("lorem-alt", r"\textsc{Lorem}$^{\dagger}$"),
    ("lorem-l0", r"\textsc{Lorem} LR $l=0$"),
    ("lorem-l0-alt", r"\textsc{Lorem} LR $l=0^{\dagger}$"),
    ("lorem-l1", r"\textsc{Lorem} LR $l=1$"),
    ("lorem-l1-alt", r"\textsc{Lorem} LR $l=1^{\dagger}$"),
    ("lorem-nolr-mp1", r"\textsc{Lorem} no LR"),
    ("lorem-nolr-mp1-alt", r"\textsc{Lorem} no LR$^{\dagger}$"),
]

base = datadir / "bio_dimers"


def load(folder, fname):
    p = base / folder / fname
    if not p.exists():
        return None
    return np.load(p)


def fmt(x, digits=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:.{digits}f}"


def cell(old, new, digits=3):
    so = fmt(old, digits)
    sn = fmt(new, digits)
    if so == sn:
        if so == "--":
            return "--"
        return r"\num{" + so + r"}"
    o_tex = r"\num{" + so + r"}" if so != "--" else "--"
    n_tex = r"\num{" + sn + r"}" if sn != "--" else "--"
    return r"\chg{" + o_tex + r"}{" + n_tex + r"}"


# ----- main: per-variant total energy + forces RMSE / MAE -----
metrics_keys = [
    ("energy_rmse", r"$E$ RMSE (meV/at)"),
    ("energy_mae", r"$E$ MAE (meV/at)"),
    ("forces_rmse", r"$\F$ RMSE (meV/Å)"),
    ("forces_mae", r"$\F$ MAE (meV/Å)"),
]

rows = []
md_rows = []
for variant, label in VARIANTS:
    old = load(variant, "metrics-broken.npz")
    new = load(variant, "metrics.npz")
    if old is None or new is None:
        continue
    cells = []
    md_cells = []
    for k, _ in metrics_keys:
        o = float(old[k])
        n = float(new[k])
        cells.append(cell(o, n, digits=3))
        if fmt(o, 3) == fmt(n, 3):
            md_cells.append(f"{n:>10.3f}")
        else:
            md_cells.append(f"{o:>5.3f} / {n:<5.3f}")
    rows.append([label] + cells)
    md_rows.append(f"| {label:>30s} | " + " | ".join(md_cells) + " |")


width = 26
layout = " ".join(["l", "|"] + ["r"] * len(metrics_keys))

heading = (
    r"  \multicolumn{1}{c}{}"
    rf" & \multicolumn{{{len(metrics_keys)}}}{{c}}"
    r"{Biodimers (test set, total)} \\ "
    + "\n"
)

rows_aligned = map_across(rows, [aligner(width)] * (len(metrics_keys) + 1))
titles = ["Variant"] + [t for _, t in metrics_keys]

table = r"\begin{tabular}{X}".replace("X", layout) + "\n"
table += r"\toprule" + "\n"
table += heading
table += table_to_tex([titles], width=width)
table += r"\midrule" + "\n"
table += table_to_tex(rows_aligned, width=width)
table += r"\bottomrule" + "\n"
table += r"\end{tabular}" + "\n"

print(table)
savefile(table, "tables/biodimers_comparison.tex")

md_header = (
    "| variant | E RMSE (meV/at) | E MAE (meV/at) | "
    "F RMSE (meV/Å) | F MAE (meV/Å) |\n"
    "|--------:|----------------:|---------------:|"
    "---------------:|--------------:|\n"
)
md = md_header + "\n".join(md_rows) + "\n"
savefile(md, "tables/biodimers_comparison.md")
