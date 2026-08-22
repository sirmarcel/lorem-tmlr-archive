"""Light-only Ewald vs PME timing comparison: published vs revised.

For the data-share / comparison document. Two side-by-side panels — one
per method (Ewald, PME) — overlaying the originally-published timings
(faded, dashed/solid as in the paper) against the matched-accuracy
re-run at the **light** settings (bold). All three crystals shown per
panel.

Capped at the originally-submitted paper's N range (~33k) so both sides
cover the same ground; the un-capped data archive view is in
`benchmark_full_figure.py`.
"""

from pathlib import Path

import yaml

from tbx import *
from benchmark import CRYSTALS, CRYSTAL_LABELS, parse_times_simple


revdir = Path(__file__).parent / "../evals/benchmark-revision/light"
evaldir = Path(__file__).parent / "../evals/benchmark"

N_MAX = 33000


def load(srcdir):
    out = {}
    for crystal in CRYSTALS:
        d = {}
        for mode, prefix in [("pme", "pme"), ("ewald", "ewald_unbatched")]:
            path = srcdir / f"{prefix}_{crystal}_times.yaml"
            d[mode] = parse_times_simple(yaml.safe_load(path.read_text()))
        out[crystal] = d
    return out


def sorted_arrays(d, n_max=None):
    Ns = sorted(d.keys())
    if n_max is not None:
        Ns = [n for n in Ns if n <= n_max]
    Ns = np.array(Ns)
    ts = np.array([d[int(n)] for n in Ns])
    return Ns, ts


crystal_colors = {"nacl": black, "cscl": teal, "cu2o": red}
crystal_markers = {"nacl": bigdot, "cscl": square, "cu2o": diamond}
crystal_ms = {"nacl": 4, "cscl": 3.5, "cu2o": 3.5}

orig = load(evaldir)
rev = load(revdir)

method_styles = {"ewald": dashed, "pme": solid}
method_titles = {"ewald": "Ewald", "pme": "PME"}

fig, axes = plt.subplots(
    1, 2, figsize=(textwidth, 3.5), sharey=True,
)

for ax, method in zip(axes, ["ewald", "pme"]):
    ls = method_styles[method]
    for crystal in CRYSTALS:
        c = crystal_colors[crystal]
        m = crystal_markers[crystal]
        ms = crystal_ms[crystal]
        label = CRYSTAL_LABELS[crystal]

        Ns, ts = sorted_arrays(orig[crystal][method], n_max=N_MAX)
        ax.plot(
            Ns, 1e3 * ts,
            color=c, marker=m, markersize=ms,
            linestyle=ls, linewidth=1.0, alpha=0.35,
            label=f"{label} (published)",
        )
        Ns, ts = sorted_arrays(rev[crystal][method], n_max=N_MAX)
        ax.plot(
            Ns, 1e3 * ts,
            color=c, marker=m, markersize=ms,
            linestyle=ls, linewidth=1.6,
            label=f"{label} (revised)",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Number of atoms $N$")
    ax.set_title(method_titles[method], fontsize=10)
    ax.set_ylim(0.4, 2000)

axes[0].set_ylabel("Inference time (ms)")

# Shared legend in the left panel: just crystals × (published/revised);
# method is conveyed by the panel title.
from matplotlib.lines import Line2D

style_handles = [
    Line2D([0], [0], color="black", linestyle="-",
           linewidth=1.0, alpha=0.35, label="published"),
    Line2D([0], [0], color="black", linestyle="-",
           linewidth=1.6, label="revised"),
]
crystal_handles = [
    Line2D([0], [0], color=crystal_colors[c], marker=crystal_markers[c],
           linestyle="", markersize=crystal_ms[c],
           label=CRYSTAL_LABELS[c])
    for c in CRYSTALS
]

axes[0].legend(handles=style_handles, fontsize=7, loc="upper left")
axes[1].legend(handles=crystal_handles, fontsize=7, loc="upper left")

savefig(fig, "figures/benchmark_comparison_light")
