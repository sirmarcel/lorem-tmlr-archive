"""Comparison figure: original (mismatched-accuracy) Ewald vs PME timings
against the matched-accuracy re-run.

Two side-by-side panels:
  - left: light regime (F_rel ≈ 1.3e-2; Ewald cutoff/4, PME cutoff/8)
  - right: tight regime (F_rel ≈ 2e-4; Ewald cutoff/6, PME cutoff/32)

Each panel overlays original (faded) and revised (bold) timings for all
three crystals.
"""

from pathlib import Path

import yaml

from tbx import *
from benchmark import CRYSTALS, CRYSTAL_LABELS, parse_times_simple


revdir = Path(__file__).parent / "../evals/benchmark-revision"
evaldir = Path(__file__).parent / "../evals/benchmark"


def load_regime(regime):
    """{crystal: {"ewald": {N: t}, "pme": {N: t}}} from data/{regime}/."""
    out = {}
    for crystal in CRYSTALS:
        d = {}
        for mode, prefix in [("pme", "pme"), ("ewald", "ewald_unbatched")]:
            path = revdir / regime / f"{prefix}_{crystal}_times.yaml"
            d[mode] = parse_times_simple(yaml.safe_load(path.read_text()))
        out[crystal] = d
    return out


def load_orig():
    out = {}
    for crystal in CRYSTALS:
        d = {}
        for mode, prefix in [("pme", "pme"), ("ewald", "ewald_unbatched")]:
            path = evaldir / f"{prefix}_{crystal}_times.yaml"
            d[mode] = parse_times_simple(yaml.safe_load(path.read_text()))
        out[crystal] = d
    return out


def sorted_arrays(d):
    Ns = np.array(sorted(d.keys()))
    ts = np.array([d[n] for n in Ns])
    return Ns, ts


crystal_colors = {"nacl": black, "cscl": teal, "cu2o": red}
crystal_markers = {"nacl": bigdot, "cscl": square, "cu2o": diamond}
crystal_ms = {"nacl": 4, "cscl": 3.5, "cu2o": 3.5}

orig = load_orig()
regimes = {"light": load_regime("light"), "tight": load_regime("tight")}
regime_titles = {
    "light": r"light  ($F_{\rm rel}\approx 1.3{\cdot}10^{-2}$)",
    "tight": r"tight  ($F_{\rm rel}\approx 2{\cdot}10^{-4}$)",
}

fig, axes = plt.subplots(
    1, 2, figsize=(textwidth, 3.5), sharey=True,
)

for ax, regime in zip(axes, ["light", "tight"]):
    rev = regimes[regime]

    for crystal in CRYSTALS:
        c = crystal_colors[crystal]
        m = crystal_markers[crystal]
        ms = crystal_ms[crystal]
        label = CRYSTAL_LABELS[crystal]

        # Ewald — original (faded), revised (bold)
        Ns, ts = sorted_arrays(orig[crystal]["ewald"])
        ax.plot(
            Ns, 1e3 * ts,
            color=c, marker=m, markersize=ms,
            linestyle=dashed, linewidth=1.0, alpha=0.35,
            label=f"{label} (Ewald, orig)",
        )
        Ns, ts = sorted_arrays(rev[crystal]["ewald"])
        ax.plot(
            Ns, 1e3 * ts,
            color=c, marker=m, markersize=ms,
            linestyle=dashed, linewidth=1.6,
            label=f"{label} (Ewald, revised)",
        )

        # PME — original (faded), revised (bold)
        Ns, ts = sorted_arrays(orig[crystal]["pme"])
        ax.plot(
            Ns, 1e3 * ts,
            color=c, marker=m, markersize=ms,
            linestyle=solid, linewidth=1.0, alpha=0.35,
            label=f"{label} (PME, orig)",
        )
        Ns, ts = sorted_arrays(rev[crystal]["pme"])
        ax.plot(
            Ns, 1e3 * ts,
            color=c, marker=m, markersize=ms,
            linestyle=solid, linewidth=1.6,
            label=f"{label} (PME, revised)",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Number of atoms $N$")
    ax.set_title(regime_titles[regime], fontsize=10)
    ax.set_ylim(0.4, 2000)

axes[0].set_ylabel("Inference time (ms)")

# Single shared legend, only NaCl entries (orig vs revised, ewald vs pme)
# plus crystal markers via dummy entries.
from matplotlib.lines import Line2D

style_handles = [
    Line2D([0], [0], color="black", linestyle=dashed,
           linewidth=1.0, alpha=0.35, label="Ewald (orig)"),
    Line2D([0], [0], color="black", linestyle=dashed,
           linewidth=1.6, label="Ewald (revised)"),
    Line2D([0], [0], color="black", linestyle=solid,
           linewidth=1.0, alpha=0.35, label="PME (orig)"),
    Line2D([0], [0], color="black", linestyle=solid,
           linewidth=1.6, label="PME (revised)"),
]
crystal_handles = [
    Line2D([0], [0], color=crystal_colors[c], marker=crystal_markers[c],
           linestyle="", markersize=crystal_ms[c],
           label=CRYSTAL_LABELS[c])
    for c in CRYSTALS
]

axes[0].legend(
    handles=style_handles, fontsize=7, loc="upper left",
)
axes[1].legend(
    handles=crystal_handles, fontsize=7, loc="upper left",
)

savefig(fig, "figures/benchmark_comparison")
