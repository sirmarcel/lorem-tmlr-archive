"""Provenance copy: regenerates the originally-published Ewald vs PME
scaling figure from the pre-revision (mismatched-accuracy) timings."""

from tbx import *
from benchmark_broken import CRYSTALS, CRYSTAL_LABELS, get_table_data

crystal_colors = {"nacl": black, "cscl": teal, "cu2o": red}
crystal_markers = {"nacl": bigdot, "cscl": square, "cu2o": diamond}
crystal_ms = {"nacl": 4, "cscl": 3.5, "cu2o": 3.5}

fig, ax = fig_and_ax(figsize=(textwidth, 3.5))

# Plot each crystal
for crystal in CRYSTALS:
    sizes, t_ewald, t_pme, t_nolr, lr_ewald, lr_pme = get_table_data(
        crystal
    )
    label = CRYSTAL_LABELS[crystal]
    c = crystal_colors[crystal]
    m = crystal_markers[crystal]
    ms = crystal_ms[crystal]

    # Ewald total (only where available)
    ewald_mask = ~np.isnan(t_ewald)
    ax.plot(
        sizes[ewald_mask],
        1e3 * t_ewald[ewald_mask],
        color=c,
        marker=m,
        markersize=ms,
        linestyle=dashed,
        linewidth=1.0,
        label=f"{label} (Ewald)",
    )

    # PME total
    ax.plot(
        sizes,
        1e3 * t_pme,
        color=c,
        marker=m,
        markersize=ms,
        linestyle=solid,
        linewidth=1.5,
        label=f"{label} (PME)",
    )

# Reference lines anchored to data at N=1000
# Ewald NaCl at N=1000: ~15ms; PME NaCl at N=1000: ~3ms
n_ref = np.array([200, 40000])
t_n2 = 15 * (n_ref / 1000) ** 2
t_n1 = 3 * (n_ref / 1000) ** 1
ax.plot(
    n_ref, t_n2, color=grey, linestyle=dotted, linewidth=0.8,
    label=r"$\propto N^2$",
)
ax.plot(
    n_ref, t_n1, color=grey, linestyle=finedash, linewidth=0.8,
    label=r"$\propto N$",
)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Number of atoms $N$")
ax.set_ylabel("Inference time (ms)")
ax.set_ylim(0.4, 2000)
ax.legend(fontsize=7, ncol=2)

savefig(fig, "figures/benchmark_scaling_broken")
