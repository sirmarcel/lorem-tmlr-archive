"""Full-coverage Ewald vs PME scaling figure for the data archive.

Plots all N from `../evals/benchmark-revision/light/`, with no
cap — NaCl PME extends to ~85k atoms, CsCl/Cu2O likewise.
"""

from tbx import *
from benchmark import CRYSTALS, CRYSTAL_LABELS, get_table_data

crystal_colors = {"nacl": black, "cscl": teal, "cu2o": red}
crystal_markers = {"nacl": bigdot, "cscl": square, "cu2o": diamond}
crystal_ms = {"nacl": 4, "cscl": 3.5, "cu2o": 3.5}

fig, ax = fig_and_ax(figsize=(textwidth, 3.5))

for crystal in CRYSTALS:
    sizes, t_ewald, t_pme, t_nolr, lr_ewald, lr_pme = get_table_data(crystal)
    label = CRYSTAL_LABELS[crystal]
    c = crystal_colors[crystal]
    m = crystal_markers[crystal]
    ms = crystal_ms[crystal]

    ewald_mask = ~np.isnan(t_ewald)
    ax.plot(
        sizes[ewald_mask],
        1e3 * t_ewald[ewald_mask],
        color=c, marker=m, markersize=ms,
        linestyle=dashed, linewidth=1.0,
        label=f"{label} (Ewald)",
    )

    ax.plot(
        sizes,
        1e3 * t_pme,
        color=c, marker=m, markersize=ms,
        linestyle=solid, linewidth=1.5,
        label=f"{label} (PME)",
    )

# Reference lines anchored to NaCl light at N=1000 (Ewald ≈ 4.5 ms,
# PME ≈ 3 ms).
n_ref = np.array([200, 100000])
t_n2 = 4.5 * (n_ref / 1000) ** 2
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

savefig(fig, "figures/benchmark_full_scaling")
