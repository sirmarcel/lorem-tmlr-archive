"""Comparison figure for the main biodimers MAE plot: adds an extra,
faded \\textsc{Lorem} bar showing the originally-published, broken-indexing
per-class force MAE alongside the corrected one. The other three models
(\\textsc{Cace-Les}, \\textsc{Mace}, \\textsc{Pet}) are unchanged by the
fix, so they appear once. Class ordering follows the revised paper
(CC, CP, PP, CA, PA, AA, ordered by power-law exponent).
"""

from tbx import *
from biodimers import metrics as metrics_new, classes, class_force_std, class_power_label
from biodimers_broken import metrics as metrics_old

# (display key, color, hatch, metrics dict, model lookup, alpha, label)
bars = [
    ("lorem-pub", red, "XXX", metrics_old, "lorem", 0.40,
     r"\textsc{Lorem} (published)"),
    ("lorem", red, "XXX", metrics_new, "lorem", 1.00,
     r"\textsc{Lorem} (revised)"),
    ("cace", orange, "OOO", metrics_new, "cace", 1.00,
     r"\textsc{Cace-Les}"),
    ("MACE", blue, "////", metrics_new, "MACE", 1.00,
     r"\textsc{Mace}"),
    ("NativePET-SR", teal, "***", metrics_new, "NativePET-SR", 1.00,
     r"\textsc{Pet}"),
]

num_bars = len(bars)
num_classes = len(classes)

width = 0.13
break_at = 2.75
restart_at = break_at

std_top = max(class_force_std.values())
ax1_top = max(28, std_top * 1.08)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(textwidth, 3.6))
fig.subplots_adjust(hspace=0.07)

break_x = []
for i, (_key, color, hatch, mdict, mkey, alpha, lab) in enumerate(bars):
    for j, c in enumerate(classes):
        label = lab if j == 0 else nolabel
        x = j - (width * num_bars / 2) + i * width + width / 2
        v = mdict[mkey][c]["forces_mae"]
        for ax in (ax1, ax2):
            ax.bar(
                x, v, width,
                color=color, hatch=hatch, alpha=alpha, label=label,
            )
        if v > break_at:
            break_x.append(x)


# per-class std reference bars
group_half = width * num_bars / 2
for j, c in enumerate(classes):
    std = class_force_std[c]
    target = ax1 if std >= break_at else ax2
    target.hlines(
        std,
        j - group_half,
        j + group_half,
        colors="black",
        linewidth=2.5,
        zorder=10,
    )

from matplotlib.lines import Line2D
std_handle = Line2D(
    [0], [0], color="black", linewidth=2.5, label=r"std of $\mathbf{F}$"
)


ax2.set_ylim(0, break_at)
ax1.set_ylim(restart_at, ax1_top)

ax1.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False)
ax2.xaxis.tick_bottom()


d = 0.24
kwargs = dict(
    marker=[(-1, -d), (1, d)], markersize=10,
    linestyle="none", color="k", mec="k", mew=1.0, clip_on=False,
)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

kwargs = dict(
    marker=[(-1, -d), (1, d)], markersize=10,
    linestyle="none", color="k", mec="k", mew=1.7, clip_on=False,
)
x = np.array(break_x)
ax2.plot(x, np.ones_like(x) * break_at, **kwargs)
ax1.plot(x, np.ones_like(x) * restart_at, **kwargs)

ax2.set_ylabel("Forces MAE (meV/Å)")
ax2.set_xlabel("Dimer class", labelpad=12)
ax2.set_xticks(np.arange(num_classes))
ax2.set_xticklabels(classes)
for j, c in enumerate(classes):
    ax2.annotate(
        class_power_label(c),
        xy=(j, 0), xycoords=("data", "axes fraction"),
        xytext=(0, -14), textcoords="offset points",
        ha="center", va="top", fontsize=9,
    )


major_ticks_every(ax1, 10, direction="y")
minor_ticks_every(ax1, 5, direction="y")
major_ticks_every(ax2, 1, direction="y")
minor_ticks_every(ax2, 0.5, direction="y")

ax2.yaxis.set_label_coords(-0.07, 1)

from matplotlib.patches import Patch

handles, labels = ax1.get_legend_handles_labels()
# Column 1: just the two LOREM bars; column 2: the other models + std.
# Pad column 1 with invisible entries so matplotlib's column-major layout
# (ncol=2 with 6 entries -> 3+3, but here we want 2+4) puts CACE/MACE/PET/std
# in column 2.
blank = Patch(visible=False, label="")
ax1.legend(
    handles[:2] + [blank, blank] + handles[2:] + [std_handle],
    labels[:2] + ["", ""] + labels[2:] + [std_handle.get_label()],
    loc="upper right",
    ncol=2,
    columnspacing=1.0,
    handletextpad=0.4,
)

savefig(fig, "figures/biodimers_comparison")
