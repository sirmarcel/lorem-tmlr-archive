from tbx import *
from biodimers import metrics, classes, class_force_std, class_power_label
from metrics import metrics as all_metrics

model_names = {
    "lorem-nolr-mp1": r"\textsc{Lorem}, No LR",
    "lorem-l0": r"\textsc{Lorem}, LR $l=0$",
    "lorem-l1": r"\textsc{Lorem}, LR $l=1$",
    "lorem": r"\textsc{Lorem}, LR $l=2$",
}



model_colors = {
    "lorem": red,
    "lorem-l1": teal,
    "lorem-l0": orange,
    "lorem-nolr-mp1": blue,
}


keys = model_names.keys()

num_bars = len(keys)
num_classes = len(classes)

width = 0.15
break_at = 2.75
restart_at = break_at

std_top = max(class_force_std.values())
ax1_top = max(28, std_top * 1.08)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(textwidth, 3.85))
fig.subplots_adjust(hspace=0.07)  # adjust space between Axes

patterns = ["XXX", "OOO", "////", "***"]
break_x = []
for i, key in enumerate(keys):
    for j, c in enumerate(classes):
        if j == 0:
            label = model_names[key]
            # ax.axhline(all_metrics["bio_dimers"][key]["energy_mae"])
        else:
            label = nolabel
        x = j - (width * num_bars / 2) + i * width + width / 2
        ax1.bar(
            x,
            metrics[key][c]["forces_mae"],
            width,
            color=model_colors[key],
            hatch=patterns[i],
            label=label,
        )
        ax2.bar(
            x,
            metrics[key][c]["forces_mae"],
            width,
            color=model_colors[key],
            hatch=patterns[i],
            label=label,
        )

        if metrics[key][c]["forces_mae"] > break_at:
            break_x.append(x)


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
std_handle = Line2D([0], [0], color="black", linewidth=2.5, label=r"std of $|\mathbf{F}|$")


# zoom-in / limit the view to different portions of the data
ax2.set_ylim(0, break_at)  # outliers only
ax1.set_ylim(restart_at, ax1_top)  # most of the data

# hide the spines between ax and ax2
ax1.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()


d = .24  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=10,
              linestyle="none", color='k', mec='k', mew=1.0, clip_on=False)

ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

kwargs = dict(marker=[(-1, -d), (1, d)], markersize=10,
              linestyle="none", color='k', mec='k', mew=1.7, clip_on=False)

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

ax2.yaxis.set_label_coords(-0.07,1)

handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles + [std_handle], labels + [std_handle.get_label()], loc="upper right")

savefig(fig, "figures/biodimers_supp_lorem_l012")
