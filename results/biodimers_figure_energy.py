from tbx import *
from biodimers import metrics, classes, class_energy_std, class_power_label
from metrics import metrics as all_metrics

keys = [
    "lorem",
    "cace",
    "MACE",
    "NativePET-SR",
]

model_names = {
    "lorem": r"\textsc{Lorem}",
    "cace": r"\textsc{Cace-Les}",
    "MACE": r"\textsc{Mace}",
    "NativePET-SR": r"\textsc{Pet}",
}

num_bars = len(keys)
num_classes = len(classes)

width = 0.15

std_scale = 0.1
std_top = max(class_energy_std.values()) * std_scale
ax_top = max(21, std_top * 1.08)

fig, ax = plt.subplots(figsize=(textwidth, 2.8))

patterns = ["XXX", "OOO", "////", "***"]
for i, key in enumerate(keys):
    for j, c in enumerate(classes):
        label = model_names[key] if j == 0 else nolabel
        x = j - (width * num_bars / 2) + i * width + width / 2
        ax.bar(
            x,
            metrics[key][c]["energy_mae"],
            width,
            color=model_colors[key],
            hatch=patterns[i],
            label=label,
        )

group_half = width * num_bars / 2
for j, c in enumerate(classes):
    ax.hlines(
        class_energy_std[c] * std_scale,
        j - group_half,
        j + group_half,
        colors="black",
        linewidth=2.5,
        zorder=10,
    )
from matplotlib.lines import Line2D
std_handle = Line2D([0], [0], color="black", linewidth=2.5, label=r"$0.1 \times $ std of $E$")

ax.set_ylim(0, ax_top)

ax.set_ylabel("Energy MAE (meV/atom)")
ax.set_xlabel("Dimer class", labelpad=12)
ax.set_xticks(np.arange(num_classes))
ax.set_xticklabels(classes)
for j, c in enumerate(classes):
    ax.annotate(
        class_power_label(c),
        xy=(j, 0), xycoords=("data", "axes fraction"),
        xytext=(0, -14), textcoords="offset points",
        ha="center", va="top", fontsize=9,
    )

major_ticks_every(ax, 5, direction="y")
minor_ticks_every(ax, 1, direction="y")

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles + [std_handle], labels + [std_handle.get_label()], loc="upper right")

savefig(fig, "figures/biodimers_energy")
