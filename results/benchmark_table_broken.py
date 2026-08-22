"""Provenance copy: regenerates the originally-published Ewald vs PME LR
breakdown tables from the pre-revision (mismatched-accuracy) timings."""

from tbx import *
from tbx.tables import *
from benchmark_broken import (
    CRYSTALS,
    CRYSTAL_LABELS,
    get_table_data,
)

# Sizes to show in the paper (rest are commented out in LaTeX).
# Pick a representative logarithmic spread for each crystal.
SHOWN = {
    "nacl": {8, 64, 512, 1728, 4096, 8000, 17576, 32768},
    "cscl": {2, 54, 432, 1458, 4394, 8192, 18522, 31250},
    "cu2o": {6, 48, 384, 1296, 4374, 7986, 16464, 29478},
}


def make_table(crystal, label):
    """Build LaTeX + markdown table for one crystal type."""
    sizes, t_ewald, t_pme, t_nolr, lr_ewald, lr_pme = get_table_data(
        crystal
    )
    shown = SHOWN[crystal]

    tex_lines = []
    md_rows = []
    for i, n in enumerate(sizes):
        n = int(n)
        te = t_ewald[i]
        tp = t_pme[i]
        le = lr_ewald[i]
        lp = lr_pme[i]

        pct_p = 100 * lp / tp

        if np.isnan(te):
            nums = [
                "--",
                "--",
                "--",
                f"{1e3 * tp:.1f}",
                f"{1e3 * lp:.1f}",
                f"{pct_p:.1f}",
            ]
            md_rows.append(
                f"| {n:>5} "
                f"|        -- |       -- |     -- "
                f"| {1e3*tp:>9.1f} | {1e3*lp:>8.1f} | {pct_p:>6.1f} |"
            )
        else:
            pct_e = 100 * le / te
            nums = [
                f"{1e3 * te:.1f}",
                f"{1e3 * le:.1f}",
                f"{pct_e:.1f}",
                f"{1e3 * tp:.1f}",
                f"{1e3 * lp:.1f}",
                f"{pct_p:.1f}",
            ]
            md_rows.append(
                f"| {n:>5} "
                f"| {1e3*te:>10.1f} | {1e3*le:>8.1f} | {pct_e:>6.1f} "
                f"| {1e3*tp:>9.1f} | {1e3*lp:>8.1f} | {pct_p:>6.1f} |"
            )

        nums_tex = map_across([nums], lambda x: to_num(x))[0]
        row_tex = table_to_tex([[f"{n}", *nums_tex]], width=14).rstrip(
            "\n"
        )
        if n not in shown:
            row_tex = "% " + row_tex
        tex_lines.append(row_tex)

    # LaTeX table
    layout = " ".join(["r", "|"] + ["r"] * 3 + ["|"] + ["r"] * 3)
    heading = (
        r"  \multicolumn{1}{c}{}"
        r" & \multicolumn{3}{c}{Ewald}"
        r" & \multicolumn{3}{c}{PME}"
        r" \\ " + "\n"
    )

    titles_tex = table_to_tex(
        [
            [
                r"$N$",
                r"SR+LR (ms)",
                r"LR (ms)",
                r"LR (\%)",
                r"SR+LR (ms)",
                r"LR (ms)",
                r"LR (\%)",
            ]
        ],
        width=14,
    )

    table = r"\begin{tabular}{X}".replace("X", layout) + "\n"
    table += r"\toprule" + "\n"
    table += heading
    table += titles_tex
    table += r"\midrule" + "\n"
    table += "\n".join(tex_lines) + "\n"
    table += r"\bottomrule" + "\n"
    table += r"\end{tabular}" + "\n"

    # Markdown table
    md_header = (
        "| N | Ewald (ms) | Ewald LR (ms) | Ewald LR (%) "
        "| PME (ms) | PME LR (ms) | PME LR (%) |\n"
    )
    md_sep = (
        "|------:|-----------:|--------------:|-------------:"
        "|---------:|------------:|-----------:|\n"
    )
    md = md_header + md_sep + "\n".join(md_rows) + "\n"

    return table, md


# Generate tables for each crystal
for crystal in CRYSTALS:
    label = CRYSTAL_LABELS[crystal]
    tex, md = make_table(crystal, label)

    print(f"\n=== {label} ===\n")
    print(tex)
    savefile(tex, f"tables/benchmark_{crystal}_broken.tex")

    print(md)
    savefile(md, f"tables/benchmark_{crystal}_broken.md")
