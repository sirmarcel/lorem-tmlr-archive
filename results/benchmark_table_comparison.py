"""Generate per-crystal LaTeX tables comparing the originally-published
benchmark timings (mismatched-accuracy, Feb-2026 epoch) against the
revised, matched-accuracy May-2026 numbers.

Each cell is either:
  - a single \\num{value} when old and new agree (to 1 decimal place), or
  - a \\chg{old}{new} call: a small struck-through old value in grey
    followed by the new value, so a glance shows what shifted.

Sizes shown are the union of the originally-shown set (`SHOWN_OLD`) and
the revised shown set (`SHOWN_NEW`) so every paper-relevant row is
present. Rows where one side has no Ewald data (-- vs ms) are also
flagged via \\chg.

Outputs `tables/benchmark_{crystal}_comparison.{tex,md}`. The companion
tex header (defining \\chg) lives in `../tex/comparison.tex`.
"""

from tbx import *
from tbx.tables import *

import benchmark
import benchmark_broken
import benchmark_table
import benchmark_table_broken

CRYSTALS = benchmark.CRYSTALS
CRYSTAL_LABELS = benchmark.CRYSTAL_LABELS

SHOWN_OLD = benchmark_table_broken.SHOWN
SHOWN_NEW = benchmark_table.SHOWN


def fmt(x):
    """Format a ms timing as a 1-decimal string, or '--' for NaN."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:.1f}"


def cell(old, new):
    """One LaTeX cell.

    Returns plain \\num{...} when unchanged (or '--' when both missing);
    otherwise wraps in \\chg{old}{new}.
    """
    so = fmt(old)
    sn = fmt(new)
    if so == sn:
        if so == "--":
            return "--"
        return r"\num{" + so + r"}"
    o_tex = r"\num{" + so + r"}" if so != "--" else "--"
    n_tex = r"\num{" + sn + r"}" if sn != "--" else "--"
    return r"\chg{" + o_tex + r"}{" + n_tex + r"}"


def get_row(d, n):
    """Return ms-scaled (te, le, pct_e, tp, lp, pct_p) for size n.

    Missing -> NaN. d comes from {pme,ewald,nolr} -> {N: t_seconds}.
    """
    pme = d["pme"].get(n)
    ewald = d["ewald"].get(n)
    nolr = d["nolr"].get(n)

    if pme is None or nolr is None:
        return None  # no row

    tp = 1e3 * pme
    lp = 1e3 * (pme - nolr)
    pct_p = 100 * (pme - nolr) / pme if pme else np.nan

    if ewald is None:
        te = le = pct_e = np.nan
    else:
        te = 1e3 * ewald
        le = 1e3 * (ewald - nolr)
        pct_e = 100 * (ewald - nolr) / ewald if ewald else np.nan

    return te, le, pct_e, tp, lp, pct_p


def make_table(crystal):
    d_old = benchmark_broken.data[crystal]
    d_new = benchmark.data[crystal]

    sizes_all = sorted(
        set(d_old["pme"].keys())
        | set(d_new["pme"].keys())
    )
    shown = SHOWN_OLD[crystal] | SHOWN_NEW[crystal]

    tex_lines = []
    md_rows = []
    for n in sizes_all:
        old = get_row(d_old, n)
        new = get_row(d_new, n)
        if old is None and new is None:
            continue

        # Pad missing side with all-NaN tuple so cell() can still compare
        nan6 = (np.nan,) * 6
        o = old if old is not None else nan6
        nv = new if new is not None else nan6

        cells = [cell(o[i], nv[i]) for i in range(6)]

        row_tex = table_to_tex(
            [[f"{n}", *cells]], width=22
        ).rstrip("\n")
        if n not in shown:
            row_tex = "% " + row_tex
        tex_lines.append(row_tex)

        # markdown: just dump old / new pairs separated by ' / '
        def md(o_, n_):
            so = fmt(o_)
            sn = fmt(n_)
            if so == sn:
                return f"{sn:>10}"
            return f"{so:>5} / {sn:<5}"

        md_rows.append(
            f"| {n:>5} "
            + " ".join(f"| {md(o[i], nv[i])}" for i in range(6))
            + " |"
        )

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
        width=22,
    )

    table = r"\begin{tabular}{X}".replace("X", layout) + "\n"
    table += r"\toprule" + "\n"
    table += heading
    table += titles_tex
    table += r"\midrule" + "\n"
    table += "\n".join(tex_lines) + "\n"
    table += r"\bottomrule" + "\n"
    table += r"\end{tabular}" + "\n"

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


for crystal in CRYSTALS:
    label = CRYSTAL_LABELS[crystal]
    tex, md = make_table(crystal)
    print(f"\n=== {label} ===\n")
    print(tex)
    savefile(tex, f"tables/benchmark_{crystal}_comparison.tex")
    savefile(md, f"tables/benchmark_{crystal}_comparison.md")
