from tbx import *
from tbx.tables import *
from metrics import metrics

# "Nearest other" is the best controlled-regime baseline (1xSR LR / 2xSR) for E
# and F separately. SpookyNet (6xSR) appears in the main table but is out of this
# comparison's regime, so it is excluded here.
others = {"cace": r"\cace", "MACE": r"\mace", "NativePET-SR": r"\pet", "4G_NN": r"\fgnn"}

num = rounder(3)


def nearest(data, target):
    cand = {k: data[k][target] for k in others if k in data}
    k = min(cand, key=cand.get)
    return f"{num(cand[k])} ({others[k]})"


for tag, experiment in {"mgo": "AuMgO", "nacl": "NaCl"}.items():
    d = metrics[experiment]
    lo, be, bf = d["lorem"], d["lorem-tuned-energy"], d["lorem-tuned-forces"]
    rows = [
        (r"\lorem", num(lo["energy_rmse"]), num(lo["forces_rmse"])),
        ("Nearest other", nearest(d, "energy_rmse"), nearest(d, "forces_rmse")),
        (r"Best $E$ model", num(be["energy_rmse"]), num(be["forces_rmse"])),
        (r"Best $F$ model", num(bf["energy_rmse"]), num(bf["forces_rmse"])),
    ]
    body = "\n".join(f"{m} & {e} & {f} \\\\" for m, e, f in rows)
    table = "\n".join([
        r"\begin{tabular}{r|cc}",
        r"\toprule",
        r"Model & Valid. RMSE $E$ (meV/atom) & Valid. RMSE $\F$ (meV/\unit{\angstrom}) \\",
        r"\midrule",
        body,
        r"\bottomrule",
        r"\end{tabular}",
    ])
    savefile(table, f"tables/hp_tuning_{tag}.tex")
    print(f"--- hp_tuning_{tag} ---\n{table}\n")
