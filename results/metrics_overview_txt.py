from tbx import *
from tbx.tables import *
from metrics import metrics

for experiment in ["AuMgO", "bio_dimers", "cumulene", "NaCl"]:
    data = metrics[experiment]

    def get_row(data, key):
        dct = {k: v[key] for k, v in data.items() if key in v}
        best = min(dct, key=dct.get)

        # emit a placeholder for models without this metric, so the row
        # stays aligned with the full column header
        num = rounder(3)
        row = []
        for k in data.keys():
            if k not in dct:
                row.append("--")
            elif k == best:
                row.append("*" + num(dct[k]) + "*")
            else:
                row.append(num(dct[k]))

        return row

    E = get_row(data, "energy_mae")
    F = get_row(data, "forces_mae")

    table = table_to_string(
        [E, F],
        colnames=[""] + list(data.keys()),
        width=16,
        rownames=["E (meV/at)", "F (meV/Å)"],
    )

    print(f"## {experiment} ##")
    print(table)
