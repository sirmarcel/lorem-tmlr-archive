from tbx import *
from tbx.tables import *
from metrics import metrics


keys = [
    "lorem",
    "cace",
    "MACE",
    "NativePET-SR",
]


width = 24
tables = []

has_test = ["bio_dimers", "cumulene"]

for experiment in ["AuMgO", "NaCl", "bio_dimers", "cumulene"]:
    data = metrics[experiment]

    def get_row(data, target):
        dct = {}
        not_present = []
        for i, k in enumerate(keys):

            if k in data and target in data[k]:
                dct[k] = data[k][target]
            else:
                dct[k] = 1e8
                not_present.append(i)

        sorted_keys = sorted(
            [k for k in dct if k not in [keys[j] for j in not_present]],
            key=dct.get,
        )
        best = sorted_keys[0] if len(sorted_keys) > 0 else None
        second = sorted_keys[1] if len(sorted_keys) > 1 else None

        row = list(dct.values())
        row = map_across([row], rounder(3))[0]
        row = map_across([row], to_num)[0]

        for i in not_present:
            row[i] = r"--"

        for i, k in enumerate(dct.keys()):
            if k == best:
                row[i] = r"\textbf{" + row[i] + "}"
            elif k == second:
                row[i] = r"\underline{" + row[i] + "}"

        row = map_across([row], aligner(width))[0]

        return row

    E = get_row(data, "energy_mae")
    F = get_row(data, "forces_mae")

    if experiment in has_test:
        split = ""
    else:
        split = r"{\small (Validation set)}"

    tables.append(
        [
            [f"{dataset_names[experiment]}" + r"\hspace{3mm} $E$ (meV/at)"]
            + E,
            [split + r"\hspace{3mm} $\F$ (meV/Å)"] + F,
        ]
    )


layout = " ".join(["r", "|"] + ["r"] * len(keys))

table = make_tabular(
    ["Dataset"]
    + [
        r"\makecell{"
        + model_names[key].replace(r"{\small", r"\\{")
        + r"}"
        for key in keys
    ],
    None,
    tables=tables,
    width=24,
    layout=layout,
)

print(table)

savefile(table, "tables/metrics_overview_mae.tex")
