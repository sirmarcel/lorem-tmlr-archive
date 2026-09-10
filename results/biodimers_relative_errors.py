"""Print per-class relative force errors (MAE/std and RMSE/std) for all
models on the biodimers test set, formatted as a LaTeX comment block.
Pipe into a tex file for archival, e.g.:
    python biodimers_relative_errors.py >> ../tex/main.tex
"""

from biodimers import metrics, classes, class_force_std

keys = ["lorem", "cace", "MACE", "NativePET-SR"]
header = "%   class | " + " | ".join(f"{k:>13s}" for k in keys)
sep = "%   " + "-" * (len(header) - 4)


def print_table(title, value_fn):
    print(title)
    print(header)
    print(sep)
    for c in classes:
        row = f"%   {c:>5s} | " + " | ".join(value_fn(k, c) for k in keys)
        print(row)
    print("%")


print("% biodimers: per-class force errors")
print(
    "% std of force components (meV/A): "
    + ", ".join(f"{c}={class_force_std[c]:.2f}" for c in classes)
)
print("%")

print_table(
    "% MAE (meV/A)",
    lambda k, c: f"{metrics[k][c]['forces_mae']:>13.3f}",
)
print_table(
    "% MAE / std (%)",
    lambda k, c: f"{100 * metrics[k][c]['forces_mae'] / class_force_std[c]:>12.1f}%",
)
print_table(
    "% RMSE / std (%)",
    lambda k, c: f"{100 * metrics[k][c]['forces_rmse'] / class_force_std[c]:>12.1f}%",
)


print("% lorem vs cace (both use scalar 1/r LR; lorem adds equivariant l>0 +")
print("% post-LR MLP/update block). Ratio < 1 means lorem is more accurate.")
print(f"%   class |  MAE_lorem/MAE_cace |  RMSE_lorem/RMSE_cace")
print("%   " + "-" * 56)
for c in classes:
    mae_ratio = metrics["lorem"][c]["forces_mae"] / metrics["cace"][c]["forces_mae"]
    rmse_ratio = metrics["lorem"][c]["forces_rmse"] / metrics["cace"][c]["forces_rmse"]
    print(f"%   {c:>5s} | {mae_ratio:>18.3f}  | {rmse_ratio:>22.3f}")
print("%")
