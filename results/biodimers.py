from tbx import *


classes = [
    "CC",
    "CP",
    "PP",
    "CA",
    "PA",
    "AA",
]

# ideal multipole-multipole interaction decay exponent for each class
class_power = {"CC": 1, "CP": 2, "PP": 3, "CA": 4, "PA": 5, "AA": 6}


def class_power_label(c):
    p = class_power[c]
    inner = r"\sim r^{-1}" if p == 1 else rf"\sim r^{{-{p}}}"
    return rf"$({inner})$"


metrics = {}

for file in (datadir / "bio_dimers").glob("*/metrics_AA.npz"):
    folder = file.parent
    model = file.parent.stem

    metrics[model] = {}
    for cl in classes:
        data = np.load(folder / f"metrics_{cl}.npz")
        metrics[model][cl] = {
            "energy_rmse": float(data["energy_rmse"]),
            "energy_mae": float(data["energy_mae"]),
            "forces_rmse": float(data["forces_rmse"]),
            "forces_mae": float(data["forces_mae"]),
        }


_stats = np.load(datadir / "bio_dimers" / "class_stats.npz")
class_force_std = {cl: float(_stats[f"{cl}_force_std"]) for cl in classes}
class_energy_std = {cl: float(_stats[f"{cl}_energy_std"]) for cl in classes}

if __name__ == "__main__":
    print(metrics)
