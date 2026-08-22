"""Loads the *original* (pre-fix) per-class biodimers metrics so the
published figures stay reproducible. The per-class indexing bug in
metrics_biodimers.py only affected the LOREM variants (other models
have their own evaluation drivers), so for non-LOREM models we read
the same metrics_<cl>.npz that the corrected loader does."""

from tbx import *


classes = [
    "AA",
    "CA",
    "CC",
    "CP",
    "PA",
    "PP",
]


def _load(folder, cl):
    if folder.name.startswith("lorem"):
        return np.load(folder / f"metrics_{cl}-broken.npz")
    return np.load(folder / f"metrics_{cl}.npz")


metrics = {}

for file in (datadir / "bio_dimers").glob("*/metrics_AA.npz"):
    folder = file.parent
    model = file.parent.stem

    metrics[model] = {}
    for cl in classes:
        data = _load(folder, cl)
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
