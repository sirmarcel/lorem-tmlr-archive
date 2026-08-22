"""Drive metrics_biodimers.main for one LOREM variant.

Mirrors the archive driver pattern: exec the shared metrics script (so its
module-level `Path` default resolves in this file's scope), build the LOREM
Calculator from the variant's checkpoint, then call main() writing into the
variant directory.

Usage:
    cd <variant_dir>   # e.g. evals/bio_dimers/lorem
    python ../run_lorem_metrics.py
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent

with open(HERE.parent / "metrics_biodimers.py") as f:
    exec(f.read())

from lorem.calculator import Calculator

DATASET = HERE / "../../datasets"

calc = Calculator.from_checkpoint("run/checkpoints/R2_E+F", add_offset=True)
main(calc, DATASET, workdir=Path.cwd())
