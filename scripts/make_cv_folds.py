"""Write the canonical (id, fold) assignment for the project's outer CV.

Materialises StratifiedKFold(n_splits=5, shuffle=True, random_state=42) over the
full training set — the splitter every logged run uses — to
experiments/cv_folds_seed42.csv.gz (committed, so Kaggle notebooks can assert
against it after cloning the repo). This makes the shared-folds contract that
stacking relies on explicit and checkable, instead of resting on "same y order
+ same seed" being re-derived identically in every environment.

y comes from the raw train.csv: src/data.py::prepare_data reads that CSV and
encodes without reordering, so CSV row order is the row order of every
processed/FE parquet (verified by scripts/check_oof_alignment.py).

Run: .venv\\Scripts\\python.exe scripts\\make_cv_folds.py
"""

import csv
import gzip
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = PROJECT_ROOT / "experiments" / "cv_folds_seed42.csv.gz"


def main() -> None:
    # Same fallback as src/data.py::_resolve_raw_dir — data/raw/ first, then data/.
    raw_csv = PROJECT_ROOT / "data" / "raw" / "train.csv"
    if not raw_csv.exists():
        raw_csv = PROJECT_ROOT / "data" / "train.csv"

    ids, y = [], []
    with open(raw_csv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        id_idx, churn_idx = header.index("id"), header.index("Churn")
        for row in reader:
            ids.append(int(row[id_idx]))
            y.append(1 if row[churn_idx] == "Yes" else 0)
    y = np.asarray(y)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = np.full(len(y), -1, dtype=int)
    # X is only consulted for its length; zeros keep this independent of features.
    for fold, (_, va_idx) in enumerate(cv.split(np.zeros(len(y)), y)):
        folds[va_idx] = fold
    assert (folds >= 0).all()

    with gzip.open(OUT_PATH, "wt", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "fold"])
        writer.writerows(zip(ids, folds.tolist()))

    print(f"Wrote {OUT_PATH}  ({len(y)} rows)")
    for fold in range(cv.n_splits):
        mask = folds == fold
        print(f"  fold {fold}: n={mask.sum():6d}  churn rate={y[mask].mean():.6f}")


if __name__ == "__main__":
    main()
