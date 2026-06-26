"""One-off validation for the subfold notebooks + canonical fold CSV (stdlib-only)."""

import ast
import csv
import gzip
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_oof_alignment import load_npy_f64  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- 1. Notebooks: valid JSON, every code cell compiles, zip code is a code cell ---
for name in ("predict-customer-churn-tabpfn-subfold.ipynb",
             "predict-customer-churn-tabicl-subfold.ipynb",
             "predict-customer-churn-tabm-gpu-min3.ipynb",
             "predict-customer-churn-realmlp-gpu-min3.ipynb"):
    nb = json.loads((PROJECT_ROOT / "kaggle" / name).read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    n_code = 0
    zip_in_code = False
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if cell["cell_type"] != "code":
            continue
        n_code += 1
        py = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("!", "%")))
        ast.parse(py)  # raises SyntaxError on any broken cell
        if "make_archive" in src:
            zip_in_code = True
    assert zip_in_code, f"{name}: zip snippet not found in a code cell"
    print(f"OK {name}: {n_code} code cells compile; zip step is a code cell")

# --- 2. Fold CSV: read back, then validate against a logged run's per-fold scores ---
with gzip.open(PROJECT_ROOT / "experiments" / "cv_folds_seed42.csv.gz", "rt", encoding="utf-8") as f:
    reader = csv.reader(f)
    assert next(reader) == ["id", "fold"]
    rows = [(int(i), int(fo)) for i, fo in reader]
ids = [r[0] for r in rows]
folds = [r[1] for r in rows]
assert len(rows) == 594194 and sorted(set(folds)) == [0, 1, 2, 3, 4]

raw_csv = PROJECT_ROOT / "data" / "raw" / "train.csv"
if not raw_csv.exists():
    raw_csv = PROJECT_ROOT / "data" / "train.csv"
with open(raw_csv, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    id_idx, churn_idx = header.index("id"), header.index("Churn")
    raw = [(int(row[id_idx]), 1 if row[churn_idx] == "Yes" else 0) for row in reader]
assert [r[0] for r in raw] == ids, "fold CSV id order != raw train.csv order"
y = [r[1] for r in raw]
print("OK cv_folds_seed42.csv.gz: 594194 rows, ids match raw train.csv order")

# Ground truth: per-fold accuracies of the local control run must be reproducible
# from this fold assignment + its OOF vector.
run_dir = PROJECT_ROOT / "experiments" / "runs" / "20260610-212446-de62f2"
fold_scores = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))["fold_scores"]
oof = load_npy_f64(run_dir / "oof_proba.npy")
for k in range(5):
    n = hits = 0
    for fo, p, yi in zip(folds, oof, y):
        if fo == k:
            n += 1
            hits += (p >= 0.5) == yi
    acc = hits / n
    assert abs(acc - fold_scores[k]) < 1e-9, f"fold {k}: {acc} != logged {fold_scores[k]}"
    print(f"OK fold {k}: recomputed accuracy {acc:.10f} == logged {fold_scores[k]:.10f}")

print("\nAll validations passed.")
