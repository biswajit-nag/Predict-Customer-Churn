"""Sanity-check OOF row alignment for runs trained off-platform (Kaggle).

For each run, reload experiments/runs/{run_id}/oof_proba.npy, recompute the OOF
ROC-AUC and accuracy against the local training target, and compare with the
values logged in runs.csv at save time. A match (to ~1e-9) proves the saved OOF
vector is row-aligned with the local train data; a mismatch means the
on-platform data regeneration ordered rows differently and the OOF vector must
not be stacked against local folds as-is.

Stdlib-only on purpose: no numpy/pandas/sklearn imports, so it runs even when
heavy imports are unavailable. y comes from data/raw/train.csv — src/data.py's
prepare_data() reads that CSV and encodes without reordering, so CSV row order
IS the row order of every processed/FE parquet, local or Kaggle-regenerated.
The .npy file is parsed by hand (v1.0 format, little-endian float64), and the
ROC-AUC uses the tie-corrected rank formula, which is mathematically identical
to sklearn's trapezoidal value.

Run: .venv\\Scripts\\python.exe scripts\\check_oof_alignment.py
"""

import ast
import csv
import struct
from array import array
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "experiments" / "runs"

# Kaggle-trained full-set runs, plus one local run as a control for the check.
CHECK_IDS = [
    ("20260610-212446-de62f2", "local control"),
    ("20260601-022539-10bd38", "kaggle"),
    ("20260603-021557-d716c1", "kaggle"),
    ("20260610-183508-48286b", "kaggle"),
    ("20260611-001429-ba1eef", "kaggle"),
    ("20260611-011634-c13e8d", "kaggle"),
    ("20260611-051031-5d6bc5", "kaggle"),
    # fe_v3/fe_v4 Optuna refinement batch (2026-06-28 Kaggle, 2026-06-29 local finalize)
    ("20260628-110124-485af0", "kaggle"),   # catboost fe_v3
    ("20260628-054913-ef8595", "kaggle"),   # lgbm fe_v3
    ("20260628-091724-48ab31", "kaggle"),   # xgb fe_v3
    ("20260628-080619-b4d2ca", "kaggle"),   # xgb fe_v4 (best Kaggle)
    ("20260628-183230-86782f", "local"),    # lgbm fe_v4 finalize
    ("20260629-135226-d1f447", "local"),    # catboost fe_v4 finalize (CPU)
    ("20260629-161428-7f0bce", "local"),    # ebm fe_v4 finalize
]


def load_npy_f64(path: Path) -> array:
    """Parse a v1.0/v2.0 .npy file holding a little-endian float64 1-D array."""
    with open(path, "rb") as f:
        magic = f.read(6)
        assert magic == b"\x93NUMPY", f"not a .npy file: {path}"
        major, _minor = f.read(1)[0], f.read(1)[0]
        hlen = struct.unpack("<H" if major == 1 else "<I", f.read(2 if major == 1 else 4))[0]
        header = ast.literal_eval(f.read(hlen).decode("latin1"))
        assert header["descr"] == "<f8", f"unexpected dtype {header['descr']} in {path}"
        assert not header["fortran_order"]
        data = array("d")
        data.frombytes(f.read())
    n = header["shape"][0]
    assert len(data) == n, f"size mismatch in {path}"
    return data


def roc_auc(y: list, scores) -> float:
    """Tie-corrected rank AUC == sklearn.metrics.roc_auc_score."""
    n = len(y)
    order = sorted(range(n), key=scores.__getitem__)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    n_pos = sum(y)
    n_neg = n - n_pos
    rank_sum_pos = sum(r for r, yi in zip(ranks, y) if yi)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main() -> None:
    # y in raw-CSV row order == processed parquet row order (see module docstring).
    # Same fallback as src/data.py::_resolve_raw_dir — data/raw/ first, then data/.
    raw_csv = PROJECT_ROOT / "data" / "raw" / "train.csv"
    if not raw_csv.exists():
        raw_csv = PROJECT_ROOT / "data" / "train.csv"
    with open(raw_csv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        churn_idx = next(reader).index("Churn")
        y = [1 if row[churn_idx] == "Yes" else 0 for row in reader]
    print(f"Local y: n={len(y)}, churn rate={sum(y) / len(y):.6f}\n")

    with open(PROJECT_ROOT / "experiments" / "runs.csv", newline="", encoding="utf-8") as f:
        runs = {row["run_id"]: row for row in csv.DictReader(f)}

    header = (f"{'run_id':23s} {'tag':26s} {'origin':14s} "
              f"{'auc(recomp)':>13s} {'auc(logged)':>13s} {'acc(recomp)':>13s} {'acc(logged)':>13s}  verdict")
    print(header)
    print("-" * len(header))

    for run_id, origin in CHECK_IDS:
        row = runs[run_id]
        oof = load_npy_f64(RUNS_DIR / run_id / "oof_proba.npy")
        if len(oof) != len(y):
            print(f"{run_id:23s} {row['tag']:26s} {origin:14s}  SKIP — oof length {len(oof)} != {len(y)}")
            continue
        auc = roc_auc(y, oof)
        acc = sum((p >= 0.5) == yi for p, yi in zip(oof, y)) / len(y)
        auc_ok = abs(auc - float(row["oof_roc_auc"])) < 1e-9
        acc_ok = abs(acc - float(row["oof_accuracy"])) < 1e-9
        verdict = "ALIGNED" if (auc_ok and acc_ok) else "MISMATCH"
        print(f"{run_id:23s} {row['tag']:26s} {origin:14s} "
              f"{auc:13.9f} {float(row['oof_roc_auc']):13.9f} "
              f"{acc:13.9f} {float(row['oof_accuracy']):13.9f}  {verdict}")


if __name__ == "__main__":
    main()
