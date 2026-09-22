"""Verify that every saved OOF vector is row-aligned with the local training data.

Every run in the ledger was cross-validated on the same frozen folds
(``StratifiedKFold(5, shuffle=True, random_state=42)`` over raw ``train.csv`` row
order), which is what makes OOF vectors from different models — and from
different machines — stackable. Some runs were trained on Kaggle kernels that
regenerate the data on-platform, so this script proves, for each run, that its
saved ``oof_proba.npy`` lines up with the local target row by row:

    recompute ROC-AUC(y_local, oof_proba)  ==  oof_roc_auc logged at save time

A match to 1e-9 means the rows are aligned; a mismatch means the rows were
reordered and the vector must not be blended or stacked. Runs whose OOF length
differs from the training set (subsample passes) and runs whose arrays are not
present locally (prediction arrays are not committed) are reported as skipped.

Run from the project root:
    python scripts/check_oof_alignment.py            # all runs
    python scripts/check_oof_alignment.py --quiet    # summary only

Exit code 0 iff no checked run is misaligned.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import prepare_data  # noqa: E402
from src.tracking import RUNS_CSV, RUNS_DIR  # noqa: E402

TOL = 1e-9


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quiet", action="store_true", help="print the summary only")
    args = parser.parse_args()

    # prepare_data never reorders rows, so this y is in raw train.csv order.
    train, _ = prepare_data(encoding="native")
    y = train["Churn"].to_numpy()
    runs = pd.read_csv(RUNS_CSV).dropna(subset=["run_id"])
    print(f"Local target: n={len(y):,}, churn rate={y.mean():.6f}; {len(runs)} runs in ledger\n")

    counts = {"ALIGNED": 0, "MISMATCH": 0, "SKIP": 0}
    for _, r in runs.iterrows():
        path = RUNS_DIR / str(r["run_id"]) / "oof_proba.npy"
        if not path.exists():
            verdict, detail = "SKIP", "oof_proba.npy not present locally"
        else:
            oof = np.load(path)
            if len(oof) != len(y):
                verdict, detail = "SKIP", f"subsample OOF ({len(oof):,} rows)"
            else:
                auc = roc_auc_score(y, oof)
                diff = abs(auc - float(r["oof_roc_auc"]))
                verdict = "ALIGNED" if diff < TOL else "MISMATCH"
                detail = f"recomputed {auc:.9f} vs logged {float(r['oof_roc_auc']):.9f}"
        counts[verdict] += 1
        if not args.quiet or verdict == "MISMATCH":
            print(f"{verdict:8s} {r['run_id']}  {str(r['tag'])[:34]:34s} {detail}")

    print(f"\n{counts['ALIGNED']} aligned, {counts['MISMATCH']} mismatched, "
          f"{counts['SKIP']} skipped")
    return 1 if counts["MISMATCH"] else 0


if __name__ == "__main__":
    sys.exit(main())
