"""Backfill Kaggle public/private leaderboard scores for logged runs.

For every successful run in experiments/runs.csv that has bagged test
predictions (test_proba_mean.npy) and a full-length test vector, this submits
the predictions to Kaggle, waits for scoring, and writes lb_public / lb_private
back into runs.csv.

Resumable and rate-limit aware. Kaggle caps daily submissions (typically 5 for
Playground competitions), so the script:
  - skips runs that already have an lb_public score (resume across days),
  - stops cleanly when Kaggle reports the daily limit is exhausted,
  - honours --max to cap submissions per invocation.

Run again the next day to continue where it left off.

Usage
-----
    # one logged run:
    python scripts/backfill_lb_scores.py --run-id 20260610-183508-48286b

    # up to N un-scored runs, best-OOF-AUC first (default order):
    python scripts/backfill_lb_scores.py --max 5

    # everything still missing (will pause at the daily limit):
    python scripts/backfill_lb_scores.py --max 1000

Prerequisites: Kaggle credentials configured — see docs/kaggle_setup.md.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.kaggle_io import (  # noqa: E402
    N_TEST, RUNS_DIR, credentials_available, submit_run,
)
from src.tracking import RUNS_CSV  # noqa: E402


def _is_scored(value) -> bool:
    return value is not None and not (isinstance(value, float) and np.isnan(value))


def _submittable(run_id: str) -> bool:
    """True if the run has a full-length bagged test vector to submit."""
    p = RUNS_DIR / run_id / "test_proba_mean.npy"
    if not p.exists():
        return False
    try:
        return len(np.load(p, mmap_mode="r")) == N_TEST
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", help="Submit only this run_id.")
    ap.add_argument("--max", type=int, default=5,
                    help="Max submissions this invocation (default 5; Kaggle's "
                         "typical daily cap). Use a large number to drain fully.")
    ap.add_argument("--no-wait", action="store_true",
                    help="Fire submissions without waiting for scoring (collect "
                         "later by re-running without --no-wait).")
    ap.add_argument("--include-scored", action="store_true",
                    help="Re-submit runs that already have an lb_public score.")
    args = ap.parse_args()

    if not credentials_available():
        sys.exit("Kaggle credentials not configured — see docs/kaggle_setup.md.")

    runs = pd.read_csv(RUNS_CSV)
    runs = runs[runs["status"] == "success"]
    # Best models first so a small daily budget scores the most useful runs.
    runs = runs.sort_values("oof_roc_auc", ascending=False)

    if args.run_id:
        targets = [args.run_id]
    else:
        targets = []
        for _, r in runs.iterrows():
            if not args.include_scored and _is_scored(r.get("lb_public")):
                continue
            if not _submittable(r["run_id"]):
                print(f"skip {r['run_id']} ({r['tag']}): no full-length "
                      f"test_proba_mean.npy")
                continue
            targets.append(r["run_id"])

    if not targets:
        print("Nothing to submit — every eligible run already has an LB score.")
        return

    print(f"{len(targets)} run(s) eligible; submitting up to {args.max} this run.\n")
    done = 0
    for run_id in targets:
        if done >= args.max:
            print(f"\nReached --max={args.max}. Re-run later to continue.")
            break
        try:
            submit_run(run_id, wait=not args.no_wait)
            done += 1
        except RuntimeError as e:
            msg = str(e).lower()
            if "daily submission limit" in msg or "too many requests" in msg:
                print(f"\nDaily submission limit reached after {done} submission(s). "
                      f"Re-run tomorrow to continue.")
                break
            print(f"ERROR submitting {run_id}: {e}")
            # Keep going: one bad run shouldn't abort the whole batch.

    print(f"\nDone: {done} submission(s) this invocation.")


if __name__ == "__main__":
    main()
