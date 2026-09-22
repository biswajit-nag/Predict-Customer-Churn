# Experiment tracking

Every model in this project — 72 runs across local CPU and Kaggle GPU kernels,
including the blends — is recorded in one plain, git-tracked ledger. This page
describes the design, the columns, and the guarantees it provides.

## Design

Two layers, written by `src/tracking.py::save_run`:

* **`experiments/runs.csv`** — one row per run, scalars only. It is the leaderboard
  and the index; `load_runs()` reads it sorted by OOF ROC-AUC.
* **`experiments/runs/{run_id}/`** — everything needed to understand or rebuild the
  run:

  | File | Contents |
  |---|---|
  | `params.json` | the full, resolved model parameters (`get_params()`), or for a blend its members and weights |
  | `metrics.json` | per-fold scores, timings, CV configuration, feature list |
  | `notes.md` | a free-text description of what the run was and why |
  | `git_diff.patch` | uncommitted changes at run time, so a dirty working tree is still recoverable |
  | `environment.txt` | a snapshot of `uv.lock` at run time |
  | `feature_importance.csv` | per-fold-averaged gain and split importance (LightGBM runs) |
  | `oof_proba.npy`, `test_proba_mean.npy`, `test_proba_folds.npy` | out-of-fold predictions for every training row, and test predictions (not committed: large and regenerable) |

`run_id` is `{YYYYMMDD-HHMMSS}-{6 hex}`: sortable by time, unique without
coordination between machines.

**Why a CSV and a folder instead of MLflow or Weights & Biases?** The ledger had
to work identically on a laptop and inside short-lived Kaggle kernels, survive
being zipped and copied back, diff cleanly in git, and be analysable with one
`pd.read_csv`. A CSV plus files does all of that with no server, account or SDK.
What it gives up — a UI, concurrent writers, artifact deduplication — did not matter
for a single-author project of this size.

**Run, inspect, then save.** `run_cv_experiment` returns an inspectable result and
writes nothing; `save_experiment(result)` persists it (and optionally submits it to
Kaggle). A run is logged only when it is worth keeping, and an expensive GPU run
cannot be lost to a failure in the saving step.

## Columns

| Group | Columns | Meaning |
|---|---|---|
| Identity | `run_id`, `timestamp`, `tag`, `notes`, `status`, `parent_run_id` | what the run was; `parent_run_id` links refinements to the run they started from |
| Code provenance | `git_hash`, `git_dirty` | the commit the run used; if dirty, `git_diff.patch` holds the difference |
| Data provenance | `data_version`, `data_hash` | the feature-set label and a SHA-256 fingerprint of the exact training parquet |
| Environment | `python_version`, `platform`, `kaggle_run` | where it ran; `kaggle_run` marks on-platform runs |
| Model | `model_class`, `params_hash`, `n_features` | `params_hash` answers "have I already run this exact configuration?" |
| Validation | `cv_type`, `n_splits`, `random_state`, `n_train`, `n_test`, `n_pseudo` | the CV splitter, data sizes, and the number of pseudo-labelled rows added to training folds |
| Primary metric | `metric`, `oof_roc_auc`, `fold_roc_auc_mean`, `fold_roc_auc_std` | ROC-AUC, the competition metric; the per-fold standard deviation is the noise scale for comparing runs |
| Secondary metric | `oof_accuracy`, `fold_acc_mean`, `fold_acc_std` | accuracy at a 0.5 threshold, a diagnostic only |
| Held-out truth | `lb_public`, `lb_private` | Kaggle leaderboard scores, filled in after submission (68 of 72 runs) |
| Cost | `training_time_sec`, `artifact_dir` | total fit time and where the artifacts live |

The schema grew during the project. When a new column appears, `save_run` rewrites
the file as an outer join — older rows get an empty value — instead of appending a
misaligned row.

## The shared-folds contract

Every run is cross-validated with
`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` over the training data
in raw `train.csv` row order. Nothing in the pipeline reorders rows, so fold *k* is
the same set of customers for every model on every machine. The assignment is
committed as `experiments/cv_folds_seed42.csv.gz`.

That is what makes the OOF vectors of different models comparable and stackable: a
blend or stack cross-validated on the same folds never sees a prediction made by a
model that trained on the row being predicted.

**The alignment guard.** Kaggle kernels regenerate the data on-platform, so their
OOF vectors could in principle arrive in a different row order. Before any saved
OOF vector is reused, its ROC-AUC is recomputed against the local target and must
equal the logged `oof_roc_auc` to within 1e-9. `scripts/check_oof_alignment.py`
runs this over the whole ledger; `src/blending.py` and `src/stacking.py` enforce it
on every load.

## Blends are runs too

A blend has no model fit of its own, but it is logged exactly like one
(`src/blending.py::save_blend_run`): OOF and test vectors, per-fold AUC, members and
weights in `params.json`, and leaderboard scores after submission. A few columns
take blend conventions: `data_version = "blend_v1"`, `model_class` names the
combining scheme, and `n_features` is the number of members. Blends are excluded
when the blending pool is loaded, so a blend is never a member of another blend.

## What it made possible

* **Rebuilding a model from its record.** `02_Experiments.ipynb` rebuilds the best
  single model from its ledger row alone and reproduces every one of its 594,194
  OOF predictions exactly.
* **Measuring the validation itself.** Because 68 runs carry both an OOF score and a
  private-leaderboard score, the fidelity of the CV setup is a number — Spearman
  0.977, with a stable offset of 0.0013 (`04_Results.ipynb`).
* **An honest record of what failed.** Negative results stay in the ledger with the
  same provenance as the successes
  ([stacking_experiments.md](stacking_experiments.md)).
