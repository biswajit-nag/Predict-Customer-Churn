# Running experiments on a Kaggle GPU

Reference for running a GPU-heavy experiment (e.g. TabM) on a Kaggle kernel and
folding the resulting artifacts back into this repo's experiment tracking.

**Mental model:** Kaggle is a *run target*, not a clone of the local dev
environment. Do not try to reproduce the `uv` environment there — use Kaggle's
pre-built GPU image (CUDA PyTorch, sklearn, LGBM/XGB/CatBoost already present)
and `pip install` only what's missing. `uv.lock` targets Python ≥3.14 and the
local platform; it will not map onto Kaggle's image.

---

## 1. Notebook settings

In a **browser** notebook (not the VS Code remote path — that does not upload
project files), right sidebar → Settings:

- **Accelerator → GPU T4 x2.** Use T4, **not P100**: Kaggle's PyTorch build is
  compiled for sm_70+ and the P100 is sm_60 (Pascal), so it fails at kernel
  execution with `no kernel image is available` even though `cuda.is_available()`
  returns `True`.
- **Internet → On** (requires phone verification). Needed for `git clone` + `pip`.

## 2. Attach the competition data

Right sidebar → **Add Input** → add `playground-series-s6e3`. Mounts read-only at
`/kaggle/input/playground-series-s6e3/{train,test}.csv`.

## 3. Setup cell — clone, paths, install

```python
import os, sys, subprocess

REPO_URL  = "https://github.com/biswajit-nag/Predict-Customer-Churn.git"
REPO_ROOT = "/kaggle/working/Predict-Customer-Churn"
# Private repo? Use a Kaggle Secret holding a GitHub PAT:
#   REPO_URL = f"https://{user}:{pat}@github.com/biswajit-nag/Predict-Customer-Churn.git"

if not os.path.exists(REPO_ROOT):
    subprocess.run(["git", "clone", REPO_URL, REPO_ROOT], check=True)

os.chdir(REPO_ROOT)                 # CWD = repo root: fixes data paths + git_info
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)   # makes `from src.xxx import ...` resolve

!pip install -q pytabkit           # reuses Kaggle's CUDA torch; ignore RAPIDS conflict warnings
```

`os.chdir` + `sys.path.insert` together replicate the local layout: `chdir` lets
`_find_project_root()` and `DATA_DIR` resolve and lets `git_info()` see the repo;
`sys.path.insert` makes `src` importable (locally this is free because Jupyter
puts the startup dir on the path; on Kaggle the kernel starts in `/kaggle/working`).

## 4. Verify the GPU can actually execute a kernel

`cuda.is_available()` is **not** sufficient — it returns `True` on incompatible
GPUs. Run a real op:

```python
import torch
print("device:", torch.cuda.get_device_name(0), "| count:", torch.cuda.device_count())
try:
    _ = (torch.randn(16, device="cuda") @ torch.randn(16, 16, device="cuda")).sum().item()
    print("GPU compute OK")
except Exception as e:
    print("GPU compute FAILED:", e)   # if this fails, switch accelerator to T4 and restart
```

## 5. Regenerate processed data on-platform

`data/processed/*.parquet` are git-ignored, so they are **absent** from the
clone. Rebuild them from the attached competition CSVs:

```python
import shutil
from pathlib import Path

raw_dir = Path(REPO_ROOT) / "data" / "raw"
raw_dir.mkdir(parents=True, exist_ok=True)
for f in ("train.csv", "test.csv"):
    shutil.copy(f"/kaggle/input/playground-series-s6e3/{f}", raw_dir / f)

from src.data import prepare_data
prepare_data(force=True)            # writes data/processed/{train,test}_df.parquet
```

## 6. Run the experiment

Build the features with `src.features.build_features(...)` (or the FE function for
the version you are running), then define a `run_config` and call `run_cv_experiment` /
`save_experiment` as in `02_Experiments.ipynb` (FE → run_config →
run → save). In `run_config`:

- `'device': 'cuda'`
- `'save_models': False` — torch-backed models are fragile to `joblib.dump`
  (CUDA tensors, pytabkit training state) and a dump failure would crash
  `save_experiment` *after* the expensive run. `oof_proba`, `test_proba_*`,
  `params`, and `metrics` are saved regardless of this flag, so nothing you
  need for the leaderboard/submission is lost.
- Record the real environment in `notes` (since `environment.txt` will hold the
  wrong local `uv.lock`):

  ```python
  'notes': ('TabM (pytabkit TabM_D_Classifier) on Kaggle T4 GPU. '
            'torch=<ver>, pytabkit=<ver>. Data regenerated on-platform — '
            'data_hash differs from local runs; GPU run not bit-reproducible.'),
  ```

**Smoke-test on one fold first** (`StratifiedKFold(n_splits=2, ...)`) to confirm
the GPU path works, measure per-fold time, and check how many epochs fold 0
actually ran (confirms early stopping is active before you commit to 5 folds).

**For the long run, do not babysit an interactive session** (it disconnects):
top-right **Save Version → Save & Run All (Commit)** runs in the background up to
12h and persists output. Mind the ~30 GPU-hours/week quota.

## 7. Bring artifacts back into the local repo

The run was saved into the *cloned* repo's `experiments/`. On Kaggle, zip the new
run directory for download:

```python
from src.tracking import RUNS_DIR
shutil.make_archive(f"/kaggle/working/{run_id}", "zip", RUNS_DIR / run_id)
```

Download it (Output tab) plus the Kaggle `experiments/runs.csv`. Locally:

1. Extract the `experiments/runs/{run_id}/` folder into the local `experiments/runs/`.
   (`artifact_dir` is stored as a repo-relative path, so it resolves as-is.)
2. Append only the new row to the local `runs.csv` — do **not** overwrite it, since
   local `runs.csv` may have advanced since the clone:

   ```python
   import pandas as pd
   local  = pd.read_csv("experiments/runs.csv")
   kaggle = pd.read_csv("path/to/downloaded/runs.csv")
   new    = kaggle[~kaggle["run_id"].isin(local["run_id"])]
   pd.concat([local, new], ignore_index=True).to_csv("experiments/runs.csv", index=False)
   ```
3. (Optional) Build `submission.csv` from `test_proba_mean.npy`, submit to the
   competition, and backfill `lb_public` / `lb_private` for the run.
4. Commit.

## 8. Save the Kaggle notebook in the repo

**Always commit the Kaggle notebook itself** (e.g. under `kaggle/`). It is the
only record of how a GPU run was produced — the setup glue, `pip install`,
`device`, and on-platform data regeneration are not captured anywhere else, and
`environment.txt` for these runs is wrong. Name it so it ties back to the run(s)
it produced, and reference the notebook filename in the run's `notes`.

---

## Gotchas

- **P100 fails, T4 works** (sm_60 vs the torch build's sm_70+ floor).
- **RAPIDS pip conflict warnings** (`cudf`, `cuml`, `dask-cuda`) are pre-existing
  in Kaggle's image and irrelevant — ignore them if `import pytabkit` / `import
  torch` succeed. Note `pytabkit` has no `__version__` attribute.
- **`data_hash` mismatch**: the on-platform parquet hashes differently from the
  local one (pyarrow version, row order) even though the data is logically equal.
- **GPU non-determinism**: OOF scores are not bit-reproducible against a local
  CPU run.
- **The run/save split is your safety net**: `run_cv_experiment` returns `result`
  (with `oof_proba`/`test_proba_mean` in memory) *before* saving, so if
  `save_experiment` errors you can still `np.save` from the live `result` object
  instead of re-running.
