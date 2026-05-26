"""Experiment tracking utilities — CSV + filesystem logging.

Path constants (PROJECT_ROOT, EXPERIMENTS_DIR, etc.) are derived by locating
pyproject.toml, so they resolve correctly regardless of which directory a
notebook or script is run from.

XGBoost ≥3.2 / pandas ≥3 compatibility
----------------------------------------
pandas 3.0 removed pd.util, but XGBoost probes pd.util.version.Version to
detect pandas ≥2.1. A minimal shim is applied at import time so the check
passes without patching XGBoost itself.
"""

import datetime
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import types as _types
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# XGBoost ≥3.2 + pandas ≥3 compatibility shim
# ---------------------------------------------------------------------------
if not hasattr(pd, "util"):
    from packaging.version import Version as _Version
    pd.util = _types.SimpleNamespace(
        version=_types.SimpleNamespace(Version=_Version)
    )

# ---------------------------------------------------------------------------
# Project-root-relative paths
# ---------------------------------------------------------------------------

def _find_project_root() -> Path:
    """Walk up from this file until pyproject.toml is found."""
    for candidate in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Could not locate project root (no pyproject.toml found in any parent directory)")


PROJECT_ROOT    = _find_project_root()
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RUNS_CSV        = EXPERIMENTS_DIR / "runs.csv"
RUNS_DIR        = EXPERIMENTS_DIR / "runs"
DATA_DIR        = PROJECT_ROOT / "data" / "processed"
UV_LOCK         = PROJECT_ROOT / "uv.lock"

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def hash_file(path: Path, n: int = 12) -> str:
    """SHA-256 of a file's bytes, truncated to n hex characters.

    Why: stores a short fingerprint of the input parquet in the CSV row so
    "did this run use the same source data as that one?" is an O(1) lookup.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def git_info() -> dict:
    """Return the current git hash, a dirty flag, and the full diff patch.

    Why: hash alone is insufficient when the working tree is dirty. The patch
    makes dirty-tree runs reconstructible (apply it on top of the commit).
    Degrades gracefully outside a git repo (returns empty strings).
    """
    def _run(args: list[str]) -> str:
        r = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
        return r.stdout.strip() if r.returncode == 0 else ""

    diff_text = _run(["diff", "HEAD"])
    return {
        "hash":  _run(["rev-parse", "--short", "HEAD"]),
        "dirty": bool(diff_text),
        "diff":  diff_text,
    }


def environment_info() -> dict:
    """Return Python version, platform string, and uv.lock contents.

    Why: uv.lock is the authoritative dependency record. It is too large to
    embed in the CSV, so save_run writes it to environment.txt in the run
    directory; only version + platform land in the CSV row.
    """
    return {
        "python_version":   sys.version.split()[0],
        "platform":         platform.platform(),
        "uv_lock_contents": UV_LOCK.read_text(encoding="utf-8") if UV_LOCK.exists() else "",
    }


def new_run_id() -> str:
    """Generate a sortable unique run ID: {YYYYMMDD-HHMMSS}-{uuid6}.

    Why: timestamp prefix keeps runs in chronological order in the filesystem;
    the UUID suffix defends against same-second collisions.
    """
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{uuid.uuid4().hex[:6]}"


def params_hash(params: dict, n: int = 8) -> str:
    """8-char SHA-256 of the sorted-JSON representation of a params dict.

    Why: makes "have I already tried this exact config?" a one-liner:
    (runs['params_hash'] == h).any(). Sort ensures dict-order differences
    don't produce spurious mismatches.
    """
    payload = json.dumps(params, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:n]


def save_run(run_dict: dict, artifacts: dict, run_id: str) -> Path:
    """Persist one experiment run.

    Writes heavy artifacts to experiments/runs/{run_id}/ and appends a scalar
    summary row to experiments/runs.csv. If run_dict introduces new columns
    vs. the existing CSV header (e.g. data_version added mid-project), the
    file is rewritten via outer join so old rows get NaN in new columns rather
    than the CSV becoming misaligned.
    """
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "params.json", "w", encoding="utf-8") as f:
        json.dump(artifacts["params"], f, indent=2, default=str)
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(artifacts["metrics"], f, indent=2, default=str)

    np.save(run_dir / "oof_proba.npy",        artifacts["oof_proba"])
    np.save(run_dir / "test_proba_folds.npy", artifacts["test_proba_folds"])
    np.save(run_dir / "test_proba_mean.npy",  artifacts["test_proba_mean"])

    if artifacts.get("feature_importance") is not None:
        artifacts["feature_importance"].to_csv(run_dir / "feature_importance.csv", index=False)

    (run_dir / "environment.txt").write_text(artifacts["environment_text"], encoding="utf-8")
    (run_dir / "git_diff.patch").write_text(artifacts["git_diff"],          encoding="utf-8")
    (run_dir / "notes.md").write_text(artifacts["notes"],                   encoding="utf-8")

    if artifacts.get("models"):
        models_dir = run_dir / "models"
        models_dir.mkdir(exist_ok=True)
        for i, model in enumerate(artifacts["models"]):
            joblib.dump(model, models_dir / f"fold_{i}.joblib")

    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    new_row = pd.DataFrame([run_dict])
    if RUNS_CSV.exists():
        existing_cols = list(pd.read_csv(RUNS_CSV, nrows=0).columns)
        if existing_cols == list(new_row.columns):
            new_row.to_csv(RUNS_CSV, mode="a", header=False, index=False)
        else:
            combined = pd.concat([pd.read_csv(RUNS_CSV), new_row], ignore_index=True)
            combined.to_csv(RUNS_CSV, index=False)
    else:
        new_row.to_csv(RUNS_CSV, index=False)

    return run_dir


def load_runs() -> pd.DataFrame:
    """Read runs.csv sorted by oof_score descending.

    Returns an empty DataFrame if no runs have been logged yet, so display
    cells don't error before the first run.
    """
    if not RUNS_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(RUNS_CSV)
    if "oof_score" in df.columns:
        df = df.sort_values("oof_score", ascending=False).reset_index(drop=True)
    return df


def delete_run(run_id: str) -> None:
    """Remove a run from runs.csv and delete its artifact directory.

    Both steps are attempted regardless of whether the other succeeds, so a
    half-persisted run (CSV row present but directory missing, or vice versa)
    can still be cleaned up cleanly.

    Parameters
    ----------
    run_id : str
        The run ID to delete, as it appears in the run_id column of runs.csv.

    Why
    ---
    Provides a single call to fully retract a run — useful after a mis-fired
    save (wrong config, corrupted data, accidental duplicate) or when pruning
    old runs to reclaim disk space.
    """
    removed_csv = False
    removed_dir = False

    # --- Remove row from runs.csv ---
    if RUNS_CSV.exists():
        df = pd.read_csv(RUNS_CSV)
        if run_id in df["run_id"].values:
            df = df[df["run_id"] != run_id]
            df.to_csv(RUNS_CSV, index=False)
            removed_csv = True
        else:
            print(f"Warning: run_id '{run_id}' not found in runs.csv — skipping CSV step.")

    # --- Delete artifact directory ---
    run_dir = RUNS_DIR / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
        removed_dir = True
    else:
        print(f"Warning: artifact directory '{run_dir}' not found — skipping directory step.")

    if removed_csv or removed_dir:
        parts = []
        if removed_csv:
            parts.append("runs.csv row")
        if removed_dir:
            parts.append(f"artifact directory ({run_dir})")
        print(f"Deleted {run_id}: {' and '.join(parts)}.")