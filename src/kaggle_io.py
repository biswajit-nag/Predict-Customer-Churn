"""Kaggle submission + leaderboard-score retrieval for logged runs.

The competition is scored on ROC-AUC. Every logged run stores the bagged test
probabilities at experiments/runs/{run_id}/test_proba_mean.npy; this module turns
those into a Kaggle submission, uploads it via the Kaggle CLI, retrieves the
public (and, when available, private) leaderboard score, and writes both back
into the run's row in experiments/runs.csv (the lb_public / lb_private columns).

Why shell out to the `kaggle` CLI rather than import the kaggle package:
the CLI is installed and on PATH here, while the Python package is not in the
project venv. The CLI reads credentials from the KAGGLE_USERNAME / KAGGLE_KEY
environment variables or ~/.kaggle/kaggle.json — see docs/kaggle_setup.md.

Public API
----------
submit_run(run_id, ...)         Submit one logged run and (optionally) collect its score.
make_submission(run_id, ...)    Build the id,Churn submission CSV for a run (no upload).
fetch_scores(run_id, ...)       Poll the Kaggle submissions list for a run's score.
update_lb_in_runs_csv(...)      Write lb_public / lb_private back into runs.csv.
credentials_available()         True if the CLI can authenticate (env vars or kaggle.json).
"""

import csv
import io
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.tracking import DATA_DIR, PROJECT_ROOT, RUNS_CSV, RUNS_DIR

COMPETITION = "playground-series-s6e3"
N_TEST = 254_655  # canonical test-set row count (must match test_proba_mean length)


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

_CREDS_PROBE_CACHE: bool | None = None


def credentials_available(force: bool = False) -> bool:
    """True if the Kaggle CLI is on PATH and can actually authenticate.

    Credentials may come from KAGGLE_USERNAME + KAGGLE_KEY, a ~/.kaggle/kaggle.json
    token file, or a CLI-managed store this process can't see directly (the CLI
    resolves them in ways that don't always surface as env vars in a child shell).
    So rather than guess from env/file presence, we treat the CLI itself as the
    source of truth: a cheap authenticated call that returns cleanly means we're
    good. The probe result is cached for the process (pass force=True to re-probe).
    """
    global _CREDS_PROBE_CACHE
    if shutil.which("kaggle") is None:
        return False
    # Fast path: explicit credentials we can see ourselves. Three supported forms:
    #  - KAGGLE_USERNAME + KAGGLE_KEY environment variables (API token via env);
    #  - ~/.kaggle/credentials.json  (the CLI's API-token store from `kaggle auth login`);
    #  - ~/.kaggle/kaggle.json       (legacy manual username+key file).
    kdir = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    if (kdir / "credentials.json").exists() or (kdir / "kaggle.json").exists():
        return True
    # Ground-truth probe: ask the CLI to hit an authenticated endpoint.
    if _CREDS_PROBE_CACHE is not None and not force:
        return _CREDS_PROBE_CACHE
    r = _kaggle(["competitions", "submissions", "-c", COMPETITION])
    out = ((r.stdout or "") + (r.stderr or "")).lower()
    # Only treat the run as 'no credentials' on a *definitive auth error*. A
    # transient failure (network, rate limit / 429, slow API) must NOT be reported
    # as missing creds — return True there and let the actual submit surface the
    # real error, rather than silently blocking a run whose creds are fine.
    auth_error = any(s in out for s in (
        "401", "unauthorized", "403", "forbidden",
        "could not find kaggle.json", "no api key", "credentials were not found",
    ))
    authed = not auth_error
    _CREDS_PROBE_CACHE = authed
    return authed


def _require_credentials() -> None:
    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "The `kaggle` CLI is not on PATH. Install it (`pip install kaggle`) "
            "and see docs/kaggle_setup.md."
        )
    if not credentials_available():
        raise RuntimeError(
            "Kaggle credentials not found. Set KAGGLE_USERNAME and KAGGLE_KEY "
            "(or place ~/.kaggle/kaggle.json). See docs/kaggle_setup.md."
        )


# ---------------------------------------------------------------------------
# Submission file construction
# ---------------------------------------------------------------------------

def _load_test_ids() -> np.ndarray:
    """Return the test-set `id` column in the canonical row order.

    The OOF/test row order is the raw test.csv order; every processed test
    parquet preserves it (src/data.py never reorders), so any of them serves as
    the id source. We try a few names for robustness.
    """
    for name in ("test_df.parquet", "test_df_native.parquet", "test_df_fe_v0.parquet"):
        path = DATA_DIR / name
        if path.exists():
            ids = pd.read_parquet(path, columns=["id"])["id"].to_numpy()
            if len(ids) == N_TEST:
                return ids
    raise FileNotFoundError(
        f"No processed test parquet with {N_TEST} rows found in {DATA_DIR}. "
        "Run prepare_data() / the FE cell first."
    )


def make_submission(run_id: str, out_path: Path | None = None) -> Path:
    """Write the id,Churn submission CSV for a logged run from test_proba_mean.

    Parameters
    ----------
    run_id : str
        The run whose experiments/runs/{run_id}/test_proba_mean.npy is submitted.
    out_path : Path, optional
        Destination CSV. Defaults to experiments/runs/{run_id}/submission.csv,
        co-located with the run's other artifacts.

    Returns
    -------
    Path to the written submission CSV.
    """
    run_dir = RUNS_DIR / run_id
    proba_path = run_dir / "test_proba_mean.npy"
    if not proba_path.exists():
        raise FileNotFoundError(f"{proba_path} not found — cannot build a submission.")
    proba = np.load(proba_path)
    ids = _load_test_ids()
    if len(proba) != len(ids):
        raise ValueError(
            f"{run_id}: test_proba_mean has {len(proba)} rows but the test set "
            f"has {len(ids)} — this run did not predict the full test set."
        )
    out_path = out_path or (run_dir / "submission.csv")
    pd.DataFrame({"id": ids, "Churn": proba}).to_csv(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# Kaggle CLI wrappers
# ---------------------------------------------------------------------------

def _kaggle(args: list[str]) -> subprocess.CompletedProcess:
    """Run a `kaggle ...` command, captured as UTF-8 text (never raises on exit)."""
    return subprocess.run(
        ["kaggle", *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
    )


def _submit(submission_csv: Path, message: str) -> None:
    """Upload one submission. Raises with the CLI message on any failure."""
    r = _kaggle(["competitions", "submit", "-c", COMPETITION,
                 "-f", str(submission_csv), "-m", message])
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0 or "Successfully submitted" not in out:
        # Surface daily-limit and auth errors verbatim so the caller can react.
        raise RuntimeError(f"Kaggle submit failed for '{message}':\n{out.strip()}")


def _submissions_table() -> list[dict]:
    """Return recent submissions for the competition as a list of dict rows.

    Columns include: fileName, date, description, status, publicScore, privateScore.
    """
    r = _kaggle(["competitions", "submissions", "-c", COMPETITION, "--csv"])
    if r.returncode != 0 or not r.stdout:
        raise RuntimeError(f"Could not list submissions:\n{(r.stderr or '').strip()}")
    # The CLI may emit a deprecation/notice line before the CSV. Locate the real
    # header line and parse from there — crucially, keep the *whole* header
    # (incl. the leading `ref` column) so columns don't shift against the rows.
    lines = r.stdout.splitlines()
    hdr = next((i for i, ln in enumerate(lines)
                if "fileName" in ln and "publicScore" in ln), 0)
    return list(csv.DictReader(io.StringIO("\n".join(lines[hdr:]))))


def _score_from_row(row: dict) -> tuple[float, float]:
    def _f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan
    return _f(row.get("publicScore")), _f(row.get("privateScore"))


def fetch_scores(run_id: str, wait: bool = True, timeout: int = 900,
                 poll: int = 20) -> tuple[float, float]:
    """Find this run's submission in the Kaggle list and return (public, private).

    Submissions are matched on the run_id embedded in the description (see
    submit_run). When wait=True we poll until the matching submission's status
    is 'complete' (or timeout), so the scores are populated on return. A missing
    private score (competition still open) comes back as NaN.
    """
    _require_credentials()
    deadline = time.time() + timeout
    while True:
        rows = [r for r in _submissions_table() if run_id in (r.get("description") or "")]
        # Kaggle reports status like "SubmissionStatus.COMPLETE" / ".PENDING" /
        # ".ERROR"; match the COMPLETE state as a case-insensitive substring.
        complete = [r for r in rows if "complete" in (r.get("status") or "").lower()]
        errored = [r for r in rows if "error" in (r.get("status") or "").lower()]
        if errored and not complete:
            raise RuntimeError(
                f"Kaggle reported an ERROR scoring the submission for {run_id}: "
                f"{errored[0].get('description')}"
            )
        if complete:
            return _score_from_row(complete[0])  # most recent first
        if not wait or time.time() > deadline:
            if rows:  # found it but still scoring
                return _score_from_row(rows[0])
            return (np.nan, np.nan)
        time.sleep(poll)


# ---------------------------------------------------------------------------
# runs.csv write-back
# ---------------------------------------------------------------------------

def update_lb_in_runs_csv(run_id: str, public: float, private: float) -> None:
    """Write lb_public / lb_private for one run into experiments/runs.csv.

    Read-modify-write keyed on run_id; creates the columns if a pre-migration
    CSV lacks them. Leaves all other rows and columns untouched.
    """
    df = pd.read_csv(RUNS_CSV)
    for col in ("lb_public", "lb_private"):
        if col not in df.columns:
            df[col] = np.nan
    mask = df["run_id"] == run_id
    if not mask.any():
        raise KeyError(f"run_id {run_id} not found in {RUNS_CSV}")
    if public is not None and not (isinstance(public, float) and np.isnan(public)):
        df.loc[mask, "lb_public"] = public
    if private is not None and not (isinstance(private, float) and np.isnan(private)):
        df.loc[mask, "lb_private"] = private
    df.to_csv(RUNS_CSV, index=False)


# ---------------------------------------------------------------------------
# One-call submit + score + log
# ---------------------------------------------------------------------------

def submit_run(run_id: str, message: str | None = None, wait: bool = True,
               timeout: int = 900) -> tuple[float, float]:
    """Submit a logged run's predictions to Kaggle and record the LB score.

    Builds experiments/runs/{run_id}/submission.csv from test_proba_mean,
    uploads it (the run_id is embedded in the submission message so the score
    can be matched back), and — when wait=True — blocks until Kaggle finishes
    scoring and writes lb_public / lb_private into runs.csv.

    Returns (public, private); private is NaN while the competition is open.
    """
    _require_credentials()
    runs = pd.read_csv(RUNS_CSV).set_index("run_id")
    tag = runs.loc[run_id, "tag"] if run_id in runs.index else run_id
    message = message or f"{tag} | run_id={run_id}"

    submission_csv = make_submission(run_id)
    print(f"[{run_id}] submitting {submission_csv.relative_to(PROJECT_ROOT)} ...")
    _submit(submission_csv, message)

    if not wait:
        print(f"[{run_id}] submitted (not waiting for score).")
        return (np.nan, np.nan)

    public, private = fetch_scores(run_id, wait=True, timeout=timeout)
    update_lb_in_runs_csv(run_id, public, private)
    priv = "n/a" if (isinstance(private, float) and np.isnan(private)) else f"{private:.6f}"
    print(f"[{run_id}] public={public:.6f}  private={priv}  -> written to runs.csv")
    return public, private
