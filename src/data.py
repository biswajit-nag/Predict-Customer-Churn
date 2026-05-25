"""Raw-data loading, one-hot encoding, and processed-parquet persistence.

The single public function prepare_data() is the canonical entry point for
turning the competition CSVs into the encoded DataFrames used by all downstream
notebooks. FEATURES is the canonical list of raw column names (before encoding).

Usage
-----
    from src.data import prepare_data, FEATURES

    train_df, test_df = prepare_data()
    encoded_features = [c for c in train_df.columns if c not in ('id', 'Churn')]

Design notes
------------
- If the processed parquets already exist, they are loaded directly so re-running
  the notebook cell is fast and idempotent.  Pass force=True to bypass the cache
  and re-encode from the raw CSVs (needed after changing FEATURES or _encode).
- Raw CSVs are resolved from data/raw/ if present, falling back to data/ for
  backwards compatibility with the pre-restructure layout.
- Binary string columns (≤2 distinct values): encoded with drop_first=True —
  one dummy column is sufficient and avoids a linearly-dependent column pair.
- Multi-class string columns (≥3 distinct values): all dummy columns are kept
  (no drop_first) so no level is silently used as a reference category; the
  models we use (tree-based and regularised LR) handle this without issue.
- Churn is mapped Yes → 1, No → 0 in train_df only; test_df has no Churn column.
- Parquets are written via pyarrow directly (not pandas.to_parquet) to avoid a
  pandas/pyarrow version-mismatch in pandas' patch_pyarrow() wrapper.
"""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Feature specification — single source of truth
# ---------------------------------------------------------------------------

# Raw column names that are used as model inputs (before one-hot encoding).
# Everything not in this list, and not 'id' / 'Churn', is dropped.
FEATURES: list[str] = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_project_root() -> Path:
    """Walk up from this file until pyproject.toml is found.

    Why: path constants must resolve correctly regardless of whether the caller
    is a notebook (CWD = project root) or a script in a subdirectory.
    """
    for candidate in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate project root (no pyproject.toml found in any parent directory)"
    )


def _resolve_raw_dir(project_root: Path) -> Path:
    """Return the directory containing train.csv and test.csv.

    Why: the planned directory restructure moves raw CSVs from data/ to
    data/raw/.  Checking data/raw/ first means the post-migration layout is
    preferred automatically, without breaking notebooks that run before the
    migration.
    """
    raw_dir = project_root / "data" / "raw"
    if raw_dir.exists() and (raw_dir / "train.csv").exists():
        return raw_dir
    legacy = project_root / "data"
    if (legacy / "train.csv").exists():
        return legacy
    raise FileNotFoundError(
        f"train.csv not found in {raw_dir} or {legacy}. "
        "Run `python data/fetch_data.py` to download the competition data."
    )


def _encode(df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic one-hot encoding to a raw competition dataframe.

    Binary string columns (≤2 unique values in the *passed* dataframe) are
    encoded with drop_first=True; multi-class string columns keep all dummies.

    Why pass the dataframe rather than use a fitted encoder: the encoding is
    purely structural (it depends only on column names and category counts, not
    on per-row statistics), so applying it to train and test separately produces
    identical column schemas without any leakage risk.
    """
    binary_str_cols = [
        f for f in FEATURES
        if f in df.columns
        and pd.api.types.is_string_dtype(df[f])
        and df[f].nunique() <= 2
    ]
    multi_str_cols = [
        f for f in FEATURES
        if f in df.columns
        and pd.api.types.is_string_dtype(df[f])
        and df[f].nunique() >= 3
    ]
    df = pd.get_dummies(df, columns=binary_str_cols, drop_first=True, dtype=int)
    df = pd.get_dummies(df, columns=multi_str_cols,  dtype=int)
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prepare_data(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, encode, and cache the competition data as processed parquets.

    If the processed parquets already exist in data/processed/ they are loaded
    and returned immediately (fast path). Pass force=True to re-run from raw
    CSVs, for example after modifying FEATURES or _encode().

    Parameters
    ----------
    force : bool
        Bypass the parquet cache and re-encode from raw CSVs.

    Returns
    -------
    train_df : pd.DataFrame
        One-hot-encoded training frame with columns: id, Churn (0/1), and all
        encoded feature columns produced by _encode().
    test_df : pd.DataFrame
        One-hot-encoded test frame with columns: id and all encoded feature
        columns (no Churn column).

    Notes
    -----
    To derive the list of model input features (what you pass to the model):
        encoded_features = [c for c in train_df.columns if c not in ('id', 'Churn')]
    """
    project_root  = _find_project_root()
    processed_dir = project_root / "data" / "processed"
    train_path    = processed_dir / "train_df.parquet"
    test_path     = processed_dir / "test_df.parquet"

    # --- Fast path: load cached parquets ---
    if not force and train_path.exists() and test_path.exists():
        train_df = pd.read_parquet(train_path)
        test_df  = pd.read_parquet(test_path)
        print(f"Loaded from cache: train_df {train_df.shape}, test_df {test_df.shape}")
        return train_df, test_df

    # --- Slow path: encode from raw CSVs and cache ---
    raw_dir  = _resolve_raw_dir(project_root)
    train_df = pd.read_csv(raw_dir / "train.csv")
    test_df  = pd.read_csv(raw_dir / "test.csv")

    train_df = _encode(train_df)
    test_df  = _encode(test_df)

    # Churn target: Yes → 1, No → 0  (test_df has no Churn column)
    train_df["Churn"] = train_df["Churn"].map({"Yes": 1, "No": 0})

    processed_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(train_df, preserve_index=False), train_path)
    pq.write_table(pa.Table.from_pandas(test_df,  preserve_index=False), test_path)
    print(f"Preprocessed and saved: train_df {train_df.shape}, test_df {test_df.shape}")

    return train_df, test_df
