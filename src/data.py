"""Raw-data loading, categorical encoding, and processed-parquet persistence.

The single public function prepare_data() is the canonical entry point for
turning the competition CSVs into the encoded DataFrames used by all downstream
notebooks. FEATURES is the canonical list of raw column names (before encoding).

Two encodings are supported and cached to separate parquets:
- encoding="onehot" (default): each categorical level becomes its own 0/1 dummy
  column. Best for linear models; the historical default for every prior run.
- encoding="native": categorical columns are kept as raw strings under pandas
  `category` dtype. LightGBM consumes these directly (optimal split search on
  categories); CatBoost takes the raw strings via cat_features. This is the
  data path for the native-categoricals work in docs/fe_ideas.md §1.

Usage
-----
    from src.data import prepare_data, FEATURES

    train_df, test_df = prepare_data()                     # one-hot (default)
    train_df, test_df = prepare_data(encoding='native')    # category dtype
    encoded_features = [c for c in train_df.columns if c not in ('id', 'Churn')]

Design notes
------------
- If the processed parquets already exist, they are loaded directly so re-running
  the notebook cell is fast and idempotent.  Pass force=True to bypass the cache
  and re-encode from the raw CSVs (needed after changing FEATURES or _encode).
  Each encoding caches to its own parquet (train_df.parquet for one-hot,
  train_df_native.parquet for native) so the two never overwrite one another.
- Raw CSVs are resolved from data/raw/ if present, falling back to data/ for
  backwards compatibility with the pre-restructure layout.
- Binary string columns (≤2 distinct values): encoded with drop_first=True —
  one dummy column is sufficient and avoids a linearly-dependent column pair.
- Multi-class string columns (≥3 distinct values): all dummy columns are kept
  (no drop_first) so no level is silently used as a reference category; the
  models we use (tree-based and regularised LR) handle this without issue.
- Native encoding leaves numeric columns (tenure, MonthlyCharges, TotalCharges,
  SeniorCitizen) untouched and converts only the string columns to category
  dtype, so train and test share an identical schema without a fitted encoder.
- Churn is mapped Yes → 1, No → 0 in train_df only; test_df has no Churn column.
- Parquets are written via pyarrow directly (not pandas.to_parquet) to avoid a
  pandas/pyarrow version-mismatch in pandas' patch_pyarrow() wrapper.  pyarrow
  round-trips category dtype losslessly (stored as a dictionary-encoded column).
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


def _encode_native(df: pd.DataFrame) -> pd.DataFrame:
    """Keep categorical features as raw strings under pandas `category` dtype.

    Only the string-typed FEATURES are converted; numeric columns (tenure,
    MonthlyCharges, TotalCharges, SeniorCitizen) and id/Churn are left as-is.

    Why category dtype rather than a fitted encoder: the conversion depends only
    on the column's values, so applying it to train and test separately yields
    the same dtype.  The category *levels* are whatever appears in each frame —
    LightGBM keys on the raw category strings, so it matches train↔test levels
    by value (no positional code mismatch across frames or CV folds).
    """
    df = df.copy()
    str_cols = [
        f for f in FEATURES
        if f in df.columns and pd.api.types.is_string_dtype(df[f])
    ]
    for col in str_cols:
        df[col] = df[col].astype("category")
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_ENCODERS = {"onehot": _encode, "native": _encode_native}


def prepare_data(
    force: bool = False, encoding: str = "onehot"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, encode, and cache the competition data as processed parquets.

    If the processed parquets already exist in data/processed/ they are loaded
    and returned immediately (fast path). Pass force=True to re-run from raw
    CSVs, for example after modifying FEATURES or the encoder.

    Parameters
    ----------
    force : bool
        Bypass the parquet cache and re-encode from raw CSVs.
    encoding : {"onehot", "native"}
        "onehot" (default) one-hot-encodes the categorical columns via _encode.
        "native" keeps them as raw strings under `category` dtype via
        _encode_native — the data path for LightGBM/CatBoost native categorical
        handling.  Each encoding caches to its own parquet, so switching between
        them never invalidates the other's cache.

    Returns
    -------
    train_df : pd.DataFrame
        Training frame with columns: id, Churn (0/1), and the encoded feature
        columns (dummy columns for "onehot"; raw category columns for "native").
    test_df : pd.DataFrame
        Test frame with columns: id and the encoded feature columns (no Churn).

    Notes
    -----
    To derive the list of model input features (what you pass to the model):
        encoded_features = [c for c in train_df.columns if c not in ('id', 'Churn')]
    """
    if encoding not in _ENCODERS:
        raise ValueError(
            f"encoding must be one of {sorted(_ENCODERS)}, got {encoding!r}"
        )

    project_root  = _find_project_root()
    processed_dir = project_root / "data" / "processed"
    suffix        = "" if encoding == "onehot" else f"_{encoding}"
    train_path    = processed_dir / f"train_df{suffix}.parquet"
    test_path     = processed_dir / f"test_df{suffix}.parquet"

    # --- Fast path: load cached parquets ---
    if not force and train_path.exists() and test_path.exists():
        train_df = pd.read_parquet(train_path)
        test_df  = pd.read_parquet(test_path)
        print(f"Loaded from cache ({encoding}): "
              f"train_df {train_df.shape}, test_df {test_df.shape}")
        return train_df, test_df

    # --- Slow path: encode from raw CSVs and cache ---
    encode   = _ENCODERS[encoding]
    raw_dir  = _resolve_raw_dir(project_root)
    train_df = pd.read_csv(raw_dir / "train.csv")
    test_df  = pd.read_csv(raw_dir / "test.csv")

    train_df = encode(train_df)
    test_df  = encode(test_df)

    # Churn target: Yes → 1, No → 0  (test_df has no Churn column)
    train_df["Churn"] = train_df["Churn"].map({"Yes": 1, "No": 0})

    processed_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(train_df, preserve_index=False), train_path)
    pq.write_table(pa.Table.from_pandas(test_df,  preserve_index=False), test_path)
    print(f"Preprocessed and saved ({encoding}): "
          f"train_df {train_df.shape}, test_df {test_df.shape}")

    return train_df, test_df
