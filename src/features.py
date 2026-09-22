"""Feature engineering for the churn models, keyed by ``data_version``.

``build_features(data_version)`` returns the train/test frames a run with that
``data_version`` trained on, caching them as
``data/processed/{train,test}_df_{data_version}.parquet``.

All features here are **stateless** (row-wise), so they are computed on the full
train and test frames before any split without leaking information across folds.
Stateful transforms (target / frequency encoding, scaling, binning fitted on
data) belong inside the cross-validation loop, e.g. in a model pipeline.

Cache safety
------------
The cache key is the ``data_version`` label plus a hash of the feature function's
source code, stored in a small sidecar file next to the parquets. If the function
changes, the hash no longer matches and the parquets are rebuilt, so a stale
cache can never be served silently under an old label. (Bump the label anyway
when a change is meant to produce a new, separately logged feature set.)
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Callable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data import prepare_data
from src.tracking import DATA_DIR


def _cross(a: pd.Series, b: pd.Series) -> pd.Series:
    """Categorical interaction of two columns, e.g. 'Month-to-month | Fiber optic'."""
    return (a.astype(str) + " | " + b.astype(str)).astype("category")


def fe_v4_native(df: pd.DataFrame) -> pd.DataFrame:
    """The best-performing feature set ("min3") on the native-categorical base.

    Adds three features to the 19 raw columns:

    * ``AverageMonthly`` = TotalCharges / tenure — what the customer has actually
      paid per month, which diverges from ``MonthlyCharges`` after plan changes.
      ``tenure`` is clipped at 1 so a brand-new customer (tenure 0, present in the
      original IBM data though not in this competition's data) cannot divide by 0.
    * ``contract_x_payment`` and ``contract_x_internet`` — the two interactions
      that carried signal. Larger sets of crosses scored worse: the target is close
      to additive in the raw features, so extra crosses added variance, not signal.
    """
    df = df.copy()
    df["AverageMonthly"] = df["TotalCharges"] / df["tenure"].clip(lower=1)
    df["contract_x_payment"] = _cross(df["Contract"], df["PaymentMethod"])
    df["contract_x_internet"] = _cross(df["Contract"], df["InternetService"])
    return df


# data_version -> (base encoding passed to prepare_data, feature function)
FEATURE_SETS: dict[str, tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]] = {
    "fe_v4_native": ("native", fe_v4_native),
}


def engineer_features(df: pd.DataFrame, data_version: str = "fe_v4_native") -> pd.DataFrame:
    """Apply the feature function registered for ``data_version`` to one frame."""
    try:
        _, fn = FEATURE_SETS[data_version]
    except KeyError:
        raise ValueError(f"unknown data_version {data_version!r}; "
                         f"known: {sorted(FEATURE_SETS)}") from None
    return fn(df)


def feature_hash(data_version: str) -> str:
    """Short SHA-256 of the feature function's source (the cache-invalidation key)."""
    _, fn = FEATURE_SETS[data_version]
    return hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()[:12]


def _write(df: pd.DataFrame, path) -> None:
    # pyarrow directly (not DataFrame.to_parquet): sidesteps a pandas>=3 / pyarrow
    # version-mismatch shim, and matches how every other cache in the repo is written.
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)


def build_features(data_version: str = "fe_v4_native", force: bool = False,
                   data_dir=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(train_df, test_df)`` for ``data_version``, building the cache if needed.

    Rows stay in raw ``train.csv`` / ``test.csv`` order (nothing here reorders), so
    the frames line up with the shared cross-validation folds and every saved OOF
    vector.

    If parquets exist without a hash sidecar (caches written before this module
    existed), the features are recomputed and compared with the cached frames; on
    an exact match only the sidecar is written, so the parquet bytes — and hence
    the ``data_hash`` recorded for past runs — are left unchanged.
    """
    data_dir = DATA_DIR if data_dir is None else data_dir
    encoding, _ = FEATURE_SETS[data_version]
    train_path = data_dir / f"train_df_{data_version}.parquet"
    test_path = data_dir / f"test_df_{data_version}.parquet"
    hash_path = data_dir / f"{data_version}.fe_hash"
    want = feature_hash(data_version)

    cached = train_path.exists() and test_path.exists()
    if cached and not force and hash_path.exists() and hash_path.read_text().strip() == want:
        return pd.read_parquet(train_path), pd.read_parquet(test_path)

    base_train, base_test = prepare_data(encoding=encoding)
    train = engineer_features(base_train, data_version)
    test = engineer_features(base_test, data_version)

    if cached and not force and not hash_path.exists():
        old_train, old_test = pd.read_parquet(train_path), pd.read_parquet(test_path)
        if old_train.equals(train) and old_test.equals(test):
            hash_path.write_text(want)
            return old_train, old_test

    data_dir.mkdir(parents=True, exist_ok=True)
    _write(train, train_path)
    _write(test, test_path)
    hash_path.write_text(want)
    print(f"Built features {data_version}: train {train.shape}, test {test.shape}")
    return train, test
