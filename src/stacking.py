"""Level-2 (stacked) feature construction: attach base-model OOF probabilities
to a base feature set, producing a new cached data_version.

This is **stacking via meta-features** (a.k.a. model-as-feature-extractor / the
"TFM-as-feature-extractor" pattern in docs/fe_ideas.md §3): every *training* row
gets each diverse base model's leakage-free **out-of-fold** probability, and
every *test* row gets that model's **bagged test-mean** probability. A downstream
GBDT then trains on the original features *plus* these probability columns.

Why it is leakage-safe
----------------------
The training-row probabilities are out-of-fold under the canonical
StratifiedKFold(5, shuffle, seed 42): no row's OOF value was produced by a model
that trained on that row. As a guard, before attaching each column we re-assert
that the loaded OOF reproduces the run's logged `oof_roc_auc` to ~1e-9 (the same
row-alignment check as scripts/check_oof_alignment.py and blending.ipynb). The
downstream model should be cross-validated on the *same* seed-42 folds (which
run_cv_experiment does), keeping the level-2 estimate honest.

This is NOT pseudo-labelling. Pseudo-labelling appends *test* rows carrying
*predicted labels* to the training set; stacking-as-features adds *columns* and
changes no rows. The two are complementary.
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

from src.tracking import DATA_DIR, RUNS_CSV, RUNS_DIR


def build_stacked_dataset(
    base_version: str,
    oof_columns: dict[str, str],
    out_version: str,
    target: str = "Churn",
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (and cache) a data_version = base features + level-2 OOF columns.

    Parameters
    ----------
    base_version : str
        The base FE data_version to extend (e.g. "fe_v4_native"); its parquets
        data/processed/{train,test}_df_{base_version}.parquet must exist.
    oof_columns : dict[str, str]
        Ordered mapping {new_column_name: run_id}. For each entry, the run's
        oof_proba.npy fills the column on train rows and test_proba_mean.npy
        fills it on test rows. Pick *diverse* base models — GBDT-on-GBDT
        probabilities add nothing (see the diversity analysis in the notebook).
    out_version : str
        data_version label for the result (e.g. "fe_v5_stack"); cached to
        data/processed/{train,test}_df_{out_version}.parquet.
    target : str
        Name of the label column in the base train parquet (used for the
        row-alignment assertion). Default "Churn".
    force : bool
        Rebuild even if the output parquets already exist.

    Returns
    -------
    (train_df, test_df) with the OOF columns appended (category dtype preserved
    from the base; the new columns are float64 probabilities).
    """
    train_path = DATA_DIR / f"train_df_{out_version}.parquet"
    test_path  = DATA_DIR / f"test_df_{out_version}.parquet"
    if not force and train_path.exists() and test_path.exists():
        print(f"Loaded cached stacked data ({out_version}).")
        return pd.read_parquet(train_path), pd.read_parquet(test_path)

    base_train_path = DATA_DIR / f"train_df_{base_version}.parquet"
    base_test_path  = DATA_DIR / f"test_df_{base_version}.parquet"
    for p in (base_train_path, base_test_path):
        if not p.exists():
            raise FileNotFoundError(f"Base parquet not found: {p}")

    train_df = pd.read_parquet(base_train_path)
    test_df  = pd.read_parquet(base_test_path)
    y = train_df[target].to_numpy()
    runs = pd.read_csv(RUNS_CSV).set_index("run_id")

    for col, run_id in oof_columns.items():
        if run_id not in runs.index:
            raise KeyError(f"{run_id} (for column {col!r}) not in runs.csv")
        run_dir = RUNS_DIR / run_id
        oof = np.load(run_dir / "oof_proba.npy")
        tst = np.load(run_dir / "test_proba_mean.npy")
        if len(oof) != len(train_df) or len(tst) != len(test_df):
            raise ValueError(
                f"{col} ({run_id}): length mismatch — oof {len(oof)} vs train "
                f"{len(train_df)}, test {len(tst)} vs {len(test_df)}. "
                "This run did not predict the full train/test set; pick a "
                "full-set (e.g. subfold-bagged) run instead."
            )
        # Row-alignment guard: the OOF must reproduce the logged ROC-AUC, proving
        # it is ordered like the local train data before we attach it as a column.
        logged = float(runs.loc[run_id, "oof_roc_auc"])
        recomputed = roc_auc_score(y, oof)
        if abs(recomputed - logged) > 1e-9:
            raise AssertionError(
                f"{col} ({run_id}): OOF not row-aligned — recomputed AUC "
                f"{recomputed:.9f} != logged {logged:.9f}. Refusing to attach."
            )
        train_df[col] = oof.astype("float64")
        test_df[col]  = tst.astype("float64")
        print(f"  + {col:14s} <- {runs.loc[run_id, 'tag']:30s} (OOF AUC {recomputed:.5f})")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(train_df, preserve_index=False), train_path)
    pq.write_table(pa.Table.from_pandas(test_df,  preserve_index=False), test_path)
    print(f"Built and cached stacked data ({out_version}): "
          f"train {train_df.shape}, test {test_df.shape}, "
          f"+{len(oof_columns)} OOF columns.")
    return train_df, test_df
