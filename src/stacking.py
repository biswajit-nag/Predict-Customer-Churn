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


class PerFoldMaskingClassifier:
    """Wrap a base GBDT to probe/repair the per-fold missingness asymmetry.

    The per-fold dataset is unavoidably asymmetric: a train row has at most 1/5
    per-fold columns filled (only its held-out fold is leakage-free), while test
    rows have all 5 filled. A tree fit on the 1/5-filled pattern then meets an
    all-5-filled pattern at test time it never trained on. Two label-free knobs:

    test_mask_bag : bool
        At predict time, present X in `n_folds` masked views — view k keeps only
        the `*_f{k}` per-fold columns and NaNs the other folds — and average the
        per-view probabilities. Every forward pass then has the train-like
        (<=1-per-model) density, and bagging recovers information across folds.
        Applied to BOTH validation (OOF) and test, so the OOF estimate is computed
        under the same inference scheme as the test prediction.
    train_dropout : float in [0, 1)
        At fit time, randomly NaN each present per-fold value with this
        probability — feature-dropout regularization against over-reliance on the
        per-fold columns. Prediction is left unmasked unless test_mask_bag is set.

    Designed for run_cv_experiment: exposes get_params/fit/predict_proba and stays
    out of the way of the LightGBM-importance check (no booster_ attribute).
    """

    def __init__(self, base_factory, perfold_cols, test_mask_bag=False,
                 train_dropout=0.0, n_folds=5, seed=42, params=None):
        self.base_factory = base_factory
        self.perfold_cols = list(perfold_cols)
        self.test_mask_bag = bool(test_mask_bag)
        self.train_dropout = float(train_dropout)
        self.n_folds = int(n_folds)
        self.seed = int(seed)
        self.params = dict(params or {})
        self._fold_of = {c: int(c.rsplit("_f", 1)[1]) for c in self.perfold_cols}

    def get_params(self, deep=True):
        return {"_mask_test_bag": self.test_mask_bag,
                "_train_dropout": self.train_dropout,
                "_mask_n_folds": self.n_folds, **self.params}

    def fit(self, X, y):
        Xf = X
        if self.train_dropout > 0:
            rng = np.random.RandomState(self.seed)
            drop = rng.random_sample(X[self.perfold_cols].shape) < self.train_dropout
            Xf = X.copy()
            Xf[self.perfold_cols] = X[self.perfold_cols].mask(drop)
        self.model_ = self.base_factory()
        self.model_.fit(Xf, y)
        self.classes_ = getattr(self.model_, "classes_", np.array([0, 1]))
        return self

    def predict_proba(self, X):
        if not self.test_mask_bag:
            return self.model_.predict_proba(X)
        acc = None
        for k in range(self.n_folds):
            drop = [c for c in self.perfold_cols if self._fold_of[c] != k]
            Xk = X.copy()
            Xk[drop] = np.nan
            p = self.model_.predict_proba(Xk)
            acc = p if acc is None else acc + p
        return acc / self.n_folds


def build_perfold_dataset(
    base_version: str,
    fold_run_ids: dict[str, str],
    out_version: str,
    target: str = "Churn",
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a data version = base features + **per-fold** OOF columns (5 per model).

    Unlike build_stacked_dataset (one collapsed OOF column per model), this expands
    each model into 5 columns `{prefix}_f0..f4` keyed on the canonical seed-42 folds:

      - **train** rows: column `{prefix}_f{k}` holds the model's held-out OOF
        prediction *only* for the rows in validation fold k, and NaN for the other
        4/5 of rows. So every train row has exactly one non-NaN column per model
        (its leakage-free OOF value); the in-sample folds are deliberately left NaN.
      - **test** rows: column `{prefix}_f{k}` holds that model's fold-k test
        prediction (`test_proba_folds[:, k]`); all 5 are filled.

    Why: each column is then "fold-k model's *out-of-sample* output" on both the
    train rows it's defined on and all test rows, so it has a consistent train/test
    distribution — removing the single-fold-OOF (train) vs bagged-mean (test)
    mismatch that a collapsed OOF column suffers. GBDTs ingest the 80%-NaN columns
    natively (learned default direction per split).

    Parameters
    ----------
    fold_run_ids : dict[str, str]
        {column_prefix: run_id}. Each run must have a full-length oof_proba.npy and
        a (n_test, 5) test_proba_folds.npy under the canonical seed-42 fold split.
    """
    train_path = DATA_DIR / f"train_df_{out_version}.parquet"
    test_path  = DATA_DIR / f"test_df_{out_version}.parquet"
    if not force and train_path.exists() and test_path.exists():
        print(f"Loaded cached per-fold data ({out_version}).")
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

    # Fold assignment per train row (merge on id so we never assume row order).
    fold_map = pd.read_csv(RUNS_DIR.parent / "cv_folds_seed42.csv.gz")
    folds = train_df[["id"]].merge(fold_map, on="id", how="left")["fold"].to_numpy()
    if np.isnan(folds).any():
        raise ValueError("Some train ids are missing from cv_folds_seed42.csv.gz")
    folds = folds.astype(int)
    n_splits = fold_map["fold"].nunique()

    for prefix, run_id in fold_run_ids.items():
        if run_id not in runs.index:
            raise KeyError(f"{run_id} (for {prefix!r}) not in runs.csv")
        run_dir = RUNS_DIR / run_id
        oof = np.load(run_dir / "oof_proba.npy")
        tpf = np.load(run_dir / "test_proba_folds.npy")
        if len(oof) != len(train_df) or tpf.shape != (len(test_df), n_splits):
            raise ValueError(
                f"{prefix} ({run_id}): expected oof {len(train_df)} and "
                f"test_proba_folds {(len(test_df), n_splits)}, got {len(oof)} / {tpf.shape}.")
        # Row-alignment guard: OOF must reproduce the logged ROC-AUC.
        recomputed = roc_auc_score(y, oof)
        logged = float(runs.loc[run_id, "oof_roc_auc"])
        if abs(recomputed - logged) > 1e-9:
            raise AssertionError(
                f"{prefix} ({run_id}): OOF not row-aligned ({recomputed:.9f} != {logged:.9f}).")
        for k in range(n_splits):
            # train: held-out OOF for fold-k rows only, NaN elsewhere
            train_df[f"{prefix}_f{k}"] = np.where(folds == k, oof, np.nan).astype("float64")
            # test: fold-k model's test prediction (always present)
            test_df[f"{prefix}_f{k}"] = tpf[:, k].astype("float64")
        print(f"  + {prefix:8s} <- {runs.loc[run_id, 'tag']:30s} "
              f"({n_splits} per-fold cols, OOF AUC {recomputed:.5f})")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(train_df, preserve_index=False), train_path)
    pq.write_table(pa.Table.from_pandas(test_df,  preserve_index=False), test_path)
    n_new = len(fold_run_ids) * n_splits
    print(f"Built and cached per-fold data ({out_version}): "
          f"train {train_df.shape}, test {test_df.shape}, +{n_new} per-fold columns "
          f"(train cols ~{100/n_splits:.0f}% filled, test cols 100% filled).")
    return train_df, test_df
