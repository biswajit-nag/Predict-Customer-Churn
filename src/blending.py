"""Log post-hoc blends of base-run OOF predictions as first-class `runs.csv` rows.

`blending.ipynb` combines the logged base runs (equal/rank/strength means, bagged
hill climbing, logistic / LightGBM stacks) but only ever reports a single *pooled*
OOF ROC-AUC per strategy, and never persists the result. This module closes both
gaps, per docs/technical_review.md §2.2:

  * `fold_aucs` scores a blend's OOF vector **per canonical seed-42 fold**, so the
    blend gets the same `fold_roc_auc_mean` / `fold_roc_auc_std` every base run
    carries — making "does the +0.0003 gain clear the ~0.0009 fold noise?"
    answerable directly from the leaderboard.
  * `save_blend_run` writes a blend as a normal experiment run (same artifacts and
    `runs.csv` schema as `src/cv.py`), so it can be submitted to Kaggle and shows
    up alongside the base models.

A blend is a combination of *already-logged* OOF/test vectors, not a model fit over
folds, so a few `runs.csv` columns take blend-specific conventions (see
`_build_run_dict`): `data_version="blend_v1"` (a sentinel, not an FE recipe),
`data_hash` = the canonical train parquet (the shared OOF row ordering every member
aligns to), `parent_run_id=""` (a blend has N parents — the member list lives in
`params.json`), `model_class` = a synthetic scheme name (precedent:
`BaggedSubsampleTabPFN`), and `n_features` = the number of member columns.

The blend constructors (rank mean, softmax, hill climb, logit stack, LGBM stack)
live here too, so the notebook and `scripts/run_blend_logging.py` build identical
blends from one source.
"""

import datetime
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.cv import PRIMARY_METRIC_NAME
from src.tracking import (
    DATA_DIR,
    EXPERIMENTS_DIR,
    PROJECT_ROOT,
    RUNS_CSV,
    RUNS_DIR,
    environment_info,
    git_info,
    hash_file,
    new_run_id,
    params_hash,
    save_run,
    train_parquet_for,
)

CANON_N = 594_194   # full training-set size; runs with shorter OOF are subsamples
N_TEST = 254_655    # canonical test-set size

# model_class -> (broad family, engine). Mirrors blending.ipynb's TAXONOMY so the
# curated pool reproduces the notebook exactly.
TAXONOMY = {
    "LGBMClassifier":                ("Boosted trees", "LightGBM"),
    "XGBClassifier":                 ("Boosted trees", "XGBoost"),
    "CatBoostClassifier":            ("Boosted trees", "CatBoost"),
    "ExplainableBoostingClassifier": ("Boosted trees", "EBM"),
    "RandomForestClassifier":        ("Bagged trees",  "RandomForest"),
    "ExtraTreesClassifier":          ("Bagged trees",  "ExtraTrees"),
    "RealMLP_TD_Classifier":         ("Deep nets",     "RealMLP"),
    "SeedEnsembleRealMLP":           ("Deep nets",     "RealMLP"),
    "TabM_D_Classifier":             ("Deep nets",     "TabM"),
    "SeedEnsembleTabM":              ("Deep nets",     "TabM"),
    "BaggedSubsampleTabPFN":         ("In-context",    "TabPFN"),
    "BaggedSubsampleTabICL":         ("In-context",    "TabICL"),
    "Pipeline":                      ("Linear",        "Logistic"),
}

# Canonical runs.csv column order (matches src/cv.py run_dict). save_run appends
# without rewriting the file only when the new row's columns match exactly, so we
# build the dict in this order and leave existing rows untouched.
RUN_COLUMNS = [
    "run_id", "timestamp", "tag", "notes", "status", "parent_run_id", "git_hash",
    "git_dirty", "data_hash", "data_version", "python_version", "platform",
    "kaggle_run", "model_class", "params_hash", "cv_type", "n_splits",
    "random_state", "n_train", "n_test", "n_features", "n_pseudo", "metric",
    "oof_roc_auc", "fold_roc_auc_mean", "fold_roc_auc_std", "oof_accuracy",
    "fold_acc_mean", "fold_acc_std", "lb_public", "lb_private",
    "training_time_sec", "artifact_dir",
]


# ---------------------------------------------------------------------------
# Targets, folds, and the aligned base-run OOF/test matrices
# ---------------------------------------------------------------------------

def load_target_and_folds() -> tuple[np.ndarray, np.ndarray]:
    """Return (y, folds): the churn target and canonical seed-42 fold id per train row.

    Folds are merged on `id` (not assumed by row order) so the assignment is robust
    regardless of how the parquet is sorted — the same guard build_perfold_dataset
    uses.
    """
    train = pd.read_parquet(DATA_DIR / "train_df_fe_v0.parquet", columns=["id", "Churn"])
    fold_map = pd.read_csv(EXPERIMENTS_DIR / "cv_folds_seed42.csv.gz")
    folds = train[["id"]].merge(fold_map, on="id", how="left")["fold"].to_numpy()
    if np.isnan(folds).any():
        raise ValueError("Some train ids are missing from cv_folds_seed42.csv.gz")
    return train["Churn"].to_numpy(), folds.astype(int)


def load_aligned_runs(y: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load every full-set base run's OOF and test-mean vectors, row-aligned.

    Returns (oof, test, meta, spearman):
      * `oof`  : DataFrame (CANON_N rows x runs), columns = disambiguated labels.
      * `test` : DataFrame (N_TEST  rows x runs), same columns (test_proba_mean).
      * `meta` : per-run label, tag, model_class, data_version, oof_roc_auc, run_id,
                 family, engine, mean_rho (mean Spearman to all other runs).
      * `spearman` : runs x runs rank-correlation matrix.

    Runs whose OOF is not CANON_N rows (subsample EDA passes) are skipped — they do
    not align row-wise. Each kept run's OOF is re-asserted to reproduce its logged
    ROC-AUC (the row-alignment guard from scripts/check_oof_alignment.py) before use.
    """
    runs = pd.read_csv(RUNS_CSV).dropna(subset=["run_id"])
    # Exclude previously-logged blends: a blend must never be a member of a blend
    # (keeps load idempotent across re-runs of log_headline_blends).
    runs = runs[runs["data_version"] != "blend_v1"]
    runs = (runs[runs["status"] == "success"]
            .sort_values("oof_roc_auc", ascending=False)
            .reset_index(drop=True))

    oof_cols, test_cols, labels, kept, seen = [], [], [], [], set()
    for _, r in runs.iterrows():
        run_dir = RUNS_DIR / str(r["run_id"])
        proba = np.load(run_dir / "oof_proba.npy")
        if proba.shape[0] != CANON_N:
            continue
        recomputed = roc_auc_score(y, proba)
        if abs(recomputed - float(r["oof_roc_auc"])) > 1e-9:
            raise AssertionError(f"{r['run_id']} OOF not row-aligned "
                                 f"({recomputed:.9f} != {float(r['oof_roc_auc']):.9f})")
        label = r["tag"] if r["tag"] not in seen else f"{r['tag']}|{r['data_version']}"
        seen.add(r["tag"])
        oof_cols.append(proba)
        test_cols.append(np.load(run_dir / "test_proba_mean.npy"))
        labels.append(label)
        kept.append(r)

    oof = pd.DataFrame(np.column_stack(oof_cols), columns=labels)
    test = pd.DataFrame(np.column_stack(test_cols), columns=labels)
    meta = pd.DataFrame(kept)[["tag", "model_class", "data_version", "oof_roc_auc", "run_id"]]
    meta.insert(0, "label", labels)

    rho, _ = spearmanr(oof.values)
    spearman = pd.DataFrame(rho, index=labels, columns=labels)
    tax = meta["model_class"].map(TAXONOMY)
    meta["family"] = tax.str[0]
    meta["engine"] = tax.str[1]
    off = spearman.where(~np.eye(len(spearman), dtype=bool))
    meta["mean_rho"] = meta["label"].map(off.mean())
    return oof, test, meta, spearman


def curate_pool(meta: pd.DataFrame) -> list[str]:
    """Best-OOF-AUC run per engine, except the linear slot picked by diversity.

    Reproduces blending.ipynb's curation: one representative per engine (max
    oof_roc_auc), but the single Logistic slot is chosen by **lowest mean_rho**
    (most de-correlated) because a linear model never wins on standalone AUC here.
    """
    best_per_engine = meta.loc[meta.groupby("engine")["oof_roc_auc"].idxmax()]
    keep = [i for i in best_per_engine.index if meta.loc[i, "engine"] != "Logistic"]
    keep.append(meta.loc[meta["engine"] == "Logistic", "mean_rho"].idxmin())
    return (meta.loc[keep].sort_values("oof_roc_auc", ascending=False)["label"].tolist())


# ---------------------------------------------------------------------------
# Per-fold scoring
# ---------------------------------------------------------------------------

def fold_aucs(oof_proba: np.ndarray, y: np.ndarray, folds: np.ndarray
              ) -> tuple[list[float], float, float]:
    """Per-fold ROC-AUC of a full-length OOF vector, plus its mean and std.

    Scores `roc_auc_score(y[folds==k], oof_proba[folds==k])` for each fold k. This
    is exactly how src/cv.py computes a base run's per-fold AUC (on each fold's
    validation rows), so a blend's fold std is directly comparable to the base
    runs' ~0.0009.
    """
    per_fold = [float(roc_auc_score(y[folds == k], oof_proba[folds == k]))
                for k in sorted(np.unique(folds))]
    return per_fold, float(np.mean(per_fold)), float(np.std(per_fold))


def _fold_accuracies(oof_proba: np.ndarray, y: np.ndarray, folds: np.ndarray
                     ) -> tuple[list[float], float, float]:
    per_fold = [float(accuracy_score(y[folds == k], (oof_proba[folds == k] >= 0.5).astype(int)))
                for k in sorted(np.unique(folds))]
    return per_fold, float(np.mean(per_fold)), float(np.std(per_fold))


# ---------------------------------------------------------------------------
# Blend constructors — each returns (oof_vector, test_vector[, extra])
# ---------------------------------------------------------------------------

def logit(p: np.ndarray) -> np.ndarray:
    """Logit transform with clipping — the natural scale for stacking probabilities."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def equal_mean(oof: pd.DataFrame, test: pd.DataFrame, cols: list[str]):
    return oof[cols].to_numpy().mean(axis=1), test[cols].to_numpy().mean(axis=1)


def rank_mean(oof: pd.DataFrame, test: pd.DataFrame, cols: list[str]):
    """Average of within-set ranks, min-max scaled to [0, 1]. Order-preserving, so
    AUC is unchanged; the [0,1] scaling just keeps the submitted column tidy.
    AUC depends only on ordering, so this is the calibration-free way to average."""
    def _rm(df):
        r = np.column_stack([rankdata(df[c].to_numpy()) for c in cols]).mean(axis=1)
        return (r - r.min()) / (r.max() - r.min())
    return _rm(oof), _rm(test)


def softmax_auc(oof: pd.DataFrame, test: pd.DataFrame, cols: list[str], y: np.ndarray,
                T: float):
    """Strength-weighted mean with weights = softmax(member OOF AUC / T)."""
    aucs = np.array([roc_auc_score(y, oof[c].to_numpy()) for c in cols])
    w = np.exp((aucs - aucs.max()) / T)
    w /= w.sum()
    oof_p = (oof[cols].to_numpy() * w).sum(axis=1)
    test_p = (test[cols].to_numpy() * w).sum(axis=1)
    return oof_p, test_p, w


def hill_climb_weights(oof: pd.DataFrame, cols: list[str], y: np.ndarray,
                       n_iter: int = 50, n_bags: int = 25, frac: float = 0.5,
                       seed: int = 42) -> np.ndarray:
    """Bagged Caruana ensemble selection on OOF probabilities -> per-column weights."""
    M = oof[cols].to_numpy()
    m = len(cols)
    rng = np.random.RandomState(seed)
    freq = np.zeros(m)
    for _ in range(n_bags):
        sub = rng.choice(m, size=max(2, int(frac * m)), replace=False)
        start = max(sub, key=lambda j: roc_auc_score(y, M[:, j]))
        ens = M[:, start].copy()
        counts = np.zeros(m)
        counts[start] = 1
        n, cur = 1, roc_auc_score(y, ens)
        for _ in range(n_iter):
            best_j, best_s = None, cur
            for j in sub:
                s = roc_auc_score(y, (ens + M[:, j]) / (n + 1))
                if s > best_s:
                    best_s, best_j = s, j
            if best_j is None:
                break
            ens += M[:, best_j]
            counts[best_j] += 1
            n, cur = n + 1, best_s
        freq += counts / counts.sum()
    return freq / freq.sum()


def hill_climb(oof: pd.DataFrame, test: pd.DataFrame, cols: list[str], y: np.ndarray,
               **kw):
    w = hill_climb_weights(oof, cols, y, **kw)
    return oof[cols].to_numpy() @ w, test[cols].to_numpy() @ w, w


def logit_stack(oof: pd.DataFrame, test: pd.DataFrame, cols: list[str], y: np.ndarray,
                folds: np.ndarray, C: float):
    """L2 logistic stack on standardized logit-probabilities.

    OOF predictions are cross-validated over the canonical seed-42 folds (leakage-
    free); the test prediction refits the same recipe on all OOF rows and applies it
    to the members' bagged test means.
    """
    Xoof = logit(oof[cols].to_numpy())
    oof_p = np.zeros(len(y))
    for k in sorted(np.unique(folds)):
        tr, va = folds != k, folds == k
        sc = StandardScaler().fit(Xoof[tr])
        lr = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(Xoof[tr]), y[tr])
        oof_p[va] = lr.predict_proba(sc.transform(Xoof[va]))[:, 1]
    sc = StandardScaler().fit(Xoof)
    lr = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(Xoof), y)
    test_p = lr.predict_proba(sc.transform(logit(test[cols].to_numpy())))[:, 1]
    return oof_p, test_p, lr.coef_.ravel().tolist()


def _lgbm_stack_space(trial):
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 600),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        num_leaves=trial.suggest_int("num_leaves", 7, 63),
        max_depth=trial.suggest_int("max_depth", 2, 6),
        min_child_samples=trial.suggest_int("min_child_samples", 20, 500),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    )


def lgbm_stack(oof: pd.DataFrame, test: pd.DataFrame, cols: list[str], y: np.ndarray,
               folds: np.ndarray, params: dict | None = None, n_trials: int = 50):
    """LightGBM stack on the raw OOF probabilities, CV'd over the seed-42 folds.

    When `params` is None a shallow, strongly-regularised Optuna study (matching the
    notebook) is run first; pass pinned params to skip tuning. Returns
    (oof_vector, test_vector, params_used).
    """
    import lightgbm as lgb
    import optuna

    Xoof, Xtest = oof[cols].to_numpy(), test[cols].to_numpy()
    fixed = dict(random_state=42, n_jobs=-1, verbose=-1)

    if params is None:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            p = _lgbm_stack_space(trial)
            pred = np.zeros(len(y))
            for k in sorted(np.unique(folds)):
                tr, va = folds != k, folds == k
                mdl = lgb.LGBMClassifier(**p, **fixed).fit(Xoof[tr], y[tr])
                pred[va] = mdl.predict_proba(Xoof[va])[:, 1]
            return roc_auc_score(y, pred)

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        params = study.best_params

    oof_p = np.zeros(len(y))
    for k in sorted(np.unique(folds)):
        tr, va = folds != k, folds == k
        mdl = lgb.LGBMClassifier(**params, **fixed).fit(Xoof[tr], y[tr])
        oof_p[va] = mdl.predict_proba(Xoof[va])[:, 1]
    final = lgb.LGBMClassifier(**params, **fixed).fit(Xoof, y)
    test_p = final.predict_proba(Xtest)[:, 1]
    return oof_p, test_p, params


# ---------------------------------------------------------------------------
# Persist a blend as a runs.csv entry
# ---------------------------------------------------------------------------

def _build_run_dict(*, run_id, tag, notes, model_class, spec, members, n_features,
                    oof_proba, test_proba_mean, y, folds, data_version,
                    probability_scale, training_time_sec, parent_run_id,
                    kaggle_run) -> tuple[dict, dict]:
    """Assemble the runs.csv row + metrics.json for a blend (schema == src/cv.py)."""
    git, env = git_info(), environment_info()
    oof_roc_auc = float(roc_auc_score(y, oof_proba))
    _, fmean, fstd = fold_aucs(oof_proba, y, folds)

    if probability_scale:
        oof_acc = float(accuracy_score(y, (oof_proba >= 0.5).astype(int)))
        acc_per_fold, acc_mean, acc_std = _fold_accuracies(oof_proba, y, folds)
    else:
        # rank-mean is on a rank scale, not a probability — accuracy@0.5 is meaningless.
        oof_acc = acc_mean = acc_std = np.nan
        acc_per_fold = []

    run_dict = {
        "run_id":            run_id,
        "timestamp":         datetime.datetime.now().isoformat(timespec="seconds"),
        "tag":               tag,
        "notes":             notes,
        "status":            "success",
        "parent_run_id":     parent_run_id,
        "git_hash":          git["hash"],
        "git_dirty":         git["dirty"],
        "data_hash":         hash_file(train_parquet_for(data_version)),
        "data_version":      data_version,
        "python_version":    env["python_version"],
        "platform":          env["platform"],
        "kaggle_run":        bool(kaggle_run),
        "model_class":       model_class,
        "params_hash":       params_hash(spec),
        "cv_type":           "StratifiedKFold",
        "n_splits":          int(len(np.unique(folds))),
        "random_state":      42,
        "n_train":           int(len(y)),
        "n_test":            int(len(test_proba_mean)),
        "n_features":        int(n_features),
        "n_pseudo":          0,
        "metric":            PRIMARY_METRIC_NAME,
        "oof_roc_auc":        oof_roc_auc,
        "fold_roc_auc_mean":  fmean,
        "fold_roc_auc_std":   fstd,
        "oof_accuracy":       oof_acc,
        "fold_acc_mean":      acc_mean,
        "fold_acc_std":       acc_std,
        "lb_public":          np.nan,
        "lb_private":         np.nan,
        "training_time_sec":  float(training_time_sec),
        "artifact_dir":       str((RUNS_DIR / run_id).relative_to(PROJECT_ROOT)),
    }
    assert list(run_dict.keys()) == RUN_COLUMNS, "run_dict columns drifted from schema"

    per_fold, _, _ = fold_aucs(oof_proba, y, folds)
    metrics = {
        "oof_roc_auc":       oof_roc_auc,
        "fold_roc_aucs":     per_fold,
        "fold_roc_auc_mean": fmean,
        "fold_roc_auc_std":  fstd,
        "oof_accuracy":      oof_acc,
        "fold_scores":       acc_per_fold,
        "fold_acc_mean":     acc_mean,
        "fold_acc_std":      acc_std,
        "members":           members,
        "scheme":            spec,
        "cv_config": {"type": "StratifiedKFold", "n_splits": int(len(np.unique(folds))),
                      "random_state": 42, "shuffle": True},
    }
    return run_dict, metrics


def save_blend_run(*, tag, notes, model_class, oof_proba, test_proba_mean, members,
                   spec, y, folds, data_version="blend_v1", probability_scale=True,
                   training_time_sec=0.0, parent_run_id="", kaggle_run=False,
                   submit=False, wait=True) -> str:
    """Persist a blend as a runs.csv entry + artifact dir, optionally submitting it.

    Parameters
    ----------
    oof_proba, test_proba_mean : the blend's full-length OOF (CANON_N) and test
        (N_TEST) prediction vectors.
    members : list of {"label", "run_id", "weight"} dicts — the blend's provenance,
        stored in params.json/metrics.json (the flat schema has no multi-parent slot).
    spec : the blend definition (scheme + members + hyperparams); hashed into
        params_hash and written verbatim to params.json.
    probability_scale : False for rank-scale blends (rank-mean) so accuracy@0.5 is
        logged as NaN instead of a meaningless number.
    submit : when True, upload test_proba_mean to Kaggle and back-fill lb_public/
        lb_private (mirrors save_experiment(submit=True)).

    Returns the new run_id.
    """
    oof_proba = np.asarray(oof_proba, dtype="float64")
    test_proba_mean = np.asarray(test_proba_mean, dtype="float64")
    if len(oof_proba) != len(y):
        raise ValueError(f"oof_proba has {len(oof_proba)} rows, expected {len(y)}")
    if len(test_proba_mean) != N_TEST:
        raise ValueError(f"test_proba_mean has {len(test_proba_mean)} rows, expected {N_TEST}")

    run_id = new_run_id()
    run_dict, metrics = _build_run_dict(
        run_id=run_id, tag=tag, notes=notes, model_class=model_class, spec=spec,
        members=members, n_features=len(members), oof_proba=oof_proba,
        test_proba_mean=test_proba_mean, y=y, folds=folds, data_version=data_version,
        probability_scale=probability_scale, training_time_sec=training_time_sec,
        parent_run_id=parent_run_id, kaggle_run=kaggle_run)

    env, git = environment_info(), git_info()
    artifacts = {
        "params":             spec,
        "metrics":            metrics,
        "oof_proba":          oof_proba,
        # A blend has no per-fold test bagging; store the single test vector as a
        # (n_test, 1) column. test_proba_mean is authoritative (what make_submission reads).
        "test_proba_folds":   test_proba_mean.reshape(-1, 1),
        "test_proba_mean":    test_proba_mean,
        "feature_importance": None,
        "environment_text":   env["uv_lock_contents"],
        "git_diff":           git["diff"],
        "notes":              notes,
        "models":             None,
    }
    save_run(run_dict, artifacts, run_id)
    print(f"[{run_id}] {tag}: OOF ROC-AUC {run_dict['oof_roc_auc']:.6f} "
          f"(fold {run_dict['fold_roc_auc_mean']:.6f} ± {run_dict['fold_roc_auc_std']:.6f})")

    if submit:
        from src.kaggle_io import submit_run
        submit_run(run_id, wait=wait)
    return run_id


# ---------------------------------------------------------------------------
# The ~6 headline blends (option 2): build + log them in one place
# ---------------------------------------------------------------------------

def build_headline_blends(oof, test, meta, curated, full, y, folds, *,
                          softmax_T: float = 0.001, curated_stack_C: float = 1.0,
                          best_stack_C: float = 0.1, lgbm_stack_params: dict | None = None):
    """Construct the six headline blends as ready-to-log spec dicts.

    Pre-registered robust combiners (rank-mean curated, moderate-C curated logit
    stack, hill-climb curated) plus the carried-forward best logit stack (full pool)
    and two extras (best-softmax mean, Optuna LGBM stack). The "best" full-pool stack
    is pinned to (full, C=best_stack_C) — the notebook sweep's winner — rather than
    re-run as an argmax.
    """
    def members_of(cols, weights=None):
        rid = dict(zip(meta["label"], meta["run_id"]))
        w = weights if weights is not None else [1.0 / len(cols)] * len(cols)
        return [{"label": c, "run_id": rid[c], "weight": float(wi)} for c, wi in zip(cols, w)]

    blends = []

    # 1. rank-mean (curated) — rank scale, accuracy logged as NaN
    oof_p, test_p = rank_mean(oof, test, curated)
    blends.append(dict(
        tag="blend-rank-mean-curated", model_class="RankMeanBlend",
        oof_proba=oof_p, test_proba_mean=test_p, probability_scale=False,
        members=members_of(curated),
        spec={"scheme": "rank_mean", "feature_scale": "rank", "pool": "curated",
              "members": [c for c in curated]},
        notes=("Rank-mean blend of the curated pool (one best-AUC run per engine; "
               "linear slot by diversity). Calibration-free average for AUC. Post-hoc "
               "blend of logged OOF; per-fold AUC over the canonical seed-42 folds.")))

    # 2. moderate-C curated logit-logistic stack (pre-registered robust default)
    oof_p, test_p, coef = logit_stack(oof, test, curated, y, folds, curated_stack_C)
    blends.append(dict(
        tag="stack-logit-lr-curated-c1", model_class="LogitStackLR",
        oof_proba=oof_p, test_proba_mean=test_p, probability_scale=True,
        members=members_of(curated, coef),
        spec={"scheme": "logit_stack_lr", "feature_scale": "logit", "pool": "curated",
              "C": curated_stack_C, "members": [c for c in curated], "coef": coef},
        notes=(f"L2 logistic stack (C={curated_stack_C:g}) on standardized logit-OOF "
               "of the curated pool; OOF CV'd over the seed-42 folds, test refit on all "
               "OOF. Pre-registered robust combiner (technical_review.md §2.1).")))

    # 3. best logit stack — full pool, pinned to the notebook sweep winner
    oof_p, test_p, coef = logit_stack(oof, test, full, y, folds, best_stack_C)
    blends.append(dict(
        tag="stack-logit-lr-best", model_class="LogitStackLR",
        oof_proba=oof_p, test_proba_mean=test_p, probability_scale=True,
        members=members_of(full, coef),
        spec={"scheme": "logit_stack_lr", "feature_scale": "logit", "pool": "full",
              "C": best_stack_C, "members": [c for c in full], "coef": coef},
        notes=(f"L2 logistic stack (C={best_stack_C:g}) on standardized logit-OOF of all "
               f"{len(full)} aligned runs — the (pool, C) cell carried to submission in "
               "blending.ipynb. Strong L2 tames the multicollinear full pool.")))

    # 4. hill-climb (curated)
    oof_p, test_p, w = hill_climb(oof, test, curated, y)
    blends.append(dict(
        tag="blend-hillclimb-curated", model_class="HillClimbBlend",
        oof_proba=oof_p, test_proba_mean=test_p, probability_scale=True,
        members=members_of(curated, w),
        spec={"scheme": "bagged_hill_climb", "feature_scale": "prob", "pool": "curated",
              "members": [c for c in curated], "weights": w.tolist()},
        notes=("Bagged Caruana ensemble selection (25 bags, frac 0.5, seed 42) on the "
               "curated pool's OOF probabilities. Convex weights = selection frequency.")))

    # 5. best-softmax mean (curated)
    oof_p, test_p, w = softmax_auc(oof, test, curated, y, softmax_T)
    blends.append(dict(
        tag="blend-softmax-curated", model_class="SoftmaxAUCBlend",
        oof_proba=oof_p, test_proba_mean=test_p, probability_scale=True,
        members=members_of(curated, w),
        spec={"scheme": "softmax_auc", "feature_scale": "prob", "pool": "curated",
              "T": softmax_T, "members": [c for c in curated], "weights": w.tolist()},
        notes=(f"Strength-weighted mean of the curated pool, weights = softmax(OOF AUC / "
               f"T={softmax_T:g}). Concentrates mass on the strongest members.")))

    # 6. Optuna LGBM stack (curated)
    oof_p, test_p, params = lgbm_stack(oof, test, curated, y, folds, params=lgbm_stack_params)
    blends.append(dict(
        tag="stack-lgbm-optuna-curated", model_class="LGBMStack",
        oof_proba=oof_p, test_proba_mean=test_p, probability_scale=True,
        members=members_of(curated),
        spec={"scheme": "lgbm_stack", "feature_scale": "prob", "pool": "curated",
              "members": [c for c in curated], "lgbm_params": params},
        notes=("LightGBM stack on the curated pool's OOF, 50-trial shallow/regularised "
               "Optuna (seed 42), CV'd over the seed-42 folds; test refit on all OOF.")))

    return blends


def summarize_blends(blends: list[dict], y: np.ndarray, folds: np.ndarray,
                     best_single: float, base_fold_std: float) -> pd.DataFrame:
    """Per-fold AUC mean ± std for each blend vs the best single member (§2.2).

    Display-only (computes no side effects) — the notebook calls this to show the
    significance table; the actual runs.csv logging is done by log_headline_blends /
    scripts/run_blend_logging.py so re-running the notebook never double-logs.
    """
    rows = []
    for b in blends:
        oof_auc = float(roc_auc_score(y, b["oof_proba"]))
        _, fmean, fstd = fold_aucs(b["oof_proba"], y, folds)
        gain = oof_auc - best_single
        rows.append({"tag": b["tag"], "model_class": b["model_class"],
                     "n_members": len(b["members"]), "oof_roc_auc": oof_auc,
                     "fold_mean": fmean, "fold_std": fstd, "vs_best_single": gain,
                     "gain_in_fold_stds": gain / base_fold_std})
    return pd.DataFrame(rows).sort_values("oof_roc_auc", ascending=False).reset_index(drop=True)


def log_headline_blends(submit: bool = False, wait: bool = True,
                        lgbm_stack_params: dict | None = None) -> pd.DataFrame:
    """Build, log (and optionally submit) the six headline blends; return a summary.

    The returned frame reports each blend's pooled OOF AUC and per-fold mean ± std
    next to the best single member and the per-fold-noise comparison — the §2.2
    "does the gain clear CV noise?" table.
    """
    y, folds = load_target_and_folds()
    oof, test, meta, _ = load_aligned_runs(y)
    curated = curate_pool(meta)
    full = list(oof.columns)
    best_label = meta.loc[meta["oof_roc_auc"].idxmax(), "label"]
    best_single = float(meta["oof_roc_auc"].max())
    # Noise floor for the significance comparison: the best single member's own
    # per-fold AUC std (~0.0009). A blend gain smaller than this is inside CV noise.
    _, _, base_fold_std = fold_aucs(oof[best_label].to_numpy(), y, folds)

    print(f"Curated pool: {len(curated)} runs (from {len(full)} aligned). "
          f"Best single OOF AUC = {best_single:.6f} ({best_label}, "
          f"fold std {base_fold_std:.6f})\n")

    blends = build_headline_blends(oof, test, meta, curated, full, y, folds,
                                   lgbm_stack_params=lgbm_stack_params)
    rows = []
    for b in blends:
        t0 = time.perf_counter()
        # Always log first; submit separately so a Kaggle daily-cap/auth error on one
        # blend cannot abort the others (the run stays logged for later backfill).
        run_id = save_blend_run(
            tag=b["tag"], notes=b["notes"], model_class=b["model_class"],
            oof_proba=b["oof_proba"], test_proba_mean=b["test_proba_mean"],
            members=b["members"], spec=b["spec"], y=y, folds=folds,
            probability_scale=b["probability_scale"],
            training_time_sec=time.perf_counter() - t0, submit=False)
        if submit:
            try:
                from src.kaggle_io import submit_run
                submit_run(run_id, wait=wait)
            except Exception as e:
                print(f"  submit failed for {b['tag']} ({run_id}): "
                      f"{type(e).__name__}: {e}\n  -> logged locally; backfill later.")
        oof_auc = float(roc_auc_score(y, b["oof_proba"]))
        _, fmean, fstd = fold_aucs(b["oof_proba"], y, folds)
        gain = oof_auc - best_single
        rows.append({"tag": b["tag"], "run_id": run_id, "oof_roc_auc": oof_auc,
                     "fold_mean": fmean, "fold_std": fstd, "vs_best_single": gain,
                     "gain_in_fold_stds": gain / base_fold_std})

    summary = pd.DataFrame(rows)
    print(f"\nNoise floor (best single member's fold std): {base_fold_std:.6f}. "
          "A blend whose |gain_in_fold_stds| < 1 is inside CV noise.")
    return summary
