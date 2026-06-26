"""Stratified k-fold cross-validation loop for experiment tracking.

Public functions
----------------
run_cv_experiment()
    Runs a full CV experiment, collects OOF and test probabilities, aggregates
    feature importance, and gathers reproducibility metadata.  Prints fold
    scores and the final OOF metric but does **not** save anything to disk.
    Returns a result dict that can be inspected before committing to storage.

save_experiment(result)
    Persists a result dict returned by run_cv_experiment() to disk and appends
    a row to experiments/runs.csv.  Call this only after reviewing the printed
    scores and deciding the run is worth keeping.
"""

import datetime
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from sklearn.metrics import accuracy_score

from src.tracking import (
    PROJECT_ROOT,
    RUNS_DIR,
    environment_info,
    git_info,
    hash_file,
    new_run_id,
    params_hash,
    save_run,
    train_parquet_for,
)

# Primary metric for this competition is ROC-AUC (the Kaggle leaderboard metric).
# Everything ranks on it; accuracy@0.5 is kept only as a secondary diagnostic.
PRIMARY_METRIC_NAME = "roc_auc"


def run_cv_experiment(
    run_config: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    encoded_features: list[str],
    kaggle_run: bool = False,
    X_pseudo: pd.DataFrame | None = None,
    y_pseudo=None,
) -> dict:
    """Run a cross-validated experiment and collect everything needed to log it.

    The **primary metric is ROC-AUC** (the competition's leaderboard metric):
    fold and OOF ROC-AUC are what every run is ranked on. Accuracy at a 0.5
    threshold is still computed and stored, but only as a *secondary* diagnostic
    (`oof_accuracy` / `fold_acc_*`).

    Parameters
    ----------
    run_config : dict
        Required keys:
            model_factory  callable(params) -> unfitted model
            params         dict passed to model_factory
            cv             StratifiedKFold (or compatible splitter)
            tag            short experiment label
            notes          one-liner for the CSV notes column
            parent_run_id  str; empty string if none
            save_models    bool; whether to joblib-dump fold models
            data_version   str; FE recipe identifier (parquet filename suffix)
        Optional keys (secondary diagnostic metric; default accuracy@0.5):
            secondary_metric       callable(y_true, y_label) -> float
            secondary_metric_name  str label for printing / CSV column
        Backwards-compat: the old keys `metric` / `metric_name` are still
        accepted and treated as the secondary metric.
    X_train, y_train : full training features and target.
    X_test           : test features (no target column).
    encoded_features : ordered list of feature column names; used for
                       feature-importance DataFrame and metrics.json.
    kaggle_run : bool
        True when this experiment is executed on a Kaggle server (GPU/CPU
        notebook) rather than locally. Recorded in the `kaggle_run` column so
        on-platform runs (whose data is regenerated from the competition CSVs
        and whose GPU results are not bit-reproducible locally) are filterable.
        Set it from the calling notebook; defaults to False (local).
    X_pseudo, y_pseudo : optional pseudo-labelled rows (e.g. confident test rows
        with a strong blend's predicted label). When given, they are appended to
        **each fold's training set only** — never to the validation fold — so the
        OOF vector stays full-length (len(X_train)) and leakage-free, aligned with
        the canonical seed-42 folds. X_pseudo must share X_train's columns; it is
        re-cast to X_train's dtypes so categorical codes line up. The count is
        logged in the `n_pseudo` column.

    Returns
    -------
    result : dict
        Inspectable result with keys run_id, run_dict, artifacts, oof_roc_auc,
        oof_accuracy, fold_roc_aucs, fold_accuracies. Pass it to
        save_experiment() to persist the run.
    """
    # Secondary diagnostic metric — accuracy@0.5 by default; `metric`/`metric_name`
    # are honoured for backwards compatibility with older run_config cells.
    secondary_metric = (
        run_config.get("secondary_metric")
        or run_config.get("metric")
        or accuracy_score
    )
    secondary_name = (
        run_config.get("secondary_metric_name")
        or run_config.get("metric_name")
        or "accuracy"
    )
    # ------------------------------------------------------------------
    # 1. Initialise accumulators
    # ------------------------------------------------------------------
    run_id = new_run_id()
    print(f"Run ID: {run_id}")
    print(f"Tag:    {run_config['tag']}")
    print()

    cv               = run_config["cv"]
    oof_proba        = np.zeros(len(X_train))
    test_proba_folds = np.zeros((len(X_test), cv.n_splits))
    fold_roc_aucs, fold_times, fold_models = [], [], []   # ROC-AUC is primary
    fold_accuracies = []                                   # accuracy@0.5 is secondary
    gain_imps, split_imps = [], []  # LightGBM gain/split; empty for other model families

    # Pseudo-labelled rows (e.g. confident test rows) — appended to every fold's
    # *training* set only, so the OOF/val scoring stays on real rows. Align their
    # schema/dtypes to X_train so categorical codes match (test-only levels -> NaN).
    use_pseudo = X_pseudo is not None and len(X_pseudo) > 0
    if use_pseudo:
        X_pseudo = X_pseudo[X_train.columns].astype(X_train.dtypes.to_dict())
        y_pseudo = pd.Series(np.asarray(y_pseudo), name=y_train.name)
    n_pseudo = int(len(X_pseudo)) if use_pseudo else 0
    if use_pseudo:
        print(f"Pseudo-labels: +{n_pseudo} test rows into each fold's training set "
              f"(churn rate {float(y_pseudo.mean()):.4f})\n")

    # ------------------------------------------------------------------
    # 2. Fold loop — fit → predict val → predict test → record
    # ------------------------------------------------------------------
    # tr_idx / va_idx are positional indices; StratifiedKFold preserves the
    # Churn class ratio in every fold.
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_train, y_train)):
        model = run_config["model_factory"](run_config["params"])

        X_fit, y_fit = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        if use_pseudo:
            X_fit = pd.concat([X_fit, X_pseudo], ignore_index=True)
            y_fit = pd.concat([y_fit.reset_index(drop=True), y_pseudo], ignore_index=True)

        t0 = time.perf_counter()
        model.fit(X_fit, y_fit)
        elapsed = time.perf_counter() - t0

        # OOF: val predictions land in their original row positions so the
        # full vector is honest (no row seen during its own fold's training).
        va_proba = model.predict_proba(X_train.iloc[va_idx])[:, 1]
        oof_proba[va_idx] = va_proba

        # Test: keep per-fold columns so the bagged mean is just .mean(axis=1).
        test_proba_folds[:, fold] = model.predict_proba(X_test)[:, 1]

        # Primary: ROC-AUC straight from the probabilities (no threshold).
        fold_roc_auc = roc_auc_score(y_train.iloc[va_idx], va_proba)
        # Secondary diagnostic: threshold at 0.5 for label-based metrics.
        fold_acc = secondary_metric(
            y_train.iloc[va_idx], (va_proba >= 0.5).astype(int)
        )
        fold_roc_aucs.append(fold_roc_auc)
        fold_accuracies.append(fold_acc)
        fold_times.append(elapsed)
        fold_models.append(model)

        # LightGBM exposes .booster_; skip silently for other model types.
        if hasattr(model, "booster_"):
            gain_imps.append(model.booster_.feature_importance(importance_type="gain"))
            split_imps.append(model.booster_.feature_importance(importance_type="split"))

        print(f"Fold {fold}: roc_auc={fold_roc_auc:.4f}  {secondary_name}={fold_acc:.4f}  (fit {elapsed:.1f}s)")

    # ------------------------------------------------------------------
    # 3. Aggregate
    # ------------------------------------------------------------------
    # test_proba_mean: bagged across folds — smoother than any single fold.
    test_proba_mean = test_proba_folds.mean(axis=1)

    # oof_roc_auc (PRIMARY): computed on the full OOF vector — the competition
    # metric and what every run is ranked on (each training row predicted once).
    oof_roc_auc = roc_auc_score(y_train, oof_proba)

    # oof_accuracy (SECONDARY): accuracy@0.5 over the full OOF vector. Kept as a
    # diagnostic only; it is threshold/calibration-sensitive and does not drive
    # ranking. Tune the threshold on OOF if accuracy is ever reported for real.
    oof_accuracy = secondary_metric(y_train, (oof_proba >= 0.5).astype(int))

    # Mean gain/split across folds gives a more stable ranking than any single fold.
    feat_imp = None
    if gain_imps:
        feat_imp = pd.DataFrame({
            "feature": encoded_features,
            "gain":    np.mean(gain_imps,  axis=0),
            "split":   np.mean(split_imps, axis=0),
        })

    # ------------------------------------------------------------------
    # 4. Collect metadata
    # ------------------------------------------------------------------
    git          = git_info()
    env          = environment_info()
    sample_model = fold_models[0]   # all folds share hyperparams; fold 0 is representative
    full_params  = sample_model.get_params()

    # Columns are ordered primary-metric-first: ROC-AUC block, then the
    # secondary accuracy block, then the Kaggle leaderboard scores (filled by
    # submit_run / backfill once the run is submitted).
    run_dict = {
        "run_id":            run_id,
        "timestamp":         datetime.datetime.now().isoformat(timespec="seconds"),
        "tag":               run_config["tag"],
        "notes":             run_config["notes"],
        "status":            "success",
        "parent_run_id":     run_config["parent_run_id"],
        "git_hash":          git["hash"],
        "git_dirty":         git["dirty"],
        # data_hash fingerprints the parquet this run actually trained on,
        # resolved from its data_version (not the one-hot default).
        "data_hash":         hash_file(train_parquet_for(run_config["data_version"])),
        "data_version":      run_config["data_version"],
        "python_version":    env["python_version"],
        "platform":          env["platform"],
        "kaggle_run":        bool(kaggle_run),
        "model_class":       type(sample_model).__name__,
        "params_hash":       params_hash(full_params),
        "cv_type":           type(cv).__name__,
        "n_splits":          cv.n_splits,
        "random_state":      cv.random_state,
        "n_train":           len(X_train),
        "n_test":            len(X_test),
        "n_features":        X_train.shape[1],
        "n_pseudo":          n_pseudo,
        "metric":            PRIMARY_METRIC_NAME,
        # --- primary metric: ROC-AUC ---
        "oof_roc_auc":        float(oof_roc_auc),
        "fold_roc_auc_mean":  float(np.mean(fold_roc_aucs)),
        "fold_roc_auc_std":   float(np.std(fold_roc_aucs)),
        # --- secondary diagnostic: accuracy@0.5 ---
        "oof_accuracy":       float(oof_accuracy),
        "fold_acc_mean":      float(np.mean(fold_accuracies)),
        "fold_acc_std":       float(np.std(fold_accuracies)),
        # --- Kaggle leaderboard (filled post-submission) ---
        "lb_public":          np.nan,
        "lb_private":         np.nan,
        "training_time_sec":  float(sum(fold_times)),
        "artifact_dir":       str((RUNS_DIR / run_id).relative_to(PROJECT_ROOT)),
    }

    metrics = {
        # Primary metric first; fold_scores kept (== accuracy) for backwards
        # compatibility with scripts that read the per-run metrics.json.
        "oof_roc_auc":        float(oof_roc_auc),
        "fold_roc_aucs":      [float(r) for r in fold_roc_aucs],
        "fold_roc_auc_mean":  float(np.mean(fold_roc_aucs)),
        "fold_roc_auc_std":   float(np.std(fold_roc_aucs)),
        "oof_accuracy":       float(oof_accuracy),
        "fold_scores":        [float(s) for s in fold_accuracies],
        "fold_acc_mean":      float(np.mean(fold_accuracies)),
        "fold_acc_std":       float(np.std(fold_accuracies)),
        "fold_times_sec":     [float(t) for t in fold_times],
        "training_time_sec":  float(sum(fold_times)),
        "cv_config": {
            "type":         type(cv).__name__,
            "n_splits":     cv.n_splits,
            "random_state": cv.random_state,
            "shuffle":      getattr(cv, "shuffle", None),
        },
        "features": encoded_features,
    }

    artifacts = {
        "params":             full_params,
        "metrics":            metrics,
        "oof_proba":          oof_proba,
        "test_proba_folds":   test_proba_folds,
        "test_proba_mean":    test_proba_mean,
        "feature_importance": feat_imp,
        "environment_text":   env["uv_lock_contents"],
        "git_diff":           git["diff"],
        "notes":              run_config["notes"],
        "models":             fold_models if run_config["save_models"] else None,
    }

    # ------------------------------------------------------------------
    # 5. Return result — caller decides whether to save
    # ------------------------------------------------------------------
    print()
    print(f"OOF ROC-AUC:  {oof_roc_auc:.4f}   (primary)")
    print(f"OOF {secondary_name}: {oof_accuracy:.4f}   (secondary)")
    print(f"Folds ROC-AUC: {np.mean(fold_roc_aucs):.4f} ± {np.std(fold_roc_aucs):.4f}")
    print()
    print("Run complete. Call save_experiment(result) to log this run permanently.")

    return {
        "run_id":          run_id,
        "run_dict":        run_dict,
        "artifacts":       artifacts,
        "oof_roc_auc":     oof_roc_auc,
        "oof_accuracy":    oof_accuracy,
        "fold_roc_aucs":   fold_roc_aucs,
        "fold_accuracies": fold_accuracies,
    }


def save_experiment(result: dict, submit: bool = False, wait: bool = True) -> str:
    """Persist a completed run to disk and append a row to experiments/runs.csv.

    Parameters
    ----------
    result : dict
        The dict returned by run_cv_experiment().
    submit : bool
        If True, also submit this run's bagged test predictions
        (test_proba_mean) to Kaggle and write the returned public/private
        leaderboard scores back into the run's runs.csv row. Requires Kaggle
        API credentials (see src/kaggle_io.py and docs/kaggle_setup.md).
    wait : bool
        When submit=True, block until Kaggle finishes scoring so the LB columns
        are filled in this call. If False, fire the submission and return; run
        backfill_lb_scores.py later to collect the score.

    Returns
    -------
    run_id : str
        Identifier for the saved run.  Use it to locate the artifact directory
        at experiments/runs/{run_id}/ or to filter runs.csv.

    Why separate from run_cv_experiment
    ------------------------------------
    Keeping the save step explicit lets you review fold scores and decide
    whether a run is worth archiving before paying the I/O cost of writing
    fold models, OOF arrays, and environment snapshots.
    """
    save_run(result["run_dict"], result["artifacts"], result["run_id"])
    print(f"Saved to: {RUNS_DIR / result['run_id']}")

    if submit:
        # Imported lazily so the experiment framework has no hard dependency on
        # the Kaggle CLI / credentials unless a submission is actually requested.
        from src.kaggle_io import submit_run

        submit_run(result["run_id"], wait=wait)

    return result["run_id"]