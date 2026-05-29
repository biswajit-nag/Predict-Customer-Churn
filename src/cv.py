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

from src.tracking import (
    DATA_DIR,
    PROJECT_ROOT,
    RUNS_DIR,
    environment_info,
    git_info,
    hash_file,
    new_run_id,
    params_hash,
    save_run,
)


def run_cv_experiment(
    run_config: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    encoded_features: list[str],
) -> str:
    """Run a cross-validated experiment and log it to experiments/runs.csv.

    Parameters
    ----------
    run_config : dict
        Required keys:
            model_factory  callable(params) -> unfitted model
            params         dict passed to model_factory
            metric         callable(y_true, y_pred) -> float
            metric_name    str label for printing / CSV column
            cv             StratifiedKFold (or compatible splitter)
            tag            short experiment label
            notes          one-liner for the CSV notes column
            parent_run_id  str; empty string if none
            save_models    bool; whether to joblib-dump fold models
            data_version   str; FE recipe identifier (parquet filename suffix)
    X_train, y_train : full training features and target.
    X_test           : test features (no target column).
    encoded_features : ordered list of feature column names; used for
                       feature-importance DataFrame and metrics.json.

    Returns
    -------
    run_id : str
        Identifier for the logged run. Use it to locate the artifact
        directory at experiments/runs/{run_id}/ or filter runs.csv.
    """
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
    fold_scores, fold_times, fold_models = [], [], []
    fold_roc_aucs = []
    gain_imps, split_imps = [], []  # LightGBM gain/split; empty for other model families

    # ------------------------------------------------------------------
    # 2. Fold loop — fit → predict val → predict test → record
    # ------------------------------------------------------------------
    # tr_idx / va_idx are positional indices; StratifiedKFold preserves the
    # Churn class ratio in every fold.
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_train, y_train)):
        model = run_config["model_factory"](run_config["params"])

        t0 = time.perf_counter()
        model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        elapsed = time.perf_counter() - t0

        # OOF: val predictions land in their original row positions so the
        # full vector is honest (no row seen during its own fold's training).
        va_proba = model.predict_proba(X_train.iloc[va_idx])[:, 1]
        oof_proba[va_idx] = va_proba

        # Test: keep per-fold columns so the bagged mean is just .mean(axis=1).
        test_proba_folds[:, fold] = model.predict_proba(X_test)[:, 1]

        # Threshold at 0.5 for accuracy_score (expects labels, not probas).
        score = run_config["metric"](
            y_train.iloc[va_idx], (va_proba >= 0.5).astype(int)
        )
        fold_roc_auc = roc_auc_score(y_train.iloc[va_idx], va_proba)
        fold_scores.append(score)
        fold_roc_aucs.append(fold_roc_auc)
        fold_times.append(elapsed)
        fold_models.append(model)

        # LightGBM exposes .booster_; skip silently for other model types.
        if hasattr(model, "booster_"):
            gain_imps.append(model.booster_.feature_importance(importance_type="gain"))
            split_imps.append(model.booster_.feature_importance(importance_type="split"))

        print(f"Fold {fold}: {run_config['metric_name']}={score:.4f}  roc_auc={fold_roc_auc:.4f}  (fit {elapsed:.1f}s)")

    # ------------------------------------------------------------------
    # 3. Aggregate
    # ------------------------------------------------------------------
    # test_proba_mean: bagged across folds — smoother than any single fold.
    test_proba_mean = test_proba_folds.mean(axis=1)

    # oof_score: computed on the full vector, not the mean of fold_scores.
    # The two agree closely but oof_score is what Kaggle-style leaderboards
    # correlate with (every training row predicted exactly once).
    oof_score = run_config["metric"](y_train, (oof_proba >= 0.5).astype(int))

    # oof_roc_auc: computed directly from probabilities — no threshold needed.
    # Always logged regardless of the run_config metric, since it is a useful
    # secondary signal for ranking and calibration.
    oof_roc_auc = roc_auc_score(y_train, oof_proba)

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

    run_dict = {
        "run_id":            run_id,
        "timestamp":         datetime.datetime.now().isoformat(timespec="seconds"),
        "tag":               run_config["tag"],
        "notes":             run_config["notes"],
        "status":            "success",
        "parent_run_id":     run_config["parent_run_id"],
        "git_hash":          git["hash"],
        "git_dirty":         git["dirty"],
        "data_hash":         hash_file(DATA_DIR / "train_df.parquet"),
        "data_version":      run_config["data_version"],
        "python_version":    env["python_version"],
        "platform":          env["platform"],
        "model_class":       type(sample_model).__name__,
        "params_hash":       params_hash(full_params),
        "cv_type":           type(cv).__name__,
        "n_splits":          cv.n_splits,
        "random_state":      cv.random_state,
        "n_train":           len(X_train),
        "n_test":            len(X_test),
        "n_features":        X_train.shape[1],
        "metric":            run_config["metric_name"],
        "fold_scores_mean":   float(np.mean(fold_scores)),
        "fold_scores_std":    float(np.std(fold_scores)),
        "oof_score":          float(oof_score),
        "oof_roc_auc":        float(oof_roc_auc),
        "fold_roc_auc_mean":  float(np.mean(fold_roc_aucs)),
        "fold_roc_auc_std":   float(np.std(fold_roc_aucs)),
        "training_time_sec":  float(sum(fold_times)),
        "artifact_dir":       str((RUNS_DIR / run_id).relative_to(PROJECT_ROOT)),
    }

    metrics = {
        "fold_scores":        [float(s) for s in fold_scores],
        "fold_times_sec":     [float(t) for t in fold_times],
        "oof_score":          float(oof_score),
        "fold_scores_mean":   float(np.mean(fold_scores)),
        "fold_scores_std":    float(np.std(fold_scores)),
        "oof_roc_auc":        float(oof_roc_auc),
        "fold_roc_aucs":      [float(r) for r in fold_roc_aucs],
        "fold_roc_auc_mean":  float(np.mean(fold_roc_aucs)),
        "fold_roc_auc_std":   float(np.std(fold_roc_aucs)),
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
    print(f"OOF {run_config['metric_name']}: {oof_score:.4f}")
    print(f"OOF ROC-AUC:  {oof_roc_auc:.4f}")
    print(f"Folds:        {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    print()
    print("Run complete. Call save_experiment(result) to log this run permanently.")

    return {
        "run_id":        run_id,
        "run_dict":      run_dict,
        "artifacts":     artifacts,
        "oof_score":     oof_score,
        "oof_roc_auc":   oof_roc_auc,
        "fold_scores":   fold_scores,
        "fold_roc_aucs": fold_roc_aucs,
    }


def save_experiment(result: dict) -> str:
    """Persist a completed run to disk and append a row to experiments/runs.csv.

    Parameters
    ----------
    result : dict
        The dict returned by run_cv_experiment().

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
    return result["run_id"]