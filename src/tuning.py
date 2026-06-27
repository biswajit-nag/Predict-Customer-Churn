"""Optuna hyperparameter tuning objectives for LightGBM, XGBoost, and CatBoost.

Each public function is a *factory*: it accepts the training data and an
optional CV splitter, and returns the callable that Optuna optimises.

Usage
-----
    from src.tuning import lgbm_objective, xgb_objective, catboost_objective

    study = optuna.create_study(direction='maximize')
    study.optimize(lgbm_objective(X, y), n_trials=50, show_progress_bar=True)

    print(study.best_value, study.best_params)

Design notes
------------
- The inner CV uses 3 folds (not 5) to keep the tuning budget manageable:
  50 trials × 3 folds = 150 fits per study. Final evaluation in
  Experiments.ipynb uses 5 folds.
- All three objectives use the same random_state (42) for the inner CV so
  that score differences between trials reflect hyperparameter effects, not
  fold-sampling noise.
- Fixed inference-time flags (verbosity=0, subsample_freq=1, etc.) are
  baked into the params dict inside the objective so they are never surfaced
  to Optuna as tunable knobs, and are never written to the saved JSON either
  (the save cell in Baselines.ipynb adds them back at load time).
"""

from sklearn.model_selection import StratifiedKFold, cross_val_score


def _default_cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=3, shuffle=True, random_state=42)


def lgbm_objective(X, y, cv=None):
    """Return an Optuna objective for LGBMClassifier tuning.

    Parameters
    ----------
    X, y : training features and target.
    cv   : CV splitter; defaults to StratifiedKFold(n_splits=3, random_state=42).

    Returns
    -------
    objective : callable(trial) -> float
        Mean 3-fold ROC AUC over the suggested hyperparameters.
    """
    from lightgbm import LGBMClassifier

    _cv = cv or _default_cv()

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 20, 300),
            "max_depth":         trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq":    1,  # must be >0 for subsample to apply; not tunable
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_split_gain":    trial.suggest_float("min_split_gain", 0.0, 1.0),
            "verbose":           -1,  # suppress per-tree output; not tunable
        }
        scores = cross_val_score(LGBMClassifier(**params), X, y, cv=_cv, scoring="roc_auc")
        return scores.mean()

    return objective


def xgb_objective(X, y, cv=None):
    """Return an Optuna objective for XGBClassifier tuning.

    Parameters
    ----------
    X, y : training features and target.
    cv   : CV splitter; defaults to StratifiedKFold(n_splits=3, random_state=42).

    Returns
    -------
    objective : callable(trial) -> float
        Mean 3-fold ROC AUC over the suggested hyperparameters.
    """
    from xgboost import XGBClassifier

    _cv = cv or _default_cv()

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma":            trial.suggest_float("gamma", 0.0, 1.0),
            "verbosity":        0,       # not tunable
            "eval_metric":      "logloss",  # not tunable
        }
        scores = cross_val_score(XGBClassifier(**params), X, y, cv=_cv, scoring="roc_auc")
        return scores.mean()

    return objective


def catboost_objective(X, y, cv=None):
    """Return an Optuna objective for CatBoostClassifier tuning.

    Parameters
    ----------
    X, y : training features and target.
    cv   : CV splitter; defaults to StratifiedKFold(n_splits=3, random_state=42).

    Returns
    -------
    objective : callable(trial) -> float
        Mean 3-fold ROC AUC over the suggested hyperparameters.
    """
    from catboost import CatBoostClassifier

    _cv = cv or _default_cv()

    def objective(trial):
        params = {
            "iterations":          trial.suggest_int("iterations", 100, 1000),
            "learning_rate":       trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth":               trial.suggest_int("depth", 3, 10),
            "l2_leaf_reg":         trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            "min_data_in_leaf":    trial.suggest_int("min_data_in_leaf", 1, 100),
            "random_strength":     trial.suggest_float("random_strength", 0.1, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "verbose":             0,  # not tunable
        }
        scores = cross_val_score(CatBoostClassifier(**params), X, y, cv=_cv, scoring="roc_auc")
        return scores.mean()

    return objective


# ===========================================================================
# Upgraded refinement API (early stopping + multivariate TPE + pruning + SQLite)
# ===========================================================================
#
# The three factories above are the ORIGINAL broad-search objectives
# (cross_val_score, n_estimators tuned, no pruning). They are kept verbatim for
# backward compatibility with Baselines.ipynb and
# scripts/run_stacked_experiments.py - do not change their signatures.
#
# The functions below are the refinement workflow used by the
# kaggle/predict-customer-churn-optuna-* notebooks. Differences from the legacy
# objectives:
#   * Early stopping replaces tuning `n_estimators`/`iterations`: a large cap +
#     an in-fold validation set, and the chosen tree count is read back from the
#     fitted model (median across folds, bumped for the larger 5-fold outer fit).
#   * A manual fold loop reports per-fold scores via ``trial.report`` so a
#     ``MedianPruner`` can stop clearly-weak trials after fold 1-2 (big win for
#     slow EBM / CatBoost).
#   * ``build_study`` persists to SQLite (resumable across Kaggle sessions),
#     uses a multivariate TPE sampler, and warm-starts from known-good params.
#   * ``finalize_topk`` returns the top-k trials' full param dicts (with the
#     ES-derived tree count baked in) for an honest 5-fold outer re-eval - pick
#     the winner by OUTER OOF ROC-AUC, not the inner-CV argmax.
#
# Conventions preserved: 3-fold inner CV at seed 42; ROC-AUC objective; fixed
# inference flags (subsample_freq=1, verbose=-1, eval_metric, ...) are merged in
# by the caller's model_factory and never surfaced to Optuna.

ES_ROUNDS_DEFAULT = 150       # early-stopping patience on the in-fold validation set
MAX_ESTIMATORS = 5000         # ceiling; ES decides the actual tree count
ITER_MULT = 1.1               # bump ES tree count for the larger 5-fold outer fit

# --- Search spaces: "narrow" = exploit (centred on current best), "broad" = explore.
# Each value is (low, high); the log flag is fixed per parameter in the objective.
_LGBM_SPACE = {
    "narrow": dict(learning_rate=(0.02, 0.12), max_depth=(4, 8), num_leaves_lo=8,
                   num_leaves_cap=64, min_child_samples=(60, 250), subsample=(0.6, 1.0),
                   colsample_bytree=(0.4, 0.8), reg_alpha=(0.5, 10.0),
                   reg_lambda=(1e-6, 5.0), min_split_gain=(0.0, 1.0),
                   cat_smooth=(5.0, 40.0), cat_l2=(1.0, 20.0),
                   min_data_per_group=(50, 300), max_cat_threshold=(16, 128)),
    "broad":  dict(learning_rate=(0.01, 0.3), max_depth=(3, 12), num_leaves_lo=4,
                   num_leaves_cap=255, min_child_samples=(10, 200), subsample=(0.5, 1.0),
                   colsample_bytree=(0.5, 1.0), reg_alpha=(1e-8, 10.0),
                   reg_lambda=(1e-8, 10.0), min_split_gain=(0.0, 1.0),
                   cat_smooth=(1.0, 500.0), cat_l2=(0.1, 500.0),
                   min_data_per_group=(10, 1000), max_cat_threshold=(8, 256)),
}
_XGB_SPACE = {
    "narrow": dict(lossguide=True, learning_rate=(0.02, 0.1), max_depth=(5, 8),
                   max_leaves=(16, 64), min_child_weight=(1, 10), subsample=(0.6, 0.9),
                   colsample_bytree=(0.5, 0.8), reg_alpha=(0.1, 5.0),
                   reg_lambda=(1.0, 8.0), gamma=(0.0, 1.0)),
    "broad":  dict(lossguide=False, learning_rate=(0.01, 0.3), max_depth=(3, 10),
                   min_child_weight=(1, 20), subsample=(0.5, 1.0),
                   colsample_bytree=(0.5, 1.0), reg_alpha=(1e-8, 10.0),
                   reg_lambda=(1e-8, 10.0), gamma=(0.0, 1.0)),
}
_CAT_SPACE = {
    "narrow": dict(learning_rate=(0.02, 0.1), depth=(5, 8), l2_leaf_reg=(2.0, 10.0),
                   random_strength=(0.1, 5.0), bagging_temperature=(0.0, 1.0),
                   border_count=(128, 254)),
    "broad":  dict(learning_rate=(0.01, 0.3), depth=(4, 10), l2_leaf_reg=(1.0, 30.0),
                   random_strength=(0.1, 10.0), bagging_temperature=(0.0, 1.0),
                   border_count=(32, 255)),
}
_EBM_SPACE = {
    "narrow": dict(learning_rate=(0.02, 0.08), max_bins=(256, 512),
                   max_interaction_bins=(32, 64), interactions=(8, 20),
                   min_samples_leaf=(20, 50), max_leaves=(3, 5),
                   smoothing_rounds=(100, 500), outer_bags=(6, 12)),
    "broad":  dict(learning_rate=(0.005, 0.25), max_bins=(128, 512),
                   max_interaction_bins=(16, 64), interactions=(0, 15),
                   min_samples_leaf=(2, 50), max_leaves=(2, 5),
                   smoothing_rounds=(0, 1000), outer_bags=(4, 12)),
}

# Tree-count parameter name per model (used by finalize_topk to bake the ES count).
ITER_PARAM = {"lgbm": "n_estimators", "xgb": "n_estimators", "catboost": "iterations"}


def make_sampler(seed: int = 42):
    """Multivariate TPE sampler: models correlated params and behaves under
    parallel trials (constant_liar). Reproducible at the given seed."""
    import optuna
    return optuna.samplers.TPESampler(
        multivariate=True, group=True, seed=seed,
        n_startup_trials=20, constant_liar=True,
    )


def make_pruner():
    """MedianPruner over the per-fold steps: keeps the first trials unpruned
    (so warm-start + startup exploration complete), then prunes trials whose
    running-mean AUC trails the median at the same fold."""
    import optuna
    return optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1)


def build_study(study_name, *, storage_dir=None, warm_start=None, seed=42):
    """Create (or resume) a persisted, multivariate-TPE, MedianPruner study.

    Parameters
    ----------
    study_name  : unique name; also the SQLite filename stem.
    storage_dir : directory for ``<study_name>.db``; None -> in-memory (not resumable).
    warm_start  : a param dict, or list of dicts, to ``enqueue_trial`` (deduped).
                  Strip any tree-count keys (n_estimators/iterations) first - they
                  are ES-derived, not searched.
    """
    import optuna
    from pathlib import Path

    storage = None
    if storage_dir is not None:
        Path(storage_dir).mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{Path(storage_dir) / (study_name + '.db')}"

    study = optuna.create_study(
        study_name=study_name, direction="maximize",
        sampler=make_sampler(seed), pruner=make_pruner(),
        storage=storage, load_if_exists=True,
    )
    if warm_start:
        for params in (warm_start if isinstance(warm_start, list) else [warm_start]):
            try:
                study.enqueue_trial(params, skip_if_exists=True)
            except TypeError:        # very old optuna without skip_if_exists
                study.enqueue_trial(params)
    return study


def _report_and_maybe_prune(trial, fold_aucs, fold, prune):
    import optuna
    import numpy as np
    if prune:
        trial.report(float(np.mean(fold_aucs)), step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()


def lgbm_tuning_objective(X, y, *, inner_cv=None, search="narrow",
                          es_rounds=ES_ROUNDS_DEFAULT, max_estimators=MAX_ESTIMATORS,
                          device="cpu", n_gpus=1, model_n_jobs=1, prune=True):
    """ES + pruning Optuna objective for LightGBM. num_leaves is coupled to
    max_depth (<= 2**max_depth). Records per-fold best tree counts in the trial's
    ``best_n_trees`` user attr for finalize_topk."""
    import numpy as np
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    from sklearn.metrics import roc_auc_score

    cv = inner_cv or _default_cv()
    s = _LGBM_SPACE[search]

    def objective(trial):
        max_depth = trial.suggest_int("max_depth", *s["max_depth"])
        num_leaves = trial.suggest_int(
            "num_leaves", s["num_leaves_lo"], min(2 ** max_depth, s["num_leaves_cap"]))
        params = dict(
            n_estimators=max_estimators,
            learning_rate=trial.suggest_float("learning_rate", *s["learning_rate"], log=True),
            max_depth=max_depth, num_leaves=num_leaves,
            min_child_samples=trial.suggest_int("min_child_samples", *s["min_child_samples"]),
            subsample=trial.suggest_float("subsample", *s["subsample"]),
            colsample_bytree=trial.suggest_float("colsample_bytree", *s["colsample_bytree"]),
            reg_alpha=trial.suggest_float("reg_alpha", *s["reg_alpha"], log=True),
            reg_lambda=trial.suggest_float("reg_lambda", *s["reg_lambda"], log=True),
            min_split_gain=trial.suggest_float("min_split_gain", *s["min_split_gain"]),
            cat_smooth=trial.suggest_float("cat_smooth", *s["cat_smooth"], log=True),
            cat_l2=trial.suggest_float("cat_l2", *s["cat_l2"], log=True),
            min_data_per_group=trial.suggest_int("min_data_per_group", *s["min_data_per_group"]),
            max_cat_threshold=trial.suggest_int("max_cat_threshold", *s["max_cat_threshold"]),
            subsample_freq=1, verbose=-1, n_jobs=model_n_jobs,
        )
        if device == "gpu":
            params.update(device_type="gpu", gpu_platform_id=0,
                          gpu_device_id=trial.number % max(1, n_gpus))
        aucs, n_trees = [], []
        for fold, (tr, va) in enumerate(cv.split(X, y)):
            model = LGBMClassifier(**params)
            model.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])],
                      eval_metric="auc",
                      callbacks=[early_stopping(es_rounds, verbose=False), log_evaluation(0)])
            proba = model.predict_proba(X.iloc[va])[:, 1]
            aucs.append(roc_auc_score(y.iloc[va], proba))
            n_trees.append(model.best_iteration_ or max_estimators)
            _report_and_maybe_prune(trial, aucs, fold, prune)
        trial.set_user_attr("best_n_trees", n_trees)
        return float(np.mean(aucs))

    return objective


def xgb_tuning_objective(X, y, *, inner_cv=None, search="narrow",
                         es_rounds=ES_ROUNDS_DEFAULT, max_estimators=MAX_ESTIMATORS,
                         enable_categorical=True, model_n_jobs=1, prune=True):
    """ES + pruning Optuna objective for XGBoost. "narrow" uses lossguide growth
    with a tunable max_leaves (the coupled recipe of the best run); "broad" uses
    depthwise. Records per-fold best tree counts for finalize_topk."""
    import numpy as np
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score

    cv = inner_cv or _default_cv()
    s = _XGB_SPACE[search]

    def objective(trial):
        params = dict(
            n_estimators=max_estimators,
            learning_rate=trial.suggest_float("learning_rate", *s["learning_rate"], log=True),
            max_depth=trial.suggest_int("max_depth", *s["max_depth"]),
            min_child_weight=trial.suggest_int("min_child_weight", *s["min_child_weight"]),
            subsample=trial.suggest_float("subsample", *s["subsample"]),
            colsample_bytree=trial.suggest_float("colsample_bytree", *s["colsample_bytree"]),
            reg_alpha=trial.suggest_float("reg_alpha", *s["reg_alpha"], log=True),
            reg_lambda=trial.suggest_float("reg_lambda", *s["reg_lambda"], log=True),
            gamma=trial.suggest_float("gamma", *s["gamma"]),
            tree_method="hist", enable_categorical=enable_categorical,
            eval_metric="auc", early_stopping_rounds=es_rounds,
            verbosity=0, n_jobs=model_n_jobs,
        )
        if s.get("lossguide"):
            params["grow_policy"] = "lossguide"
            params["max_leaves"] = trial.suggest_int("max_leaves", *s["max_leaves"])
        aucs, n_trees = [], []
        for fold, (tr, va) in enumerate(cv.split(X, y)):
            model = XGBClassifier(**params)
            model.fit(X.iloc[tr], y.iloc[tr],
                      eval_set=[(X.iloc[va], y.iloc[va])], verbose=False)
            proba = model.predict_proba(X.iloc[va])[:, 1]
            aucs.append(roc_auc_score(y.iloc[va], proba))
            bi = getattr(model, "best_iteration", None)
            n_trees.append((bi + 1) if bi is not None else max_estimators)
            _report_and_maybe_prune(trial, aucs, fold, prune)
        trial.set_user_attr("best_n_trees", n_trees)
        return float(np.mean(aucs))

    return objective


def catboost_tuning_objective(X, y, *, cat_features, inner_cv=None, search="narrow",
                              es_rounds=ES_ROUNDS_DEFAULT, max_estimators=MAX_ESTIMATORS,
                              device="gpu", n_gpus=2, prune=True):
    """ES + pruning Optuna objective for CatBoost. Builds a fresh model per fold
    (no sklearn.clone, which trips on CatBoost's cat_features normalisation).
    GPU fits run sequentially across both T4s via devices='0:1'. Records per-fold
    best tree counts for finalize_topk."""
    import numpy as np
    from catboost import CatBoostClassifier
    from sklearn.metrics import roc_auc_score

    cv = inner_cv or _default_cv()
    s = _CAT_SPACE[search]
    gpu = dict(task_type="GPU", devices=("0:1" if (device == "gpu" and n_gpus >= 2)
                                          else "0")) if device == "gpu" else {}

    def objective(trial):
        params = dict(
            iterations=max_estimators,
            learning_rate=trial.suggest_float("learning_rate", *s["learning_rate"], log=True),
            depth=trial.suggest_int("depth", *s["depth"]),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", *s["l2_leaf_reg"], log=True),
            random_strength=trial.suggest_float("random_strength", *s["random_strength"], log=True),
            bagging_temperature=trial.suggest_float("bagging_temperature", *s["bagging_temperature"]),
            border_count=trial.suggest_int("border_count", *s["border_count"]),
            eval_metric="AUC", od_type="Iter", od_wait=es_rounds, use_best_model=True,
            verbose=0, random_seed=42, **gpu,
        )
        aucs, n_trees = [], []
        for fold, (tr, va) in enumerate(cv.split(X, y)):
            model = CatBoostClassifier(**params, cat_features=cat_features)
            model.fit(X.iloc[tr], y.iloc[tr],
                      eval_set=(X.iloc[va], y.iloc[va]))
            proba = model.predict_proba(X.iloc[va])[:, 1]
            aucs.append(roc_auc_score(y.iloc[va], proba))
            n_trees.append(model.get_best_iteration() + 1)
            _report_and_maybe_prune(trial, aucs, fold, prune)
        trial.set_user_attr("best_n_trees", n_trees)
        return float(np.mean(aucs))

    return objective


def ebm_tuning_objective(X, y, *, feature_types, inner_cv=None, search="narrow", prune=True):
    """Pruning Optuna objective for the Explainable Boosting Machine. No boosting
    early stopping (EBM has no single tree count); the outer bags already use all
    cores, so trials run one at a time. No best_n_trees attr (finalize_topk leaves
    the params untouched)."""
    import numpy as np
    from interpret.glassbox import ExplainableBoostingClassifier
    from sklearn.metrics import roc_auc_score

    cv = inner_cv or _default_cv()
    s = _EBM_SPACE[search]

    def objective(trial):
        params = dict(
            learning_rate=trial.suggest_float("learning_rate", *s["learning_rate"], log=True),
            max_bins=trial.suggest_int("max_bins", *s["max_bins"]),
            max_interaction_bins=trial.suggest_int("max_interaction_bins", *s["max_interaction_bins"]),
            interactions=trial.suggest_int("interactions", *s["interactions"]),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", *s["min_samples_leaf"]),
            max_leaves=trial.suggest_int("max_leaves", *s["max_leaves"]),
            smoothing_rounds=trial.suggest_int("smoothing_rounds", *s["smoothing_rounds"]),
            outer_bags=trial.suggest_int("outer_bags", *s["outer_bags"]),
            feature_types=feature_types, n_jobs=-1, random_state=42,
        )
        aucs = []
        for fold, (tr, va) in enumerate(cv.split(X, y)):
            model = ExplainableBoostingClassifier(**params)
            model.fit(X.iloc[tr], y.iloc[tr])
            proba = model.predict_proba(X.iloc[va])[:, 1]
            aucs.append(roc_auc_score(y.iloc[va], proba))
            _report_and_maybe_prune(trial, aucs, fold, prune)
        return float(np.mean(aucs))

    return objective


def finalize_topk(study, *, k=3, iter_param=None, iter_mult=ITER_MULT):
    """Return the top-k COMPLETE trials as full param dicts, best inner-CV first.

    If ``iter_param`` is given (e.g. 'n_estimators'/'iterations'), the ES-derived
    tree count (median of the trial's per-fold best_n_trees, x iter_mult for the
    larger 5-fold fit) is injected under that key. Feed each returned ``params``
    into a run_config and pick the winner by 5-fold OUTER OOF ROC-AUC.
    """
    import numpy as np
    import optuna

    complete = [t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    complete.sort(key=lambda t: t.value, reverse=True)
    out = []
    for t in complete[:k]:
        params = dict(t.params)
        n_trees = t.user_attrs.get("best_n_trees")
        if iter_param and n_trees:
            params[iter_param] = int(round(float(np.median(n_trees)) * iter_mult))
        out.append({"trial_number": t.number, "inner_roc_auc": t.value, "params": params})
    return out
