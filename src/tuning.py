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
