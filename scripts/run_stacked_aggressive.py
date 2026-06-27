"""Aggressive full-tuning test of the three stacked GBDTs on 100% pseudo-labelled test.

Experiment (exploratory): does an aggressive search — 60 trials (the budget the best
fe_v4_native runs used), an *unconstrained* space, and full-data tuning — on the
stacked features plus the ENTIRE test set as pseudo-labels close the gap to the best
single model? Prior evidence says the earlier ~0.001 deficit was the conservative /
subsampled tuning regime, not the techniques; this stress-tests that.

Differences vs scripts/run_stacked_experiments.py:
  - pseudo-labels = 100% of test rows (hard label = blend > 0.5), not the top/bottom 5%;
  - 60 trials, unconstrained (capacity-allowing) search space;
  - LGBM/XGB tune on the full 594k rows. CatBoost tunes on a 120k subsample because
    local CPU full-data CatBoost at 60 trials is ~5h (the original used a Kaggle GPU);
    its final 5-fold fit still uses all rows + all pseudo rows.

Run: .venv\\Scripts\\python.exe scripts\\run_stacked_aggressive.py [lgbm xgb catboost]
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

import numpy as np
import optuna
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.cv import run_cv_experiment, save_experiment
from src.stacking import build_stacked_dataset
from src.tracking import RUNS_DIR
from run_stacked_experiments import (  # reuse shared pieces (single source of truth)
    DATA_VERSION_STACK, INNER_CV, OUTER_CV, PSEUDO_BLEND_RUNS,
    STACK_OOF_COLUMNS, make_model,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

PLAN = {
    "lgbm":     dict(n_trials=60, tune_subsample=None,    tag="lgbm-stack-pl100-aggr-fe_v5",
                     parent="20260610-183508-48286b"),
    "xgb":      dict(n_trials=60, tune_subsample=None,    tag="xgb-stack-pl100-aggr-fe_v5",
                     parent="20260610-200857-74b63f"),
    "catboost": dict(n_trials=60, tune_subsample=120_000, tag="catboost-stack-pl100-aggr-fe_v5",
                     parent="20260611-051031-5d6bc5"),
}

FIXED_EXTRA = {
    "lgbm":     dict(subsample_freq=1, verbose=-1, random_state=42, n_jobs=-1),
    "xgb":      dict(verbosity=0, eval_metric="auc", random_state=42, n_jobs=-1),
    "catboost": dict(verbose=0, random_state=42),
}
_STRIP = {"subsample_freq", "verbose", "random_state", "n_jobs", "eval_metric"}


def aggr_space(name, trial):
    """Unconstrained, capacity-allowing search space (no regularization bias)."""
    if name == "lgbm":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 300, 1200),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            max_depth=trial.suggest_int("max_depth", 4, 12),
            num_leaves=trial.suggest_int("num_leaves", 16, 255),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 200),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            subsample_freq=1, verbose=-1, random_state=42, n_jobs=-1,
        )
    if name == "xgb":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 300, 1200),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            max_depth=trial.suggest_int("max_depth", 4, 12),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            gamma=trial.suggest_float("gamma", 0.0, 5.0),
            verbosity=0, eval_metric="auc", random_state=42, n_jobs=-1,
        )
    if name == "catboost":
        return dict(
            iterations=trial.suggest_int("iterations", 300, 1200),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            depth=trial.suggest_int("depth", 4, 10),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 1, 100),
            random_strength=trial.suggest_float("random_strength", 0.1, 10.0, log=True),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
            verbose=0, random_state=42,
        )
    raise ValueError(name)


def tune(name, X, y, cat_features, n_trials, tune_subsample):
    Xt, yt = X, y
    if tune_subsample and tune_subsample < len(X):
        Xt, _, yt, _ = train_test_split(X, y, train_size=tune_subsample,
                                        stratify=y, random_state=42)

    def objective(trial):
        params = aggr_space(name, trial)
        scores = []
        for tr_i, va_i in INNER_CV.split(Xt, yt):       # manual CV loop (no clone)
            model = make_model(name, params, cat_features)
            model.fit(Xt.iloc[tr_i], yt.iloc[tr_i])
            scores.append(roc_auc_score(
                yt.iloc[va_i], model.predict_proba(Xt.iloc[va_i])[:, 1]))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"  {name}: best inner-CV ROC-AUC {study.best_value:.6f} "
          f"(trial {study.best_trial.number}, tuned on {len(Xt):,} rows)", flush=True)
    return {k: v for k, v in study.best_params.items() if k not in _STRIP}


def main() -> None:
    train_df, test_df = build_stacked_dataset(
        base_version="fe_v4_native", oof_columns=STACK_OOF_COLUMNS,
        out_version=DATA_VERSION_STACK)
    feats = [c for c in train_df.columns if c not in ("id", "Churn")]
    X, y, X_test = train_df[feats], train_df["Churn"], test_df[feats]
    cat_features = [c for c in feats if isinstance(X[c].dtype, pd.CategoricalDtype)]

    # FULL pseudo-labels: every test row, hard label from the rank-mean blend.
    tp = np.column_stack([np.load(RUNS_DIR / r / "test_proba_mean.npy")
                          for r in PSEUDO_BLEND_RUNS])
    blend = np.column_stack([rankdata(tp[:, j]) for j in range(tp.shape[1])]).mean(axis=1)
    blend = (blend - blend.min()) / (blend.max() - blend.min())
    X_pseudo = X_test.reset_index(drop=True)             # 100% of test rows
    y_pseudo = (blend > 0.5).astype(int)
    print(f"Pseudo-labels: ALL {len(X_pseudo):,} test rows "
          f"({y_pseudo.mean():.3f} positive)\n", flush=True)

    only = set(sys.argv[1:])
    for name, cfg in PLAN.items():
        if only and name not in only:
            continue
        print(f"=== {cfg['tag']} ===", flush=True)
        try:
            best = tune(name, X, y, cat_features, cfg["n_trials"], cfg["tune_subsample"])
            params = {**best, **FIXED_EXTRA[name]}
            tuned_on = "full 594k" if cfg["tune_subsample"] is None \
                else f"{cfg['tune_subsample'] // 1000}k subsample"
            run_config = {
                "model_factory": (lambda p, _n=name: make_model(_n, p, cat_features)),
                "params": params, "cv": OUTER_CV, "tag": cfg["tag"],
                "notes": f"AGGRESSIVE test: {name} on {DATA_VERSION_STACK} (fe_v4_native + 6 "
                         f"diverse OOF columns) with 100% of test rows pseudo-labelled "
                         f"(hard label = rank-mean blend > 0.5) into training folds only. "
                         f"Unconstrained Optuna search, {cfg['n_trials']} trials, 3-fold inner "
                         f"CV tuned on {tuned_on}. Tests whether aggressive full tuning closes "
                         f"the gap to the fe_v4 base. scripts/run_stacked_aggressive.py.",
                "parent_run_id": cfg["parent"], "save_models": False,
                "data_version": DATA_VERSION_STACK,
            }
            result = run_cv_experiment(run_config, X, y, X_test, feats,
                                       X_pseudo=X_pseudo, y_pseudo=y_pseudo)
            save_experiment(result, submit=True)
            print(f"  saved + submitted: OOF ROC-AUC {result['oof_roc_auc']:.6f}\n", flush=True)
        except Exception as e:
            print(f"  ERROR on {name}: {type(e).__name__}: {e}\n", flush=True)


if __name__ == "__main__":
    main()
