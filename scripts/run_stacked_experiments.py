"""Execute the three stacked-feature + pseudo-label GBDT runs from Experiments.ipynb.

Mirrors the `## Stacked OOF meta-features + true pseudo-labels` notebook section,
as a script so it can run unattended (and so re-running the whole notebook's slow
EBM study isn't needed). For each of LightGBM, XGBoost, CatBoost it: tunes a
regularization-biased Optuna study (on a subsample), fits a 5-fold outer run on
the full data with confident test rows pseudo-labelled into the training folds,
then saves the run and submits it to Kaggle.

Run: .venv\\Scripts\\python.exe scripts\\run_stacked_experiments.py
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import optuna
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.cv import run_cv_experiment, save_experiment
from src.stacking import build_stacked_dataset
from src.tracking import RUNS_DIR

optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_VERSION_STACK = "fe_v5_stack"
STACK_OOF_COLUMNS = {
    "oof_tabpfn":  "20260612-063413-30f430",
    "oof_tabicl":  "20260612-031158-5adf7a",
    "oof_realmlp": "20260612-185205-91e530",
    "oof_tabm":    "20260612-181438-3a3dd2",
    "oof_rf":      "20260612-133154-c3656f",
    "oof_lr":      "20260610-030317-343e5b",
}
PSEUDO_BLEND_RUNS = [
    "20260610-183508-48286b", "20260610-200857-74b63f", "20260611-051031-5d6bc5",
    "20260612-202507-e65b68", "20260612-133154-c3656f", "20260612-063413-30f430",
    "20260612-031158-5adf7a", "20260612-185205-91e530", "20260612-181438-3a3dd2",
    "20260610-030317-343e5b",
]
PSEUDO_LO, PSEUDO_HI = 0.05, 0.95

INNER_CV = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
OUTER_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Per-model trial budgets / tuning subsample (CatBoost is ~60 s/fit on CPU).
PLAN = {
    "lgbm":     dict(n_trials=40, tune_subsample=150_000, tag="lgbm-stack-pl-fe_v5",
                     parent="20260610-183508-48286b"),
    "xgb":      dict(n_trials=40, tune_subsample=150_000, tag="xgb-stack-pl-fe_v5",
                     parent="20260610-200857-74b63f"),
    "catboost": dict(n_trials=20, tune_subsample=100_000, tag="catboost-stack-pl-fe_v5",
                     parent="20260611-051031-5d6bc5"),
}


def make_model(name, params, cat_features):
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**params)
    if name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(cat_features=cat_features, **params)
    if name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(enable_categorical=True, tree_method="hist", **params)
    raise ValueError(name)


def space(name, trial):
    if name == "lgbm":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 800),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 7),
            num_leaves=trial.suggest_int("num_leaves", 8, 64),
            min_child_samples=trial.suggest_int("min_child_samples", 50, 400),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 0.9),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            subsample_freq=1, verbose=-1, random_state=42, n_jobs=-1,
        )
    if name == "catboost":
        return dict(
            iterations=trial.suggest_int("iterations", 200, 800),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            depth=trial.suggest_int("depth", 3, 7),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
            min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 20, 200),
            random_strength=trial.suggest_float("random_strength", 0.1, 10.0, log=True),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
            verbose=0, random_state=42,
        )
    if name == "xgb":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 800),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 7),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 30),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 0.9),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            verbosity=0, eval_metric="auc", random_state=42, n_jobs=-1,
        )
    raise ValueError(name)


FIXED = {"subsample_freq", "verbose", "random_state", "n_jobs", "eval_metric"}


def tune(name, X, y, cat_features, n_trials, tune_subsample):
    Xt, yt = X, y
    if tune_subsample and tune_subsample < len(X):
        Xt, _, yt, _ = train_test_split(X, y, train_size=tune_subsample,
                                        stratify=y, random_state=42)

    def objective(trial):
        params = space(name, trial)
        # Manual CV loop (fresh model per fold) — avoids sklearn clone(), which
        # rejects CatBoostClassifier when cat_features is set in the constructor.
        scores = []
        for tr_i, va_i in INNER_CV.split(Xt, yt):
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
    return {k: v for k, v in study.best_params.items() if k not in FIXED}


def main() -> None:
    train_df, test_df = build_stacked_dataset(
        base_version="fe_v4_native", oof_columns=STACK_OOF_COLUMNS,
        out_version=DATA_VERSION_STACK)
    feats = [c for c in train_df.columns if c not in ("id", "Churn")]
    X, y, X_test = train_df[feats], train_df["Churn"], test_df[feats]
    cat_features = [c for c in feats if isinstance(X[c].dtype, pd.CategoricalDtype)]

    # Pseudo-labels: confident test rows from a rank-mean blend of the curated cohort.
    tp = np.column_stack([np.load(RUNS_DIR / r / "test_proba_mean.npy")
                          for r in PSEUDO_BLEND_RUNS])
    blend = np.column_stack([rankdata(tp[:, j]) for j in range(tp.shape[1])]).mean(axis=1)
    blend = (blend - blend.min()) / (blend.max() - blend.min())
    conf = (blend < PSEUDO_LO) | (blend > PSEUDO_HI)
    X_pseudo = X_test[conf].reset_index(drop=True)
    y_pseudo = (blend[conf] > 0.5).astype(int)
    print(f"Pseudo-labels: {int(conf.sum()):,} confident test rows "
          f"({y_pseudo.mean():.3f} positive)\n", flush=True)

    fixed_extra = {
        "lgbm":     dict(subsample_freq=1, verbose=-1, random_state=42, n_jobs=-1),
        "xgb":      dict(verbosity=0, eval_metric="auc", random_state=42, n_jobs=-1),
        "catboost": dict(verbose=0, random_state=42),
    }

    # Optional CLI filter: `python run_stacked_experiments.py catboost xgb`
    only = set(sys.argv[1:])
    for name, cfg in PLAN.items():
        if only and name not in only:
            continue
        print(f"=== {cfg['tag']} ===", flush=True)
        try:
            best = tune(name, X, y, cat_features, cfg["n_trials"], cfg["tune_subsample"])
            params = {**best, **fixed_extra[name]}
            run_config = {
                "model_factory": (lambda p, _n=name: make_model(_n, p, cat_features)),
                "params": params, "cv": OUTER_CV, "tag": cfg["tag"],
                "notes": f"{name} on {DATA_VERSION_STACK} (fe_v4_native + 6 diverse OOF "
                         f"columns) with true pseudo-labels ({int(conf.sum())} confident "
                         f"test rows, top/bottom 5% of a rank-mean blend, into training folds "
                         f"only). Regularization-biased Optuna ({cfg['n_trials']} trials, "
                         f"3-fold inner CV on a {cfg['tune_subsample']//1000}k subsample, "
                         f"ROC-AUC). Run via scripts/run_stacked_experiments.py.",
                "parent_run_id": cfg["parent"], "save_models": False,
                "data_version": DATA_VERSION_STACK,
            }
            result = run_cv_experiment(run_config, X, y, X_test, feats,
                                       X_pseudo=X_pseudo, y_pseudo=y_pseudo)
            save_experiment(result, submit=True)
            print(f"  saved + submitted: OOF ROC-AUC {result['oof_roc_auc']:.6f}\n", flush=True)
        except Exception as e:  # one model failing shouldn't abort the others
            print(f"  ERROR on {name}: {type(e).__name__}: {e}\n", flush=True)


if __name__ == "__main__":
    main()
