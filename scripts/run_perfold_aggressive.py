"""Aggressive 60-trial tuning of LGBM & XGB on the per-fold OOF columns (fe_v6_perfold).

No pseudo-labels and no collapsed OOF columns. Each of the 6 stacked models contributes
5 per-fold columns instead of one: on train, `{model}_f{k}` is that model's held-out OOF
prediction for fold-k rows only (NaN for the other 4/5); on test, it's that fold-model's
test prediction (all 5 filled). Each column is therefore the same fitted model's
*out-of-sample* output on both its train rows and all test rows, removing the
single-fold-OOF (train) vs bagged-mean (test) distribution mismatch a collapsed column
suffers. GBDTs consume the 80%-NaN columns natively.

Aggressive search: 60 trials (the budget the best fe_v4 runs used), unconstrained space,
full-data tuning. LGBM/XGB only (fast on CPU); CatBoost is skipped.

Run: .venv\\Scripts\\python.exe scripts\\run_perfold_aggressive.py [lgbm xgb]
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

import pandas as pd

from src.cv import run_cv_experiment, save_experiment
from src.stacking import build_perfold_dataset
# Reuse the aggressive tuner / model builder (single source of truth).
from run_stacked_aggressive import FIXED_EXTRA, OUTER_CV, make_model, tune

DATA_VERSION = "fe_v6_perfold"
FOLD_RUNS = {
    "tabpfn":  "20260612-063413-30f430",
    "tabicl":  "20260612-031158-5adf7a",
    "realmlp": "20260612-185205-91e530",
    "tabm":    "20260612-181438-3a3dd2",
    "rf":      "20260612-133154-c3656f",
    "lr":      "20260610-030317-343e5b",
}
PLAN = {
    "lgbm": dict(n_trials=60, tune_subsample=None, tag="lgbm-perfold-aggr-fe_v6",
                 parent="20260610-183508-48286b"),
    "xgb":  dict(n_trials=60, tune_subsample=None, tag="xgb-perfold-aggr-fe_v6",
                 parent="20260610-200857-74b63f"),
}


def main() -> None:
    train_df, test_df = build_perfold_dataset("fe_v4_native", FOLD_RUNS, DATA_VERSION)
    feats = [c for c in train_df.columns if c not in ("id", "Churn")]
    X, y, X_test = train_df[feats], train_df["Churn"], test_df[feats]
    cat_features = [c for c in feats if isinstance(X[c].dtype, pd.CategoricalDtype)]
    n_pf = sum("_f" in c and c.rsplit("_f", 1)[-1].isdigit() for c in feats)
    print(f"fe_v6_perfold: {len(feats)} features "
          f"({len(cat_features)} categorical + {n_pf} per-fold OOF). No pseudo-labels.\n",
          flush=True)

    only = set(sys.argv[1:])
    for name, cfg in PLAN.items():
        if only and name not in only:
            continue
        print(f"=== {cfg['tag']} ===", flush=True)
        try:
            best = tune(name, X, y, cat_features, cfg["n_trials"], cfg["tune_subsample"])
            params = {**best, **FIXED_EXTRA[name]}
            run_config = {
                "model_factory": (lambda p, _n=name: make_model(_n, p, cat_features)),
                "params": params, "cv": OUTER_CV, "tag": cfg["tag"],
                "notes": f"AGGRESSIVE per-fold test: {name} on {DATA_VERSION} (fe_v4_native + "
                         f"{n_pf} per-fold OOF columns = 6 models x 5 seed-42 folds; train cols "
                         f"~20% filled with held-out OOF, NaN elsewhere; test cols = per-fold "
                         f"test preds). No pseudo-labels, no collapsed OOF. Removes the "
                         f"single-fold-vs-bagged-mean train/test mismatch. Unconstrained Optuna, "
                         f"{cfg['n_trials']} trials, full-data 3-fold inner CV. "
                         f"scripts/run_perfold_aggressive.py.",
                "parent_run_id": cfg["parent"], "save_models": False,
                "data_version": DATA_VERSION,
            }
            result = run_cv_experiment(run_config, X, y, X_test, feats)   # no pseudo
            save_experiment(result, submit=True)
            print(f"  saved + submitted: OOF ROC-AUC {result['oof_roc_auc']:.6f}\n", flush=True)
        except Exception as e:
            print(f"  ERROR on {name}: {type(e).__name__}: {e}\n", flush=True)


if __name__ == "__main__":
    main()
